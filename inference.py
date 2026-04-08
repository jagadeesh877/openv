"""
OpenEnv Inference Script
========================
Compliant inference script for the OpenEnv benchmark evaluation.
Handles Email Triage, Code Review, and Meeting Scheduler tasks.

This script connects to the running environment server via HTTP
(no local package imports required — fully self-contained).

Required Environment Variables:
- OPENAI_API_KEY: Your OpenAI/Groq API key.
- API_BASE_URL: The LLM endpoint (defaults to Groq).
- MODEL_NAME: The model identifier (optional).
- TASK_NAME / OPENENV_TASK: one of [email_triage, code_review, meeting_scheduler].
- ENV_BASE_URL: Base URL of the running env server (default: http://localhost:7860).
"""

import asyncio
import os
import re
import sys
import time
import textwrap
import logging
import argparse
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Optionally load .env (ignored if file absent — safe in validator sandbox)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# ---------------------------------------------------------------------------
# HTTP client — prefer httpx, fall back to urllib (stdlib)
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx

    def _http_post(url: str, payload: dict) -> dict:
        with _httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    def _http_get(url: str) -> dict:
        with _httpx.Client(timeout=30.0) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()

except ImportError:
    import json
    import urllib.request
    import urllib.error

    def _http_post(url: str, payload: dict) -> dict:  # type: ignore[misc]
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _http_get(url: str) -> dict:  # type: ignore[misc]
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())

# ---------------------------------------------------------------------------
# LLM client — prefer openai SDK, fall back to urllib
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("OPENAI_API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "qwen/qwen3-32b"
TASK_NAME    = os.getenv("TASK_NAME") or os.getenv("OPENENV_TASK") or "email_triage"
ENV_BASE_URL = (os.getenv("ENV_BASE_URL") or "http://localhost:7860").rstrip("/")

MAX_STEPS               = 30
TEMPERATURE             = 0.1
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Logging helpers (OpenEnv standard format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------
TASK_PROMPTS: Dict[str, str] = {
    "email_triage": (
        "You are an expert Executive Assistant. Your goal is to triage an inbox of 8 emails.\n"
        "For each email, you must select one of the following actions:\n"
        "1. PRIORITIZE: Use 'prioritize_high/medium/low' for real professional emails.\n"
        "2. RESPOND: Use 'respond_...' if a specific action (like confirming a meeting) is requested.\n"
        "3. DELETE: Use 'delete_email' for ANY spam, flash sales, generic newsletters, or surveys.\n\n"
        "IMPORTANT: Be aggressive with 'delete_email' for non-essential promotional content. "
        "Do NOT prioritize spam as 'low'; DELETE it instead.\n"
        "Rules:\n"
        "- Output ONLY the action string from the provided list.\n"
        "- No quotes, no markdown, no explanation."
    ),

    "code_review": textwrap.dedent("""
        You are a senior security engineer performing a code review.
        Goal: Review code for bugs in 4 stages (detect -> fix -> severity -> verdict).
        Rules:
        - Output ONLY the action string (e.g., 'detect_logic_error' or 'reject_code').
        - You MUST choose an action from the Available Actions list provided.
        - No reasoning, no chatter.
    """).strip(),

    "meeting_scheduler": (
        "You are an expert Strategic Scheduler. Your goal is to manage meeting requests.\n"
        "1. TOP PRIORITY: Pick a slot from the 'fully_free_slots_for_these_attendees' list.\n"
        "2. DURATION: If the meeting needs 2 slots, ensure you pick a starting slot that has enough following free slots.\n"
        "3. MANDATORY: Never cancel meetings with 'must_schedule: True'.\n"
        "Strategy: Always choose from the suggested free slots to avoid all conflicts."
    ),
}


# ---------------------------------------------------------------------------
# HTTP-based environment wrapper
# ---------------------------------------------------------------------------

class HTTPEnvWrapper:
    """Calls the running OpenEnv server via HTTP — no local imports needed."""

    def __init__(self, task_id: str, base_url: str = ENV_BASE_URL):
        self.task_id  = task_id
        self.base_url = base_url

    def reset(self) -> dict:
        return _http_post(f"{self.base_url}/reset", {"task_id": self.task_id})

    def step(self, action_type: str, parameters: dict = {}) -> dict:
        return _http_post(f"{self.base_url}/step",
                          {"action_type": action_type, "parameters": parameters})

    def state(self) -> dict:
        return _http_get(f"{self.base_url}/state")

    def get_final_score(self) -> float:
        try:
            s = self.state()
            return float(s.get("final_score", 0.0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# LLM action selector
# ---------------------------------------------------------------------------

def extract_action(text: str, valid_actions: List[str]) -> str:
    """Robustly extract the action string from LLM output."""
    # Strip reasoning blocks (<think>…</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Exact whole-word match
    for action in valid_actions:
        if re.search(rf"\b{re.escape(action)}\b", text):
            return action

    # Fuzzy: last snake_case token in text
    lines = [l.strip().strip("*`\"'#") for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        candidate = re.sub(r"[^a-z0-9_]", "", line.lower())
        for valid in valid_actions:
            if candidate == valid.lower():
                return valid

    return valid_actions[0]


def _llm_call_openai(messages: list) -> str:
    """Call LLM using the openai SDK."""
    client = _OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _llm_call_urllib(messages: list) -> str:
    """Call LLM via raw HTTP (stdlib urllib) — fallback when openai is absent."""
    import json, urllib.request
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API_BASE_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"] or ""


def get_model_action(step: int, obs: dict, history: List[Dict]) -> str:
    """Query the LLM for the next action with retry logic."""
    task_id = obs.get("task_id", TASK_NAME)
    sys_prompt = TASK_PROMPTS.get(task_id, "Output an action.")

    actions_list: List[str] = obs.get("available_actions", [])
    if not actions_list:
        return ""

    # Build readable history string
    history_str = "None"
    if history:
        lines = [f"- Step {i+1}: {h['action']} -> Result: {h['reward']} ({h['reason']})"
                 for i, h in enumerate(history[-10:])]
        history_str = "\n".join(lines)

    user_msg = (
        f"CURRENT STEP: {step}\n"
        f"CONTEXT: {obs.get('state_description', '')}\n"
        f"AVAILABLE ACTIONS: {', '.join(actions_list)}\n\n"
        f"EPISODE HISTORY:\n{history_str}\n\n"
        "Important Rule: If an action in history has a NEGATIVE reward, do NOT repeat it for the same context.\n"
        "Select the next action string:"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_msg},
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if _OPENAI_AVAILABLE:
                content = _llm_call_openai(messages)
            else:
                content = _llm_call_urllib(messages)
            return extract_action(content, actions_list)
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                log.warning(f"Rate limit hit. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                log.error(f"LLM error: {exc}")
                break

    return actions_list[0]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def run_evaluation(task_id: str) -> float:
    """Run a single evaluation episode against the HTTP env server."""
    env = HTTPEnvWrapper(task_id=task_id)
    history: List[Dict] = []
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env="openenv_v1", model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_type = get_model_action(step, obs, history)
            result      = env.step(action_type)

            rwd    = result.get("reward", {}).get("value", 0.0)
            reason = result.get("reward", {}).get("reason", "")
            obs    = result.get("observation", obs)

            history.append({"action": action_type, "reward": rwd, "reason": reason})
            rewards.append(rwd)
            steps_taken = step

            log_step(step=step, action=action_type, reward=rwd,
                     done=result.get("done", False), error=None)

            if result.get("done", False):
                break

        final_score = env.get_final_score()
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
        return final_score

    except Exception as e:
        log.critical(f"Task execution error: {e}")
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return 0.0


async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Benchmark Inference Script")
    parser.add_argument(
        "--task", type=str, default=TASK_NAME,
        help="Task ID to run (email_triage, code_review, meeting_scheduler, or ALL)"
    )
    args = parser.parse_args()

    tasks_to_run: List[str]
    if args.task.upper() == "ALL":
        tasks_to_run = ["email_triage", "code_review", "meeting_scheduler"]
    else:
        tasks_to_run = [args.task]

    print("\n" + "=" * 60)
    print(f"  OPENENV BENCHMARK - Model: {MODEL_NAME}")
    print(f"  Env server : {ENV_BASE_URL}")
    print("=" * 60)

    results: Dict[str, float] = {}
    for tid in tasks_to_run:
        results[tid] = await run_evaluation(tid)

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'TASK':25} | {'SCORE':8} | {'STATUS':10}")
    print("-" * 70)
    for tid, score in results.items():
        status = "PASSED" if score >= SUCCESS_SCORE_THRESHOLD else "FAILED"
        print(f"  {tid:25} | {score:8.4f} | {status:10}")

    if len(results) > 1:
        avg = sum(results.values()) / len(results)
        print("-" * 70)
        print(f"  {'MEAN BENCHMARK SCORE':25} | {avg:8.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
