"""
OpenEnv Inference Script
========================
Compliant inference script for the OpenEnv benchmark evaluation.
Handles Email Triage, Code Review, and Meeting Scheduler tasks.

Required Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (or provider API key).
- API_BASE_URL: The API endpoint for the LLM (defaults to OpenAI).
- MODEL_NAME: The model identifier (optional, defaults to gpt-4o-mini).
- TASK_NAME: one of [email_triage, code_review, meeting_scheduler].
"""

import asyncio
import os
import textwrap
import logging
import argparse
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

from openai import OpenAI
from pydantic import BaseModel

# Internal imports from our environment
from env.environment import OpenEnvEnvironment, Action, Observation

# --- Configuration ---
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"

# Default to gpt-4o-mini for OpenAI/Groq or similar providers
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODEL
TASK_NAME = os.getenv("TASK_NAME") or os.getenv("OPENENV_TASK") or "email_triage"

MAX_STEPS = 30
TEMPERATURE = 0.1  # Low temperature for best format adherence
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# --- Logging Helpers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# --- Task System Prompts ---
TASK_PROMPTS = {
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
    )
}


# --- Environment Async Wrapper ---
class OpenEnvWrapper:
    """Wraps synchronous environment for async compatibility."""
    def __init__(self, task_id: str):
        self.env = OpenEnvEnvironment(task_id=task_id)

    async def reset(self):
        return self.env.reset()

    async def step(self, action_type: str, parameters: Dict = {}):
        action = Action(action_type=action_type, parameters=parameters)
        return self.env.step(action)

    async def close(self):
        pass 

    def get_final_score(self):
        return self.env.final_score()


# --- LLM Interaction ---
def extract_action(text: str, valid_actions: List[str]) -> str:
    """Robustly extract the action string from LLM responses containing reasoning."""
    import re
    
    # 1. Remove reasoning blocks if model used <think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Check if any valid action is explicitly present in the text
    # We prioritize exact matches first for robustness
    for action in valid_actions:
        # Check for whole word match to avoid substrings
        if re.search(rf"\b{re.escape(action)}\b", text):
            return action

    # 3. If no exact match found, try to find the last snake_case word
    # (Many models put the action at the end)
    lines = [l.strip().strip("*`\"'#") for l in text.split("\n") if l.strip()]
    if lines:
        for line in reversed(lines):
            # Potential snake_case candidate
            candidate = re.sub(r'[^a-z0-9_]', '', line.lower())
            for valid in valid_actions:
                if candidate == valid.lower():
                    return valid

    # 4. Fallback to first available action to avoid crashing
    return valid_actions[0]

def get_model_action(client: OpenAI, step: int, obs: Observation, history: List[Dict]) -> str:
    """Query the LLM for the next action with retry logic."""
    import time
    task_id = obs.task_id
    sys_prompt = TASK_PROMPTS.get(task_id, "Output an action.")
    
    ctx = obs.context or {}
    actions_list = obs.available_actions
    
    # Format a readable history string that includes rewards and reasons
    history_str = "None"
    if history:
        history_lines = []
        for i, h in enumerate(history[-10:]):
            history_lines.append(f"- Step {i+1}: {h['action']} -> Result: {h['reward']} ({h['reason']})")
        history_str = "\n".join(history_lines)

    user_msg = (
        f"CURRENT STEP: {step}\n"
        f"CONTEXT: {obs.state_description}\n"
        f"AVAILABLE ACTIONS: {', '.join(actions_list)}\n\n"
        f"EPISODE HISTORY:\n{history_str}\n\n"
        "Important Rule: If an action in history has a NEGATIVE reward, do NOT repeat it for the same context.\n"
        "Select the next action string:"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content or ""
            return extract_action(content, actions_list)
        except Exception as exc:
            if "429" in str(exc) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logging.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            logging.error(f"LLM API Error: {exc}")
            return actions_list[0]
    return actions_list[0]


async def run_evaluation(task_id: str, client: OpenAI) -> float:
    """Run a single evaluation episode for a given task."""
    env = OpenEnvWrapper(task_id=task_id)
    history: List[Dict] = []  # Stores structured outcomes
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env="openenv_v1", model=MODEL_NAME)

    try:
        obs = await env.reset()
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_type = get_model_action(client, step, obs, history)
            result = await env.step(action_type)
            
            rwd = result.reward.value
            reason = result.reward.reason
            obs = result.observation
            
            # Store structured outcome
            history.append({
                "action": action_type,
                "reward": rwd,
                "reason": reason
            })
            
            rewards.append(rwd)
            steps_taken = step
            
            log_step(step=step, action=action_type, reward=rwd, done=result.done, error=None)

            if result.done:
                break

        final_score = env.get_final_score()
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
        return final_score

    except Exception as e:
        logging.critical(f"Task Execution Error: {e}")
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return 0.0
    finally:
        await env.close()

async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv Benchmark Inference Script")
    parser.add_argument("--task", type=str, default=TASK_NAME, 
                        help="Task ID to run (email_triage, code_review, meeting_scheduler, or ALL)")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    tasks_to_run = []
    if args.task.upper() == "ALL":
        tasks_to_run = ["email_triage", "code_review", "meeting_scheduler"]
    else:
        tasks_to_run = [args.task]

    print("\n" + "=" * 60)
    print(f"  OPENENV BENCHMARK - Model: {MODEL_NAME}")
    print("=" * 60)

    results = {}
    for tid in tasks_to_run:
        score = await run_evaluation(tid, client)
        results[tid] = score

    # Final Summary Table
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
