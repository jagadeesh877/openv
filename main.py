"""
OpenEnv — FastAPI Application & Baseline Agent
===============================================
Serves the environment over HTTP on port 7860 (Hugging Face Spaces compatible).

Endpoints:
  GET  /                   — Environment info
  POST /reset              — Reset environment for a given task
  POST /step               — Apply action, get step result
  GET  /state              — Current environment state
  GET  /run-baseline       — Run baseline agent, return scores
  GET  /tasks              — List available tasks
  GET  /docs               — Swagger UI (auto-generated)
"""
from __future__ import annotations

import json
import sys
import os
import time
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

# ── Logging Setup ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("openenv")

# ── Hugging Face Secrets Support ─────────────────────────────────────────

# Hugging Face Secrets Support
# Priority: API_KEY > HF_TOKEN > OPENAI_API_KEY  (mirrors inference.py)
api_key = (os.getenv("API_KEY") or os.getenv("HF_TOKEN")
           or os.getenv("OPENAI_API_KEY"))
api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Log configuration status (for HF Spaces debugging)
if api_key:
    log.info(f"API Key detected (length: {len(api_key)})")
else:
    log.warning("No API key found! Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.")

log.info(f"API Base URL: {api_base}")
log.info(f"Model Name: {model_name}")

# ── Path setup so imports work regardless of CWD ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openenv_logic.environment import OpenEnvEnvironment, Action, Observation
from inference_engine import InferenceAgent, run_llm_benchmark_task

from openenv_logic.environment import OpenEnvEnvironment, Action, TaskID

# ── Path setup so imports work regardless of CWD ──────────────────────────

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenEnv AI Environment",
    description=(
        "Production-grade OpenEnv-compatible AI simulation environment.\n\n"
        "Three real-world tasks: Email Triage, Code Review, Meeting Scheduler.\n"
        "Each task includes partial-credit grading and a baseline agent."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(ROOT, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Global environment state ───────────────────────────────────────────────
_env: Optional[OpenEnvEnvironment] = None
_inference_agent = InferenceAgent()
_last_benchmark_results: Optional[Dict[str, Any]] = None


# ── Request/Response models ────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "email_triage"


class StepRequest(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}


class OptionalResetRequest(BaseModel):
    task_id: Optional[str] = "email_triage"


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", summary="Environment Dashboard")
def root():
    """Serve the beautiful dashboard UI."""
    index_path = os.path.join(ROOT, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": "OpenEnv AI Environment",
        "version": "1.0.0",
        "status": "static_not_found",
        "hint": "Check /docs for API description."
    }


@app.get("/health", summary="Health check — instant response for cold-start probes")
def health():
    """Lightweight health check endpoint. Always returns 200 immediately."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/ping", summary="Ping — alias health check")
def ping():
    """Alias for /health — responds instantly to keep-alive pings."""
    return {"pong": True}


_TASK_GRADER_META = {
    "email_triage": {
        "class": "EmailGrader",
        "module": "graders.email_grader",
        "score_range": [0.0, 1.0],
        "partial_credit": True,
    },
    "code_review": {
        "class": "CodeGrader",
        "module": "graders.code_grader",
        "score_range": [0.0, 1.0],
        "partial_credit": True,
    },
    "meeting_scheduler": {
        "class": "MeetingGrader",
        "module": "graders.meeting_grader",
        "score_range": [0.0, 1.0],
        "partial_credit": True,
    },
}


@app.get("/tasks", summary="List available tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "email_triage",
                "title": "Email Triage System",
                "description": "Prioritize, respond to, or delete 8 realistic work emails.",
                "difficulty": "easy",
                "max_steps": 30,
                "reward_range": [-1.0, 1.0],
                "grader": _TASK_GRADER_META["email_triage"],
            },
            {
                "id": "code_review",
                "title": "Code Review Assistant",
                "description": "Detect bugs, suggest fixes, set severity, approve/reject 6 code snippets.",
                "difficulty": "medium",
                "max_steps": 30,
                "reward_range": [-1.0, 1.0],
                "grader": _TASK_GRADER_META["code_review"],
            },
            {
                "id": "meeting_scheduler",
                "title": "Meeting Scheduler",
                "description": "Schedule 10 meetings resolving conflicts and optimising preferences.",
                "difficulty": "hard",
                "max_steps": 30,
                "reward_range": [-1.0, 1.0],
                "grader": _TASK_GRADER_META["meeting_scheduler"],
            },
        ]
    }


@app.get("/graders", summary="List graders for all tasks")
def list_graders():
    """Return grader metadata for every task — required by OpenEnv validator."""
    return {
        "graders": [
            {"task_id": task_id, **meta}
            for task_id, meta in _TASK_GRADER_META.items()
        ]
    }


@app.get("/score", summary="Get final score of current episode")
def get_score():
    """Return the final graded score for the current episode."""
    global _env
    if _env is None:
        return {"final_score": 0.0, "status": "not_initialized"}
    return {
        "final_score": round(_env.final_score(), 4),
        "status": "ok",
    }


@app.post("/reset", summary="Reset the environment")
def reset(request: ResetRequest = ResetRequest()):
    global _env
    log.info(f"[UI CALL] /reset request for task: {request.task_id}")
    try:
        task_id = request.task_id or "email_triage"
        _env = OpenEnvEnvironment(task_id=task_id)
        obs = _env.reset()
        log.info(f"Environment reset successfully for: {task_id}")
        return obs.model_dump()
    except Exception as e:
        log.error(f"Reset error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", summary="Apply an action")
def step(request: StepRequest):
    global _env
    log.info(f"[UI CALL] /step -> {request.action_type}")
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. POST /reset first.")
    try:
        action = Action(action_type=request.action_type, parameters=request.parameters)
        result = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", summary="Current environment state")
def state():
    global _env
    if _env is None:
        return {"status": "not_initialized", "hint": "POST /reset to begin."}
    return _env.state()


@app.get("/run-baseline", summary="Start baseline evaluation in background")
async def run_baseline(background_tasks: BackgroundTasks):
    global _inference_agent, _last_benchmark_results
    log.info("[UI CALL] /run-baseline (Background)")
    if _inference_agent.is_running:
        return {"status": "already_running", "message": "Benchmark is currently in progress."}
    
    agent_status = _inference_agent # Reuse status tracking
    agent_status.is_running = True
    agent_status.current_status = "Running baseline..."

    async def wrapped_baseline():
        global _last_benchmark_results
        try:
            log.info("Starting baseline background task...")
            # run_baseline_agent is synchronous, run in thread
            results = await asyncio.to_thread(run_baseline_agent)
            avg_score = round(sum(r["final_score"] for r in results.values()) / len(results), 4) if results else 0
            
            _last_benchmark_results = {
                "baseline_agent": "rule-based",
                "results": results,
                "average_score": avg_score,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            log.info("Baseline background task complete.")
        finally:
            agent_status.is_running = False
            agent_status.current_status = "idle"

    background_tasks.add_task(wrapped_baseline)
    return {"status": "started", "message": "Baseline evaluation started in background."}


@app.get("/benchmark-results", summary="Get the results of the last benchmark run")
def get_benchmark_results():
    global _last_benchmark_results
    if _last_benchmark_results is None:
        raise HTTPException(status_code=404, detail="No benchmark results found.")
    return _last_benchmark_results


@app.post("/run-llm-benchmark", summary="Start full LLM-based evaluation in background")
async def run_llm_benchmark(background_tasks: BackgroundTasks):
    global _inference_agent, _last_benchmark_results
    log.info("[UI CALL] /run-llm-benchmark (Background)")
    if _inference_agent.is_running:
        return {"status": "already_running", "message": "Benchmark is currently in progress."}
    
    async def wrapped_task():
        global _last_benchmark_results
        try:
            results = await run_llm_benchmark_task(_inference_agent, "http://localhost:7860")
            avg_score = 0.0
            if results and len(results) > 0:
                avg_score = round(sum(r.get("final_score", 0.0) for r in results.values()) / len(results), 4)
            
            _last_benchmark_results = {
                "baseline_agent": "llm-agent (llama-3.1)",
                "results": results,
                "average_score": avg_score,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:
            log.error(f"Error in wrapped inference task: {e}")

    background_tasks.add_task(wrapped_task)
    return {"status": "started", "message": "LLM evaluation started in background."}


@app.get("/inference-status", summary="Get status of the running LLM agent")
def inference_status():
    global _inference_agent
    return {
        "is_running": _inference_agent.is_running,
        "current_task": _inference_agent.current_task,
        "status_message": _inference_agent.current_status
    }


@app.get("/baseline-suggestion", summary="Get a suggestion from the baseline agent for current state")
def baseline_suggestion():
    """Returns the 'best' next action for the current environment state."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    if _env._done:
        return {"action": "finish", "parameters": {}}

    state = _env.state()
    task_id = state["task_id"]
    obs = _env._build_observation()
    ctx = obs.context

    if task_id == "email_triage":
        action = _suggest_email_action(ctx)
    elif task_id == "code_review":
        action = _suggest_code_action(ctx)
    elif task_id == "meeting_scheduler":
        action = _suggest_meeting_action(ctx)
    else:
        action = "skip"

    return {"action": action, "parameters": {}}


def _suggest_email_action(ctx: Dict) -> str:
    sender = ctx.get("from", "").lower()
    subject = ctx.get("subject", "").lower()
    body = ctx.get("body_preview", "").lower()
    full_text = f"{sender} {subject} {body}"
    
    SPAM_SIGNALS = {"promo", "flash sale", "survey", "noreply@survey", "noreply@promo", "deals.xyz"}
    if any(sig in full_text for sig in SPAM_SIGNALS):
        return "delete_email"
    
    if "ceo@" in sender: return "respond_confirm_attendance"
    if "urgent" in subject or "outage" in subject or "degraded" in subject:
        return "respond_escalate_incident"
    if "board meeting" in subject or "confirm attendance" in subject:
        return "respond_confirm_attendance"
    if "compliance" in subject or "legal" in subject:
        return "respond_request_legal_review"
    
    return "prioritize_medium"


def _suggest_code_action(ctx: Dict) -> str:
    stage = ctx.get("stage", "detect_bug")
    code = ctx.get("code", "").lower()
    title = ctx.get("title", "").lower()
    combined = code + " " + title

    # Minimal heuristic mapping
    if "sql" in combined: bug, fix, sev = "security_vulnerability", "parameterized_query", "critical"
    elif "thread" in combined: bug, fix, sev = "concurrency_issue", "add_lock_or_atomic", "critical"
    elif "open(" in combined: bug, fix, sev = "resource_leak", "use_context_manager", "major"
    elif "lst=[]" in combined: bug, fix, sev = "logic_error", "use_none_default", "minor"
    elif "int(" in combined: bug, fix, sev = "error_handling", "add_try_except", "major"
    else: bug, fix, sev = "logic_error", "boundary_correction", "major"

    if stage == "detect_bug": return f"detect_{bug}"
    if stage == "suggest_fix": return f"fix_{fix}"
    if stage == "set_severity": return f"severity_{sev}"
    if stage == "verdict": return "reject_code"
    return "finish_review"


def _suggest_meeting_action(ctx: Dict) -> str:
    from tasks.meeting_scheduler import ALL_SLOTS
    preferred = ctx.get("preferred_slots", [])
    duration = ctx.get("duration_slots", 1)
    
    if preferred:
        for slot in preferred:
            if slot in ALL_SLOTS:
                idx = ALL_SLOTS.index(slot)
                if idx + duration <= len(ALL_SLOTS):
                    return f"schedule_{slot.replace(':', '')}"
    
    if ctx.get("must_schedule", False):
        return f"schedule_{ALL_SLOTS[0].replace(':', '')}"
    
    return "cancel_meeting"


# ── Baseline Agent ─────────────────────────────────────────────────────────

def run_baseline_agent() -> Dict[str, Any]:
    """
    Rule-based baseline agent that plays through all three tasks.
    Uses deterministic heuristics derived from observable context.
    Returns final scores per task.
    """
    results = {}

    # ── Task 1: Email Triage ──────────────────────────────────────────────
    results["email_triage"] = _run_email_triage_agent()

    # ── Task 2: Code Review ───────────────────────────────────────────────
    results["code_review"] = _run_code_review_agent()

    # ── Task 3: Meeting Scheduler ─────────────────────────────────────────
    results["meeting_scheduler"] = _run_meeting_scheduler_agent()

    return results


def _run_email_triage_agent() -> Dict[str, Any]:
    """
    Email heuristics:
    - If 'URGENT' in subject → respond_escalate or respond_confirm_attendance
    - If from CEO/client → prioritize_high
    - If 'promo'/'survey'/'flash' in sender/subject → delete
    - Otherwise → prioritize by category
    """
    env = OpenEnvEnvironment(task_id="email_triage")
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    transcript = []

    SENDER_PRIORITY_MAP = {
        "ceo@": ("respond_confirm_attendance", "high"),
        "client": ("respond_request_legal_review", "high"),
        "dev.team": ("respond_escalate_incident", "high"),
        "hr@": ("prioritize_medium", "medium"),
        "ops.team": ("prioritize_low", "low"),
        "newsletter": ("prioritize_low", "low"),
    }
    SPAM_SIGNALS = {"promo", "flash sale", "survey", "noreply@survey", "noreply@promo", "deals.xyz"}

    while True:
        if obs.done:
            break
        ctx = obs.context
        sender = ctx.get("from", "").lower()
        subject = ctx.get("subject", "").lower()
        body = ctx.get("body_preview", "").lower()

        # Detect spam
        full_text = sender + " " + subject + " " + body
        is_spam = any(sig in full_text for sig in SPAM_SIGNALS)

        if is_spam:
            action_type = "delete_email"
        else:
            # Match sender heuristics
            action_type = "prioritize_medium"  # default
            for pattern, (act, _) in SENDER_PRIORITY_MAP.items():
                if pattern in sender:
                    action_type = act
                    break
            # Urgency keyword override
            if "urgent" in subject or "outage" in subject or "degraded" in subject:
                action_type = "respond_escalate_incident"
            elif "contract" in subject or "renewal" in subject:
                action_type = "respond_request_legal_review"
            elif "board meeting" in subject or "confirm attendance" in subject:
                action_type = "respond_confirm_attendance"
            elif "compliance" in subject or "legal" in subject:
                action_type = "respond_request_legal_review"
            elif "performance review" in subject:
                action_type = "prioritize_medium"
            elif "cost" in subject or "budget" in subject:
                action_type = "prioritize_low"
            elif "newsletter" in subject or "digest" in subject:
                action_type = "prioritize_low"

        action = Action(action_type=action_type, parameters={})
        result = env.step(action)

        total_reward += result.reward.value
        steps += 1
        transcript.append({
            "step": steps,
            "email_id": ctx.get("email_id"),
            "action": action_type,
            "reward": result.reward.value,
        })
        obs = result.observation
        if result.done:
            break

    return {
        "task": "email_triage",
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(env.final_score(), 4),
        "transcript": transcript,
    }


def _run_code_review_agent() -> Dict[str, Any]:
    """
    Code review heuristics:
    - Keyword scan of code/title to classify bug and fix
    - Always reject (none of the snippets should be approved)
    """
    env = OpenEnvEnvironment(task_id="code_review")
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    transcript = []

    # Maps indicator keywords to (bug_category, fix_category, severity)
    CODE_HEURISTICS = [
        (["sql", "f\"select", "f'select", "query", "username"], "security_vulnerability", "parameterized_query", "critical"),
        (["threading", "thread", "global counter", "counter +="], "concurrency_issue", "add_lock_or_atomic", "critical"),
        (["open(", "f = open", "file never"], "resource_leak", "use_context_manager", "major"),
        (["lst=[]", "mutable default", "def append_item"], "logic_error", "use_none_default", "minor"),
        (["int(value)", "parse_age", "no exception"], "error_handling", "add_try_except", "major"),
        (["paginate", "page_size", "end + 1", "off-by-one"], "logic_error", "boundary_correction", "major"),
    ]

    while True:
        if obs.done:
            break
        ctx = obs.context
        code = ctx.get("code", "").lower()
        title = ctx.get("title", "").lower()
        stage = ctx.get("stage", "detect_bug")

        # Build a combined text for matching
        combined = code + " " + title

        # Identify best match
        matched_bug = "logic_error"
        matched_fix = "boundary_correction"
        matched_severity = "major"
        best_hits = 0
        for keywords, bug, fix, severity in CODE_HEURISTICS:
            hits = sum(1 for kw in keywords if kw in combined)
            if hits > best_hits:
                best_hits = hits
                matched_bug = bug
                matched_fix = fix
                matched_severity = severity

        if stage == "detect_bug":
            action_type = f"detect_{matched_bug}"
        elif stage == "suggest_fix":
            action_type = f"fix_{matched_fix}"
        elif stage == "set_severity":
            action_type = f"severity_{matched_severity}"
        elif stage == "verdict":
            action_type = "reject_code"  # all snippets have bugs → reject
        else:
            action_type = "finish_review"

        action = Action(action_type=action_type, parameters={})
        result = env.step(action)

        total_reward += result.reward.value
        steps += 1
        transcript.append({
            "step": steps,
            "snippet_id": ctx.get("snippet_id"),
            "stage": stage,
            "action": action_type,
            "reward": result.reward.value,
        })
        obs = result.observation
        if result.done:
            break

    return {
        "task": "code_review",
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(env.final_score(), 4),
        "transcript": transcript,
    }


def _run_meeting_scheduler_agent() -> Dict[str, Any]:
    """
    Meeting scheduler heuristics:
    - Always try the first preferred slot
    - If meeting is mandatory and prefers specific slot, schedule it
    - If optional and conflicts exist, cancel it
    """
    from tasks.meeting_scheduler import ALL_SLOTS

    env = OpenEnvEnvironment(task_id="meeting_scheduler")
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    transcript = []

    while True:
        if obs.done:
            break
        ctx = obs.context
        preferred = ctx.get("preferred_slots", [])
        must_schedule = ctx.get("must_schedule", True)
        duration = ctx.get("duration_slots", 1)
        conflicts = ctx.get("conflicts_in_calendar", 0)

        action_type = None

        # Find best available start slot from preferred
        if preferred:
            for slot in preferred:
                if slot in ALL_SLOTS:
                    idx = ALL_SLOTS.index(slot)
                    if idx + duration <= len(ALL_SLOTS):
                        slot_label = slot.replace(":", "")
                        action_type = f"schedule_{slot_label}"
                        break

        if not action_type:
            # Try any available slot window
            if must_schedule:
                for i in range(len(ALL_SLOTS) - duration + 1):
                    slot_label = ALL_SLOTS[i].replace(":", "")
                    action_type = f"schedule_{slot_label}"
                    break
            else:
                action_type = "cancel_meeting"

        if not action_type:
            action_type = "finalize_schedule"

        action = Action(action_type=action_type, parameters={})
        result = env.step(action)

        total_reward += result.reward.value
        steps += 1
        transcript.append({
            "step": steps,
            "meeting_id": ctx.get("meeting_id"),
            "action": action_type,
            "reward": result.reward.value,
        })
        obs = result.observation
        if result.done:
            break

    return {
        "task": "meeting_scheduler",
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(env.final_score(), 4),
        "transcript": transcript,
    }


# ── Entry point (CLI) ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenEnv Baseline Agent + Server")
    parser.add_argument("--mode", choices=["server", "agent"], default="server",
                        help="Run as API server or run baseline agent only.")
    args = parser.parse_args()

    if args.mode == "agent":
        print("\n" + "=" * 60)
        print("  OpenEnv Baseline Agent - Running All Tasks")
        print("=" * 60)
        results = run_baseline_agent()
        for task_id, result in results.items():
            print(f"\n" + "-" * 50)
            print(f"  Task: {task_id.upper()}")
            print(f"  Steps taken:   {result['steps']}")
            print(f"  Total reward:  {result['total_reward']}")
            print(f"  Final score:   {result['final_score']}")
        avg = sum(r["final_score"] for r in results.values()) / len(results)
        print(f"\n" + "=" * 60)
        print(f"  AVERAGE SCORE: {round(avg, 4)}")
        print("=" * 60 + "\n")
    else:
        import uvicorn
        port = int(os.getenv("PORT", 7860))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
