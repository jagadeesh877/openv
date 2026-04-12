import os
import re
import logging
import asyncio
from typing import List, Dict, Any
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (Shared with inference.py)
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "EMPTY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "gpt-4o-mini"
ENV_BASE_URL = (os.getenv("ENV_BASE_URL") or "http://localhost:7860").rstrip("/")

MAX_STEPS               = 25
TEMPERATURE             = 0.1
MAX_TOKENS              = 512

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inference_engine")

PROMPTS = {
    "email_triage": (
        "You are an expert Executive Assistant. Your task is to triage an inbox.\n"
        "You must output ONLY the action name from the valid actions list.\n"
        "Valid Actions: prioritize_high, prioritize_medium, prioritize_low, delete_email, "
        "respond_confirm_attendance, respond_escalate_incident, respond_request_legal_review, skip_email\n"
        "Rules:\n1. Delete SPAM immediately.\n2. Prioritize CEO/Client emails as HIGH.\n3. Escalation for incidents."
    ),
    "code_review": (
        "You are a Senior Security Engineer. Review the code snippet.\n"
        "Identify the bug category and suggested fix in 4 stages.\n"
        "Valid Actions: detect_logic_error, detect_security_vulnerability, detect_concurrency_issue, detect_resource_leak, detect_error_handling, "
        "fix_parameterized_query, fix_add_lock_or_atomic, fix_use_context_manager, fix_use_none_default, fix_add_try_except, fix_boundary_correction, "
        "severity_critical, severity_major, severity_minor, verdict, finish_review"
    ),
    "meeting_scheduler": (
        "You are a Scheduling Coordinator. Resolve calendar conflicts.\n"
        "Valid Actions: schedule_HHMM (e.g. schedule_0900), cancel_meeting, finalize_schedule\n"
        "Rules: 1. No double-booking. 2. Respect 'preferred_slots'. 3. Never cancel 'must_schedule' meetings."
    ),
}

class InferenceAgent:
    def __init__(self):
        # Refresh config from environment
        # Prioritize OPENAI_API_KEY (Groq) first
        self.api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "EMPTY"
        self.api_base = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
        self.model = os.getenv("MODEL_NAME") or "gpt-4o-mini"
        
        if self.api_key == "EMPTY":
            logging.warning("[AGENT] Initialized WITHOUT an API key. LLM evaluations will fail.")
        else:
            logging.info(f"[AGENT] Initialized with API key (length: {len(self.api_key)})")

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        self.is_running = False
        self.current_status = "idle"
        self.current_task = None

    def get_model_action(self, task_id: str, obs: Dict, history: List) -> str:
        """Selects an action using the LLM."""
        system_prompt = PROMPTS.get(task_id, "Choose the best action.")
        
        # Build history context
        history_str = "\n".join([f"Action: {h['action']} -> Reward: {h['reward']}" for h in history[-5:]])
        
        user_message = (
            f"Task: {task_id}\n"
            f"State: {obs.get('state_description')}\n"
            f"Available Actions: {', '.join(obs.get('available_actions', []))}\n"
            f"Context: {obs.get('context')}\n"
            f"Recent History:\n{history_str}\n\n"
            "Result (Action Name Only):"
        )

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        raw_text = resp.choices[0].message.content or ""
        return self._extract_action(raw_text, obs.get("available_actions", []))

    def _extract_action(self, text: str, valid_actions: List[str]) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        for action in valid_actions:
            if re.search(rf"\b{re.escape(action)}\b", text, re.IGNORECASE):
                return action
        return valid_actions[0] if valid_actions else "skip"

async def run_llm_benchmark_task(agent: InferenceAgent, env_url: str):
    """Runs a full 3-task benchmark in the background."""
    import httpx
    import traceback
    agent.is_running = True
    tasks = ["email_triage", "code_review", "meeting_scheduler"]
    results = {}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for tid in tasks:
                agent.current_task = tid
                agent.current_status = f"Evaluating {tid}..."
                log.info(f"LLM Agent: Starting task {tid}")
                
                # Reset
                await client.post(f"{env_url}/reset", json={"task_id": tid})
                history = []
                
                for step in range(MAX_STEPS):
                    # Get state
                    resp = await client.get(f"{env_url}/state")
                    state = resp.json()
                    obs = state.get("observation") or state
                    
                    if obs.get("done"):
                        break
                    
                    # Get AI action
                    try:
                        action = await asyncio.to_thread(agent.get_model_action, tid, obs, history)
                        log.info(f"LLM Agent ({tid}): Step {step} -> {action}")
                    except Exception as e:
                        log.error(f"LLM Agent Action Error: {e}")
                        action = "skip"
                    
                    # Step
                    step_resp = await client.post(f"{env_url}/step", json={"action_type": action, "parameters": {}})
                    step_data = step_resp.json()
                    
                    history.append({
                        "action": action, 
                        "reward": step_data.get("reward", {}).get("value", 0.0)
                    })
                    
                    if step_data.get("done"):
                        break
                
                # Record final results
                final_resp = await client.get(f"{env_url}/state")
                final_state = final_resp.json()
                
                results[tid] = {
                    "task": tid,
                    "steps": len(history),
                    "total_reward": round(sum(h['reward'] for h in history), 4),
                    "final_score": round(final_state.get("final_score", 0.0), 4),
                    "transcript": history
                }
                log.info(f"LLM Agent: Finished {tid} with score {results[tid]['final_score']}")

    except Exception as e:
        log.error(f"LLM Benchmark Global Error: {e}")
        traceback.print_exc()
    finally:
        agent.is_running = False
        agent.current_status = "idle"
        agent.current_task = None
        log.info("LLM Agent: Background task finished and unlocked.")
    return results
