"""
OpenEnv Core Environment
========================
Production-grade AI environment with Pydantic schema validation,
deterministic transitions, and real-world task simulation.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schema Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Structured state observation returned to the agent."""
    task_id: str = Field(..., description="Active task identifier")
    step: int = Field(..., description="Current step number (0-indexed)")
    state_description: str = Field(..., description="Human-readable state summary")
    available_actions: List[str] = Field(..., description="List of valid action names")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task-specific structured context")
    done: bool = Field(default=False, description="Whether the episode is finished")


class Action(BaseModel):
    """Agent action submitted to the environment."""
    action_type: str = Field(..., description="Action name (must be in available_actions)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")


class Reward(BaseModel):
    """Reward signal after each step."""
    value: float = Field(..., ge=-1.0, le=1.0, description="Step reward in [-1, 1]")
    cumulative: float = Field(..., description="Total cumulative reward so far")
    reason: str = Field(..., description="Human-readable reason for this reward")


class StepResult(BaseModel):
    """Full result of a single environment step."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

class TaskID(str, Enum):
    EMAIL_TRIAGE = "email_triage"
    CODE_REVIEW = "code_review"
    MEETING_SCHEDULER = "meeting_scheduler"


# ---------------------------------------------------------------------------
# OpenEnv Environment
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi] so Pydantic Reward validation never fails."""
    return max(lo, min(hi, v))


class OpenEnvEnvironment:
    """
    Main environment class that orchestrates tasks, routes agent actions,
    maintains state, and computes rewards.
    """

    MAX_STEPS_PER_TASK = 30
    INVALID_ACTION_PENALTY = -0.15
    REPEAT_ACTION_PENALTY = -0.10
    EFFICIENCY_BONUS_THRESHOLD = 0.7  # reward bonus if completed under 70% of max steps

    def __init__(self, task_id: str = TaskID.EMAIL_TRIAGE):
        # Lazy import to avoid circular dependencies
        from tasks.email_triage import EmailTriageTask
        from tasks.code_review import CodeReviewTask
        from tasks.meeting_scheduler import MeetingSchedulerTask
        from graders.email_grader import EmailGrader
        from graders.code_grader import CodeGrader
        from graders.meeting_grader import MeetingGrader

        self._task_map = {
            TaskID.EMAIL_TRIAGE: (EmailTriageTask, EmailGrader),
            TaskID.CODE_REVIEW: (CodeReviewTask, CodeGrader),
            TaskID.MEETING_SCHEDULER: (MeetingSchedulerTask, MeetingGrader),
        }

        self.task_id = TaskID(task_id)
        self._task: Any = None
        self._grader: Any = None
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._action_history: List[str] = []
        self._start_time: float = 0.0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        task_cls, grader_cls = self._task_map[self.task_id]
        self._task = task_cls()
        self._grader = grader_cls(self._task)
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._action_history = []
        self._start_time = time.time()
        self._initialized = True

        initial_state = self._task.get_initial_state()
        return Observation(
            task_id=self.task_id.value,
            step=0,
            state_description=initial_state["description"],
            available_actions=initial_state["available_actions"],
            context=initial_state["context"],
            done=False,
        )

    def step(self, action: Action) -> StepResult:
        """Execute an action and return (observation, reward, done, info)."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1

        # --- Validate action ---
        current_valid = self._task.get_valid_actions()
        current_state = self._task.get_current_state()
        ctx = current_state.get("context", {})
        item_id = ctx.get("email_id") or ctx.get("snippet_id") or ctx.get("meeting_id") or "global"
        
        action_key = f"{item_id}:{action.action_type}:{str(sorted(action.parameters.items()))}"

        if action.action_type not in current_valid:
            reward_val = self.INVALID_ACTION_PENALTY
            self._cumulative_reward += reward_val
            obs = self._build_observation()
            rwd = Reward(
                value=_clamp(reward_val),
                cumulative=round(self._cumulative_reward, 4),
                reason=f"Invalid action '{action.action_type}'. Valid: {current_valid[:5]}...",
            )
            done = self._step_count >= self.MAX_STEPS_PER_TASK
            if done:
                self._done = True
            return StepResult(observation=obs, reward=rwd, done=done, info={"error": "invalid_action"})

        # --- Penalise repetition ---
        if self._action_history.count(action_key) >= 2:
            reward_val = self.REPEAT_ACTION_PENALTY
            self._cumulative_reward += reward_val
            obs = self._build_observation()
            rwd = Reward(
                value=_clamp(reward_val),
                cumulative=round(self._cumulative_reward, 4),
                reason=f"Repeated action penalised (seen {self._action_history.count(action_key)} times).",
            )
            done = self._step_count >= self.MAX_STEPS_PER_TASK
            if done:
                self._done = True
            return StepResult(observation=obs, reward=rwd, done=done, info={"warning": "repeated_action"})

        self._action_history.append(action_key)

        # --- Delegate to task ---
        task_result = self._task.apply_action(action.action_type, action.parameters)

        # --- Score via grader ---
        step_score = self._grader.score_step(action.action_type, action.parameters, task_result)
        reward_val = float(step_score)
        self._cumulative_reward += reward_val

        # --- Check completion ---
        task_done = task_result.get("done", False)
        max_steps_done = self._step_count >= self.MAX_STEPS_PER_TASK

        if task_done and not max_steps_done:
            efficiency_ratio = self._step_count / self.MAX_STEPS_PER_TASK
            if efficiency_ratio <= self.EFFICIENCY_BONUS_THRESHOLD:
                # Cap bonus so total step reward stays within [-1, 1]
                bonus = round(min(0.2, 0.2 * (1.0 - efficiency_ratio)), 4)
                self._cumulative_reward += bonus
                reward_val += bonus
                task_result["efficiency_bonus"] = bonus

        self._done = task_done or max_steps_done
        obs = self._build_observation(task_result)
        rwd = Reward(
            value=_clamp(round(reward_val, 4)),   # Always clamp before Pydantic
            cumulative=round(self._cumulative_reward, 4),
            reason=task_result.get("reason", "Action applied."),
        )

        return StepResult(
            observation=obs,
            reward=rwd,
            done=self._done,
            info={
                "step": self._step_count,
                "steps_remaining": self.MAX_STEPS_PER_TASK - self._step_count,
                "task_result": task_result,
                "elapsed_seconds": round(time.time() - self._start_time, 2),
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return current full state snapshot."""
        if not self._initialized:
            return {"status": "not_initialized"}
        return {
            "task_id": self.task_id.value,
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "final_score": round(self.final_score(), 4),
            "done": self._done,
            "action_history_count": len(self._action_history),
            "task_state": self._task.export_state() if self._task else {},
        }

    def final_score(self) -> float:
        """Compute final normalized score from grader."""
        if not self._initialized or not self._grader:
            return 0.0
        return self._grader.final_score()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self, task_result: Optional[Dict] = None) -> Observation:
        current = self._task.get_current_state()
        context = dict(current.get("context", {}))
        # Inject universal fields so agents always know episode progress
        context["steps_remaining"] = self.MAX_STEPS_PER_TASK - self._step_count
        context["step"] = self._step_count
        return Observation(
            task_id=self.task_id.value,
            step=self._step_count,
            state_description=current["description"],
            available_actions=self._task.get_valid_actions(),
            context=context,
            done=self._done,
        )
