"""
Email Triage Grader
====================
Evaluates agent decisions against ground truth for each email.

Partial credit breakdown per email (sum = 1.0 per email):
- Correct action category (priority/delete/respond): 0.40
- Correct priority level (if applicable):            0.30
- Correct response type (if applicable):             0.30

Final score = average over all 8 emails, in [0.0, 1.0].
"""
from __future__ import annotations

from typing import Any, Dict


class EmailGrader:
    def __init__(self, task):
        self._task = task
        self._step_scores: Dict[str, float] = {}

    def score_step(self, action_type: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Return per-step reward in [-0.2, 0.4]."""
        if result.get("done") and action_type == "finish":
            return 0.0

        email_id = result.get("email_id")
        if not email_id:
            return 0.0

        decision = result.get("decision", {})
        gt = result.get("ground_truth", {})

        if not gt:
            return 0.0

        score = 0.0

        # Action category correctness (0.40)
        agent_action = self._normalise_action(decision.get("action", ""))
        gt_action = gt.get("action", "")
        if agent_action == gt_action:
            score += 0.40
        elif self._similar_action(agent_action, gt_action):
            score += 0.25 # Increased partial credit for triage vs respond

        # Priority correctness (0.30)
        # We give these points if the GT has a priority, regardless of agent's action category
        if gt.get("priority"):
            if decision.get("priority") == gt["priority"]:
                score += 0.30
            elif self._adjacent_priority(decision.get("priority"), gt["priority"]):
                score += 0.10

        # Response type correctness (0.30)
        # We give these points if the GT has a response, regardless of agent's action category 
        if gt.get("response"):
            if decision.get("response") == gt["response"]:
                score += 0.30

        # Penalise wrongly deleting a real email
        if agent_action == "delete" and not self._is_spam(email_id):
            score -= 0.20

        # Clamp per-email score between 0 and 1 for the final tally
        # Note: step reward can be negative, but email credit is non-negative
        self._step_scores[email_id] = max(0.0, min(1.0, score))
        
        # Return raw score as reward (can be negative due to penalties)
        return round(max(-1.0, min(1.0, score)), 4)

    def final_score(self) -> float:
        """Normalised average score over all emails [0.0, 1.0]."""
        inbox = self._task.get_inbox()
        if not inbox:
            return 0.0

        total_score = sum(self._step_scores.values())
        return round(total_score / len(inbox), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_action(self, action: str) -> str:
        if action in ("delete", "deleted"):
            return "delete"
        if action in ("prioritize", "priority"):
            return "prioritize"
        if action in ("respond", "response"):
            return "respond"
        return action

    def _similar_action(self, got: str, expected: str) -> bool:
        """Partial credit for close misses."""
        return (
            (got == "respond" and expected == "prioritize") or
            (got == "prioritize" and expected == "respond")
        )

    def _adjacent_priority(self, got: str, expected: str) -> bool:
        order = ["low", "medium", "high"]
        if got not in order or expected not in order:
            return False
        return abs(order.index(got) - order.index(expected)) == 1

    def _is_spam(self, email_id: str) -> bool:
        for e in self._task.get_inbox():
            if e["id"] == email_id:
                return e.get("is_spam", False)
        return False
