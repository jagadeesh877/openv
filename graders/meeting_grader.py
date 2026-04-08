"""
Meeting Scheduler Grader
=========================
Evaluates the quality of the agent's produced schedule.

Scoring criteria:
1. Conflict-free scheduling:  40% — penalise each conflict slot
2. Must-schedule compliance:  30% — penalise cancelling mandatory meetings
3. Preference alignment:      20% — reward preferred slot usage
4. Priority weighting:         10% — high priority meetings correctly handled

Score operates at the step level (rewarding/penalising each decision),
and final_score() produces the overall calendar quality score.
"""
from __future__ import annotations

from typing import Any, Dict, List

PRIORITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


class MeetingGrader:
    def __init__(self, task):
        self._task = task
        self._step_rewards: List[float] = []

    def score_step(self, action_type: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Per-step reward for scheduling decisions."""
        if action_type == "finalize_schedule":
            return 0.0

        if "error" in result:
            reward = -0.10
            self._step_rewards.append(reward)
            return reward

        reward = 0.0

        # Cancellation penalty
        if action_type == "cancel_meeting":
            must = result.get("must_schedule", False)
            priority = result.get("cancelled_priority", "low")
            pw = PRIORITY_WEIGHT.get(priority, 1)
            if must:
                # Heavy penalty for cancelling mandatory meetings
                reward -= 0.40 * (pw / 3)
            else:
                reward -= 0.05
            self._step_rewards.append(round(reward, 4))
            return round(reward, 4)

        # Scheduling reward
        if action_type.startswith("schedule_"):
            preferred_match = result.get("preferred_match", False)
            no_meeting_violation = result.get("no_meeting_violation", False)
            conflicts = result.get("conflicts_in_calendar", 0)
            priority = result.get("meeting_priority", "low")
            pw = PRIORITY_WEIGHT.get(priority, 1) / 3.0

            # Base reward (Higher base to encourage scheduling)
            reward += 0.50 * pw

            # Conflict penalty (Reduced penalty)
            conflict_penalty = min(0.10, conflicts * 0.02)
            reward -= conflict_penalty

        self._step_rewards.append(round(reward, 4))
        return round(reward, 4)

    def final_score(self) -> float:
        """Compute final calendar quality score [0.0, 1.0]."""
        task_state = self._task.export_state()
        meetings = self._task.get_meetings()
        scheduled = self._task.get_scheduled()
        cancelled = self._task.get_cancelled()

        if not meetings:
            return 0.0

        total_score = 0.0
        max_score = 0.0

        conflicting_ids = self._task._get_conflicting_ids()

        for mtg in meetings:
            mid = mtg["id"]
            pw = PRIORITY_WEIGHT.get(mtg["priority"], 1)
            max_possible = pw * 1.0
            max_score += max_possible
            s = 0.0

            if mid in scheduled:
                slots = scheduled[mid]

                # Conflict-free: 40% of max per meeting (Local check)
                is_conflicting = mid in conflicting_ids
                conflict_score = 0.0 if is_conflicting else 1.0
                s += 0.40 * pw * conflict_score

                # Preference alignment: 20%
                preferred = mtg.get("preferred_slots", [])
                if preferred and all(sl in preferred for sl in slots):
                    s += 0.20 * pw
                elif preferred and any(sl in preferred for sl in slots):
                    s += 0.10 * pw

                # No-meeting violation: -10% per slot
                no_meeting = set(mtg.get("no_meeting_slots", []))
                violations = sum(1 for sl in slots if sl in no_meeting)
                s -= violations * 0.10 * pw

                # Must-schedule compliance: 30%
                if mtg.get("must_schedule", False):
                    s += 0.30 * pw

                # Priority bonus: 10%
                if mtg["priority"] == "high":
                    s += 0.10 * pw

            elif mid in cancelled:
                # Penalise mandatory cancellations heavily
                if mtg.get("must_schedule", False):
                    s -= 0.50 * pw

            total_score += max(0.0, s)

        if max_score == 0:
            return 0.0

        raw = total_score / max_score
        return round(min(1.0, max(0.0, raw)), 4)
