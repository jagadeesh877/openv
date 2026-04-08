"""
Code Review Grader
===================
Evaluates agent review decisions against ground truth per snippet.

Partial credit breakdown per snippet (sum = 1.0):
- Correct bug_category detected:  0.35
- Correct fix_category suggested: 0.35
- Correct severity set:           0.20
- Correct verdict (approve/reject):0.10

Final score = weighted average across all snippets.
Higher weight for critical severity bugs.
"""
from __future__ import annotations

from typing import Any, Dict

SEVERITY_WEIGHT = {"critical": 2.0, "major": 1.5, "minor": 1.0}


class CodeGrader:
    def __init__(self, task):
        self._task = task
        self._snippet_scores: Dict[str, float] = {}
        # Track which snippet index we are currently grading so snippet_id is
        # always correct — avoids the bug of matching by bug_category (ambiguous).
        self._current_snippet_idx: int = 0

    def score_step(self, action_type: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Score each stage's action. Only give substantive reward at verdict stage."""
        if result.get("done") and action_type == "finish_review":
            return 0.0

        review = result.get("review")
        if not review:
            # Intermediate stage (detect / fix / severity) — small progress signal
            # so trajectory has non-sparse rewards throughout the episode.
            return 0.02

        gt = review.get("ground_truth", {})
        if not gt:
            return 0.0

        snippets = self._task.get_snippets()
        # Use the current index tracker (not a fragile loop match)
        snippet_id = None
        if self._current_snippet_idx < len(snippets):
            snippet_id = snippets[self._current_snippet_idx]["id"]
        # Advance for next verdict call
        self._current_snippet_idx += 1

        score = 0.0

        # Bug category (0.35)
        if review.get("detected_bug") == gt.get("bug_category"):
            score += 0.35
        elif self._partial_bug_match(review.get("detected_bug", ""), gt.get("bug_category", "")):
            score += 0.10

        # Fix category (0.35)
        if review.get("suggested_fix") == gt.get("fix_category"):
            score += 0.35
        elif self._partial_fix_match(review.get("suggested_fix", ""), gt.get("fix_category", "")):
            score += 0.10

        # Severity (0.20)
        if review.get("severity") == gt.get("severity"):
            score += 0.20
        elif self._adjacent_severity(review.get("severity", ""), gt.get("severity", "")):
            score += 0.08

        # Verdict (0.10)
        agent_verdict = review.get("verdict", "")
        expected_approve = gt.get("should_approve", False)
        if (agent_verdict == "approved") == expected_approve:
            score += 0.10

        # Weight by ground truth severity — critical bugs matter more
        weight = SEVERITY_WEIGHT.get(gt.get("severity", "minor"), 1.0)
        # Normalise so weighted score stays in [0, 1]
        weighted_score = min(1.0, score) * weight / 2.0

        if snippet_id:
            self._snippet_scores[snippet_id] = round(weighted_score, 4)

        # Return raw (un-weighted) score clamped to [0, 1] as step reward
        return round(min(1.0, max(-1.0, score)), 4)

    def final_score(self) -> float:
        """Weighted final score across all snippets [0.0, 1.0]."""
        snippets = self._task.get_snippets()
        if not snippets:
            return 0.0

        total_weight = 0.0
        total_score = 0.0
        for snip in snippets:
            sid = snip["id"]
            weight = SEVERITY_WEIGHT.get(snip["severity"], 1.0)
            total_weight += weight
            total_score += self._snippet_scores.get(sid, 0.0) * weight

        if total_weight == 0:
            return 0.0

        return round(total_score / total_weight, 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _partial_bug_match(self, got: str, expected: str) -> bool:
        """Partial credit for bugs in the same family."""
        families = {
            "logic_error": {"logic_error", "error_handling"},
            "security_vulnerability": {"security_vulnerability"},
            "concurrency_issue": {"concurrency_issue"},
            "resource_leak": {"resource_leak"},
            "error_handling": {"error_handling", "logic_error"},
        }
        return got in families.get(expected, set())

    def _partial_fix_match(self, got: str, expected: str) -> bool:
        """Partial credit for fix categories in the same family."""
        families = {
            "boundary_correction": {"boundary_correction", "use_none_default"},
            "parameterized_query": {"parameterized_query"},
            "add_lock_or_atomic": {"add_lock_or_atomic"},
            "use_context_manager": {"use_context_manager"},
            "use_none_default": {"use_none_default", "boundary_correction"},
            "add_try_except": {"add_try_except"},
        }
        return got in families.get(expected, set())

    def _adjacent_severity(self, got: str, expected: str) -> bool:
        order = ["minor", "major", "critical"]
        if got not in order or expected not in order:
            return False
        return abs(order.index(got) - order.index(expected)) == 1
