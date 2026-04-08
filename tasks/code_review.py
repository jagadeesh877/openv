"""
Task 2: Code Review Assistant
==============================
Agent receives 6 Python code snippets, each containing one or more bugs.
Agent must:
1. detect_bug — identify the bug category
2. suggest_fix — select the correct fix category
3. set_severity — rate impact (critical/major/minor)
4. approve or reject the code

Scoring rewards correct identification + correct fix + correct severity.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Code snippets with ground truth annotations
# ---------------------------------------------------------------------------

SNIPPETS: List[Dict[str, Any]] = [
    {
        "id": "c001",
        "title": "Off-by-one in pagination",
        "language": "python",
        "code": """
def paginate(items, page, page_size):
    start = page * page_size
    end = start + page_size
    return items[start:end + 1]   # <-- bug: includes one extra item
""",
        "bug_category": "logic_error",
        "fix_category": "boundary_correction",
        "severity": "major",
        "approved": False,
    },
    {
        "id": "c002",
        "title": "SQL injection vulnerability",
        "language": "python",
        "code": """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return db.execute(query)   # <-- bug: f-string SQL injection
""",
        "bug_category": "security_vulnerability",
        "fix_category": "parameterized_query",
        "severity": "critical",
        "approved": False,
    },
    {
        "id": "c003",
        "title": "Race condition in counter",
        "language": "python",
        "code": """
import threading
counter = 0

def increment():
    global counter
    counter += 1   # <-- not thread-safe

threads = [threading.Thread(target=increment) for _ in range(1000)]
""",
        "bug_category": "concurrency_issue",
        "fix_category": "add_lock_or_atomic",
        "severity": "critical",
        "approved": False,
    },
    {
        "id": "c004",
        "title": "Memory leak — unclosed file",
        "language": "python",
        "code": """
def read_config(path):
    f = open(path)           # <-- bug: file never closed
    data = f.read()
    return data
""",
        "bug_category": "resource_leak",
        "fix_category": "use_context_manager",
        "severity": "major",
        "approved": False,
    },
    {
        "id": "c005",
        "title": "Mutable default argument",
        "language": "python",
        "code": """
def append_item(item, lst=[]):   # <-- bug: mutable default
    lst.append(item)
    return lst
""",
        "bug_category": "logic_error",
        "fix_category": "use_none_default",
        "severity": "minor",
        "approved": False,
    },
    {
        "id": "c006",
        "title": "Unhandled exception path",
        "language": "python",
        "code": """
def parse_age(value):
    return int(value)   # <-- no exception handling for non-numeric input
""",
        "bug_category": "error_handling",
        "fix_category": "add_try_except",
        "severity": "major",
        "approved": False,
    },
]

BUG_CATEGORIES = {
    "logic_error", "security_vulnerability", "concurrency_issue",
    "resource_leak", "error_handling"
}
FIX_CATEGORIES = {
    "boundary_correction", "parameterized_query", "add_lock_or_atomic",
    "use_context_manager", "use_none_default", "add_try_except"
}
SEVERITIES = {"critical", "major", "minor"}


class CodeReviewTask:
    """Stateful code review task."""

    def __init__(self):
        self._snippets: List[Dict[str, Any]] = copy.deepcopy(SNIPPETS)
        self._reviews: Dict[str, Dict] = {}   # snippet_id -> agent review
        self._current_index: int = 0
        self._current_stage: str = "detect_bug"  # stages per snippet

    # ------------------------------------------------------------------
    # OpenEnv task interface
    # ------------------------------------------------------------------

    def get_initial_state(self) -> Dict[str, Any]:
        snip = self._snippets[0]
        return {
            "description": (
                f"CODE REVIEW [Snippet 1/{len(self._snippets)}]: {snip['title']}. "
                f"Current stage: {self._current_stage}."
            ),
            "available_actions": self.get_valid_actions(),
            "context": self._snippet_context(snip),
        }

    def get_current_state(self) -> Dict[str, Any]:
        if self._current_index >= len(self._snippets):
            return {
                "description": f"All {len(self._snippets)} snippets reviewed. Done.",
                "context": {"reviewed_count": len(self._reviews)},
            }
        snip = self._snippets[self._current_index]
        return {
            "description": (
                f"CODE REVIEW [Snippet {self._current_index+1}/{len(self._snippets)}]: "
                f"'{snip['title']}'. Current stage: {self._current_stage}. "
                f"Please select one of the following actions to proceed."
            ),
            "context": self._snippet_context(snip),
        }

    def get_valid_actions(self) -> List[str]:
        if self._current_index >= len(self._snippets):
            return ["finish_review"]

        if self._current_stage == "detect_bug":
            return [f"detect_{cat}" for cat in sorted(BUG_CATEGORIES)] + ["skip_snippet"]
        elif self._current_stage == "suggest_fix":
            return [f"fix_{cat}" for cat in sorted(FIX_CATEGORIES)] + ["skip_snippet"]
        elif self._current_stage == "set_severity":
            return ["severity_critical", "severity_major", "severity_minor", "skip_snippet"]
        elif self._current_stage == "verdict":
            return ["approve_code", "reject_code", "skip_snippet"]
        return ["finish_review"]

    def apply_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if action_type == "finish_review":
            return {"done": True, "reason": "Agent submitted all code reviews."}

        if self._current_index >= len(self._snippets):
            return {"done": True, "reason": "No more snippets."}

        snip = self._snippets[self._current_index]
        sid = snip["id"]

        if action_type == "skip_snippet":
            if sid not in self._reviews:
                self._reviews[sid] = {}
            self._reviews[sid]["skipped"] = True
            self._advance_snippet()
            return {
                "done": self._current_index >= len(self._snippets),
                "reason": f"Skipped snippet {sid}.",
            }

        if sid not in self._reviews:
            self._reviews[sid] = {}

        result = {}

        if self._current_stage == "detect_bug":
            detected = action_type.replace("detect_", "", 1)
            self._reviews[sid]["detected_bug"] = detected
            self._current_stage = "suggest_fix"
            result = {
                "done": False,
                "reason": f"Bug detected as '{detected}'. Now suggest a fix.",
                "snippet_id": sid,
                "stage": "suggest_fix",
            }
        elif self._current_stage == "suggest_fix":
            fix = action_type.replace("fix_", "", 1)
            self._reviews[sid]["suggested_fix"] = fix
            self._current_stage = "set_severity"
            result = {
                "done": False,
                "reason": f"Fix '{fix}' noted. Now set severity.",
                "stage": "set_severity",
            }
        elif self._current_stage == "set_severity":
            severity = action_type.replace("severity_", "", 1)
            self._reviews[sid]["severity"] = severity
            self._current_stage = "verdict"
            result = {
                "done": False,
                "reason": f"Severity '{severity}' set. Approve or reject code?",
                "stage": "verdict",
            }
        elif self._current_stage == "verdict":
            verdict = "approved" if action_type == "approve_code" else "rejected"
            self._reviews[sid]["verdict"] = verdict
            self._reviews[sid]["ground_truth"] = {
                "bug_category": snip["bug_category"],
                "fix_category": snip["fix_category"],
                "severity": snip["severity"],
                "should_approve": snip["approved"],
            }
            self._advance_snippet()
            result = {
                "done": self._current_index >= len(self._snippets),
                "reason": f"Snippet {sid} verdict: {verdict}.",
                "review": self._reviews[sid],
            }

        result.setdefault("done", False)
        return result

    def export_state(self) -> Dict[str, Any]:
        return {
            "total_snippets": len(self._snippets),
            "reviewed_count": len(self._reviews),
            "current_index": self._current_index,
            "current_stage": self._current_stage,
            "reviews": self._reviews,
        }

    def get_snippets(self) -> List[Dict]:
        return self._snippets

    def get_reviews(self) -> Dict[str, Dict]:
        return self._reviews

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snippet_context(self, snip: Dict) -> Dict[str, Any]:
        return {
            "snippet_id": snip["id"],
            "title": snip["title"],
            "language": snip["language"],
            "code": snip["code"].strip(),
            "stage": self._current_stage,
            "snippets_remaining": len(self._snippets) - self._current_index,
        }

    def _advance_snippet(self):
        self._current_index += 1
        self._current_stage = "detect_bug"
