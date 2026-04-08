"""
Task 1: Email Triage System
============================
Agent receives an inbox of 8 emails with varying urgency, sender importance,
and content type. Agent must correctly triage each email by:
- Prioritizing (high/medium/low urgency flag)
- Responding (draft a response category)
- Deleting (spam/irrelevant)

Goal: maximize accuracy of triage decisions against ground truth metadata.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Email data — realistic professional inbox
# ---------------------------------------------------------------------------

INBOX: List[Dict[str, Any]] = [
    {
        "id": "e001",
        "from": "ceo@company.com",
        "subject": "Q4 Board Meeting — URGENT action required",
        "body": "Please confirm attendance for tomorrow's 9 AM board meeting. Board deck needs updating with Q4 actuals.",
        "category": "executive",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "high",
        "ground_truth_response": "confirm_attendance",
        "is_spam": False,
    },
    {
        "id": "e002",
        "from": "noreply@promo.deals.xyz",
        "subject": "50% OFF — Limited time flash sale!!!",
        "body": "Click here to claim your exclusive deal. Unsubscribe link broken.",
        "category": "spam",
        "ground_truth_action": "delete",
        "ground_truth_priority": None,
        "ground_truth_response": None,
        "is_spam": True,
    },
    {
        "id": "e003",
        "from": "client.jane@enterprise.com",
        "subject": "Contract renewal — decision needed by EOD",
        "body": "Our legal team has reviewed the MSA. We need your signature by end of business today.",
        "category": "client",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "high",
        "ground_truth_response": "request_legal_review",
        "is_spam": False,
    },
    {
        "id": "e004",
        "from": "hr@company.com",
        "subject": "Reminder: Annual performance review due Friday",
        "body": "Please submit your self-assessment form via the HR portal before end of day Friday.",
        "category": "internal",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "medium",
        "ground_truth_response": "acknowledge",
        "is_spam": False,
    },
    {
        "id": "e005",
        "from": "newsletter@techcrunch.com",
        "subject": "This week in AI: GPT-5, robotics funding, and more",
        "body": "Your weekly tech digest is ready to read.",
        "category": "newsletter",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "low",
        "ground_truth_response": None,
        "is_spam": False,
    },
    {
        "id": "e006",
        "from": "dev.team@company.com",
        "subject": "Production outage — service degraded since 2:00 AM",
        "body": "Payment API is returning 503 errors. Users affected: ~12k. Rollback attempted but failed.",
        "category": "incident",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "high",
        "ground_truth_response": "escalate_incident",
        "is_spam": False,
    },
    {
        "id": "e007",
        "from": "no-reply@survey.monkey.com",
        "subject": "You have been selected for a survey",
        "body": "Take our 45-minute brand awareness survey and enter for a chance to win $500.",
        "category": "spam",
        "ground_truth_action": "delete",
        "ground_truth_priority": None,
        "ground_truth_response": None,
        "is_spam": True,
    },
    {
        "id": "e008",
        "from": "ops.team@company.com",
        "subject": "Server cost optimisation — review attached report",
        "body": "Attached is the monthly AWS cost analysis. No immediate action needed unless you want to reallocate budgets.",
        "category": "ops",
        "ground_truth_action": "prioritize",
        "ground_truth_priority": "low",
        "ground_truth_response": "schedule_review",
        "is_spam": False,
    },
]

VALID_PRIORITIES = {"high", "medium", "low"}
VALID_RESPONSES = {
    "confirm_attendance", "request_legal_review", "acknowledge",
    "escalate_incident", "schedule_review", None,
}


class EmailTriageTask:
    """Stateful email triage task."""

    def __init__(self):
        self._inbox: List[Dict[str, Any]] = copy.deepcopy(INBOX)
        self._triaged: Dict[str, Dict] = {}   # email_id -> agent decision
        self._current_index: int = 0

    # ------------------------------------------------------------------
    # OpenEnv task interface
    # ------------------------------------------------------------------

    def get_initial_state(self) -> Dict[str, Any]:
        email = self._inbox[0]
        return {
            "description": (
                f"EMAIL TRIAGE [Email 1/{len(self._inbox)}]: "
                f"Subject: '{email['subject']}' from {email['from']} (ID: {email['id']})."
            ),
            "available_actions": self.get_valid_actions(),
            "context": self._email_context(email),
        }

    def get_current_state(self) -> Dict[str, Any]:
        if self._current_index >= len(self._inbox):
            return {
                "description": f"All {len(self._inbox)} emails triaged. Done.",
                "context": {"triaged_count": len(self._triaged)},
            }
        email = self._inbox[self._current_index]
        return {
            "description": (
                f"EMAIL TRIAGE [Email {self._current_index+1}/{len(self._inbox)}]: "
                f"Subject: '{email['subject']}' from {email['from']} (ID: {email['id']})."
            ),
            "context": self._email_context(email),
        }

    def get_valid_actions(self) -> List[str]:
        if self._current_index >= len(self._inbox):
            return ["finish"]
        return ["prioritize_high", "prioritize_medium", "prioritize_low",
                "delete_email", "respond_confirm_attendance",
                "respond_request_legal_review", "respond_acknowledge",
                "respond_escalate_incident", "respond_schedule_review",
                "skip_email"]

    def apply_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if action_type == "finish":
            return {"done": True, "reason": "Agent finished triage session."}

        if self._current_index >= len(self._inbox):
            return {"done": True, "reason": "No more emails to process."}

        email = self._inbox[self._current_index]

        if action_type == "skip_email":
            self._triaged[email["id"]] = {"action": "skipped", "priority": None, "response": None}
            self._current_index += 1
            done = self._current_index >= len(self._inbox)
            return {"done": done, "reason": f"Skipped email {email['id']}."}

        decision = self._parse_decision(action_type)
        self._triaged[email["id"]] = decision
        self._current_index += 1

        done = self._current_index >= len(self._inbox)
        return {
            "done": done,
            "reason": f"Triaged email {email['id']} as {decision}.",
            "email_id": email["id"],
            "decision": decision,
            "ground_truth": {
                "action": email["ground_truth_action"],
                "priority": email["ground_truth_priority"],
                "response": email["ground_truth_response"],
            },
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "total_emails": len(self._inbox),
            "triaged_count": len(self._triaged),
            "current_index": self._current_index,
            "decisions": self._triaged,
        }

    def get_inbox(self) -> List[Dict]:
        return self._inbox

    def get_triaged(self) -> Dict[str, Dict]:
        return self._triaged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _email_context(self, email: Dict) -> Dict[str, Any]:
        return {
            "email_id": email["id"],
            "from": email["from"],
            "subject": email["subject"],
            "body_preview": email["body"][:120],
            "category": email["category"],
            "emails_remaining": len(self._inbox) - self._current_index,
        }

    def _parse_decision(self, action_type: str) -> Dict[str, Any]:
        if action_type.startswith("delete"):
            return {"action": "delete", "priority": None, "response": None}
        elif action_type.startswith("prioritize_"):
            level = action_type.split("_", 1)[1]
            return {"action": "prioritize", "priority": level, "response": None}
        elif action_type.startswith("respond_"):
            response_type = action_type.split("_", 1)[1]
            return {"action": "respond", "priority": "high", "response": response_type}
        return {"action": "unknown", "priority": None, "response": None}
