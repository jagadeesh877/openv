"""
Task 3: Meeting Scheduler
==========================
Agent manages a calendar with 10 meeting requests, some overlapping.
Attendees have preferences (preferred slots, no-meeting windows).
Agent must:
- schedule meetings in valid slots
- resolve conflicts (reschedule or cancel)
- optimize total attendee satisfaction

Scoring rewards: conflict-free schedule + attendee preference alignment.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Calendar data
# ---------------------------------------------------------------------------

# Working slots: 9:00–17:00, 30-min blocks
ALL_SLOTS = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30",
]

MEETING_REQUESTS: List[Dict[str, Any]] = [
    {
        "id": "m001",
        "title": "Q4 Strategy Planning",
        "required_attendees": ["CEO", "CFO", "CTO"],
        "duration_slots": 2,      # 1 hour
        "preferred_slots": ["10:00", "10:30"],
        "no_meeting_slots": [],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m002",
        "title": "Engineering sprint planning",
        "required_attendees": ["CTO", "Lead-Dev", "QA-Lead"],
        "duration_slots": 2,
        "preferred_slots": ["09:00", "09:30"],
        "no_meeting_slots": ["12:00", "12:30"],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m003",
        "title": "1:1 CEO — CFO",
        "required_attendees": ["CEO", "CFO"],
        "duration_slots": 1,      # 30 min
        "preferred_slots": ["14:00"],
        "no_meeting_slots": ["09:00", "09:30"],
        "priority": "medium",
        "must_schedule": True,
    },
    {
        "id": "m004",
        "title": "Vendor demo: Cloud provider",
        "required_attendees": ["CTO", "Cloud-Arch"],
        "duration_slots": 2,
        "preferred_slots": ["11:00", "11:30"],
        "no_meeting_slots": [],
        "priority": "medium",
        "must_schedule": False,
    },
    {
        "id": "m005",
        "title": "Legal compliance review",
        "required_attendees": ["CFO", "Legal"],
        "duration_slots": 2,
        "preferred_slots": ["15:00", "15:30"],
        "no_meeting_slots": ["09:00", "09:30", "10:00"],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m006",
        "title": "Marketing campaign review",
        "required_attendees": ["CMO", "Content-Lead"],
        "duration_slots": 1,
        "preferred_slots": ["13:00"],
        "no_meeting_slots": [],
        "priority": "low",
        "must_schedule": False,
    },
    {
        "id": "m007",
        "title": "Incident post-mortem",
        "required_attendees": ["CTO", "Lead-Dev", "DevOps"],
        "duration_slots": 2,
        "preferred_slots": ["16:00", "16:30"],
        "no_meeting_slots": ["09:00"],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m008",
        "title": "New hire orientation",
        "required_attendees": ["HR", "New-Hire"],
        "duration_slots": 3,
        "preferred_slots": ["09:00", "09:30", "10:00"],
        "no_meeting_slots": ["15:00", "15:30", "16:00"],
        "priority": "medium",
        "must_schedule": True,
    },
    {
        "id": "m009",
        "title": "Team lunch planning (optional)",
        "required_attendees": ["ALL"],
        "duration_slots": 1,
        "preferred_slots": ["12:00"],
        "no_meeting_slots": [],
        "priority": "low",
        "must_schedule": False,
    },
    {
        "id": "m010",
        "title": "Budget review deep-dive",
        "required_attendees": ["CFO", "Finance-Lead", "CEO"],
        "duration_slots": 3,
        "preferred_slots": ["10:00", "10:30", "11:00"],
        "no_meeting_slots": ["09:00", "09:30"],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m011",
        "title": "Urgent CEO Update",
        "required_attendees": ["CEO", "CFO", "COO"],
        "duration_slots": 2,
        "preferred_slots": ["10:00", "10:30"],
        "no_meeting_slots": ["15:00", "15:30", "16:00"],
        "priority": "high",
        "must_schedule": True,
    },
    {
        "id": "m012",
        "title": "Security Patch Review",
        "required_attendees": ["CTO", "Lead-Dev", "Security-Ops"],
        "duration_slots": 1,
        "preferred_slots": ["09:00"],
        "no_meeting_slots": ["12:00", "12:30", "13:00"],
        "priority": "high",
        "must_schedule": True,
    },
]


class MeetingSchedulerTask:
    """Stateful meeting scheduler task."""

    def __init__(self):
        self._meetings: List[Dict[str, Any]] = copy.deepcopy(MEETING_REQUESTS)
        # scheduled: meeting_id -> list of slot strings
        self._scheduled: Dict[str, List[str]] = {}
        # cancelled: set of meeting_ids
        self._cancelled: Set[str] = set()
        # pending: meetings not yet decided
        self._pending: List[str] = [m["id"] for m in self._meetings]
        self._current_mid: Optional[str] = self._pending[0] if self._pending else None

    # ------------------------------------------------------------------
    # OpenEnv task interface
    # ------------------------------------------------------------------

    def get_initial_state(self) -> Dict[str, Any]:
        mtg = self._get_meeting(self._current_mid)
        return {
            "description": (
                f"Meeting Scheduler — {len(self._meetings)} meetings to process. "
                f"Current: [{mtg['id']}] '{mtg['title']}'."
            ),
            "available_actions": self.get_valid_actions(),
            "context": self._meeting_context(mtg),
        }

    def get_current_state(self) -> Dict[str, Any]:
        if not self._pending:
            conflicts = self._count_conflicts()
            return {
                "description": (
                    f"All meetings processed. "
                    f"Scheduled: {len(self._scheduled)}, "
                    f"Cancelled: {len(self._cancelled)}, "
                    f"Conflicts remaining: {conflicts}."
                ),
                "context": self._summary_context(),
            }
        mtg = self._get_meeting(self._current_mid)
        return {
            "description": (
                f"Meeting [{len(self._meetings)-len(self._pending)+1}/{len(self._meetings)}] — "
                f"'{mtg['title']}'. Priority: {mtg['priority']}. "
                f"Conflicts in calendar: {self._count_conflicts()}."
            ),
            "context": self._meeting_context(mtg),
        }

    def get_valid_actions(self) -> List[str]:
        if not self._pending:
            return ["finalize_schedule"]

        actions = []
        mtg = self._get_meeting(self._current_mid)
        dur = mtg["duration_slots"]
        # Generate valid schedule actions for each possible slot window
        for i in range(len(ALL_SLOTS) - dur + 1):
            slot_window = ALL_SLOTS[i: i + dur]
            slot_label = f"schedule_{slot_window[0].replace(':', '')}"
            actions.append(slot_label)
        
        # Move cancel_meeting to the end so it is not the default fallback
        actions.append("cancel_meeting")
        return actions

    def _get_busy_map(self) -> Dict[str, List[str]]:
        """Identify which attendees are busy in which slots."""
        busy_map: Dict[str, List[str]] = {}
        for mid, slots in self._scheduled.items():
            mtg = self._get_meeting(mid)
            attendees = mtg.get("required_attendees", [])
            for slot in slots:
                if slot not in busy_map:
                    busy_map[slot] = []
                # Ensure we don't duplicate names in the same slot
                for a in attendees:
                    if a not in busy_map[slot]:
                        busy_map[slot].append(a)
        return busy_map

    def apply_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if action_type == "finalize_schedule":
            conflicts = self._count_conflicts()
            return {
                "done": True,
                "reason": f"Schedule finalized. Conflicts: {conflicts}.",
                "summary": self._summary_context(),
            }

        if not self._pending:
            return {"done": True, "reason": "No pending meetings."}

        mid = self._current_mid
        mtg = self._get_meeting(mid)

        if action_type == "cancel_meeting":
            self._cancelled.add(mid)
            self._advance()
            return {
                "done": not self._pending,
                "reason": f"Cancelled meeting {mid}: '{mtg['title']}'.",
                "cancelled_priority": mtg["priority"],
                "must_schedule": mtg["must_schedule"],
            }

        if action_type.startswith("schedule_"):
            start_raw = action_type.replace("schedule_", "")
            # Convert e.g. "0900" -> "09:00"
            start_slot = f"{start_raw[:2]}:{start_raw[2:]}"
            dur = mtg["duration_slots"]

            if start_slot not in ALL_SLOTS:
                return {
                    "done": False,
                    "reason": f"Invalid start slot '{start_slot}'.",
                    "error": "invalid_slot",
                }

            start_idx = ALL_SLOTS.index(start_slot)
            if start_idx + dur > len(ALL_SLOTS):
                return {
                    "done": False,
                    "reason": f"Slot window exceeds working hours.",
                    "error": "out_of_bounds",
                }

            slot_window = ALL_SLOTS[start_idx: start_idx + dur]
            in_preferred = all(s in mtg["preferred_slots"] for s in slot_window)
            in_no_meeting = any(s in mtg["no_meeting_slots"] for s in slot_window)

            self._scheduled[mid] = slot_window
            self._advance()

            conflicts_after = self._count_conflicts()
            return {
                "done": not self._pending,
                "reason": (
                    f"Scheduled '{mtg['title']}' at {slot_window}. "
                    f"Preferred: {in_preferred}. "
                    f"Violates no-meeting: {in_no_meeting}. "
                    f"Total conflicts: {conflicts_after}."
                ),
                "scheduled_slots": slot_window,
                "preferred_match": in_preferred,
                "no_meeting_violation": in_no_meeting,
                "conflicts_in_calendar": conflicts_after,
                "meeting_priority": mtg["priority"],
            }

        return {"done": False, "reason": f"Unknown action '{action_type}'."}

    def export_state(self) -> Dict[str, Any]:
        return {
            "total_meetings": len(self._meetings),
            "scheduled": self._scheduled,
            "cancelled": list(self._cancelled),
            "pending": self._pending,
            "conflicts": self._count_conflicts(),
        }

    def get_meetings(self) -> List[Dict]:
        return self._meetings

    def get_scheduled(self) -> Dict[str, List[str]]:
        return self._scheduled

    def get_cancelled(self) -> Set[str]:
        return self._cancelled

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_meeting(self, mid: Optional[str]) -> Dict[str, Any]:
        if mid is None:
            return {}
        return next((m for m in self._meetings if m["id"] == mid), {})

    def _advance(self):
        if self._current_mid in self._pending:
            self._pending.remove(self._current_mid)
        self._current_mid = self._pending[0] if self._pending else None

    def _get_conflicting_ids(self) -> List[str]:
        """Return IDs of meetings currently involved in a conflict."""
        slot_users: Dict[str, List[str]] = {}
        for mid, slots in self._scheduled.items():
            mtg = self._get_meeting(mid)
            attendees = mtg.get("required_attendees", [])
            for slot in slots:
                if slot not in slot_users:
                    slot_users[slot] = []
                slot_users[slot].extend(attendees)

        conflicting_mids = set()
        for slot, attendees in slot_users.items():
            if len(attendees) != len(set(attendees)):
                # Conflict found in this slot. Find which meetings use this slot.
                for mid, slots in self._scheduled.items():
                    if slot in slots:
                        conflicting_mids.add(mid)
        return list(conflicting_mids)

    def _count_conflicts(self) -> int:
        """Count total unique slots with conflicts."""
        slot_users: Dict[str, List[str]] = {}
        for mid, slots in self._scheduled.items():
            mtg = self._get_meeting(mid)
            attendees = mtg.get("required_attendees", [])
            for slot in slots:
                if slot not in slot_users:
                    slot_users[slot] = []
                slot_users[slot].extend(attendees)

        conflicts = 0
        for slot, attendees in slot_users.items():
            if len(attendees) != len(set(attendees)):
                conflicts += 1
        return conflicts

    def _meeting_context(self, mtg: Dict) -> Dict[str, Any]:
        busy_map = self._get_busy_map()
        required_attendees = mtg.get("required_attendees", [])
        
        fully_free_slots = []
        for slot in ALL_SLOTS:
            is_busy = False
            slot_attendees = busy_map.get(slot, [])
            for attendee in required_attendees:
                if attendee in slot_attendees or attendee == "ALL" and slot_attendees:
                    is_busy = True
                    break
            if not is_busy:
                fully_free_slots.append(slot)

        return {
            "meeting_id": mtg.get("id"),
            "title": mtg.get("title"),
            "attendees": required_attendees,
            "duration_slots": mtg.get("duration_slots"),
            "priority": mtg.get("priority"),
            "must_schedule": mtg.get("must_schedule"),
            "fully_free_slots_for_these_attendees": fully_free_slots,
            "busy_map": busy_map,
            "slots_available": ALL_SLOTS,
            "pending_count": len(self._pending),
        }

    def _summary_context(self) -> Dict[str, Any]:
        return {
            "scheduled": self._scheduled,
            "cancelled": list(self._cancelled),
            "conflicts": self._get_conflicting_ids(),
            "total_meetings": len(self._meetings),
        }
