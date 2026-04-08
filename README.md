---
title: OpenEnv AI Workspace
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🤖 OpenEnv AI Workspace

**Production-grade OpenEnv-compatible AI environment for agent evaluation**

> Simulates three realistic professional workflows where an AI agent must reason, act, and improve performance over time — fully evaluable by automated graders and human reviewers.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-orange)](https://openenv.ai)

---

## 🎯 Real-World Motivation

The "Evaluation Gap" is the single greatest bottleneck in AI Agent development. While we have excellent benchmarks for code snippets and math problems, we lack environments that model the **high-frequency, messy professional tasks** that knowledge workers perform daily.

This environment is designed to fill that gap by simulating three distinct real-world domains:

1. **Email Triage** — Managing high-volume inboxes (CEO vs. Spam) with urgency and response-type logic.
2. **Code Review** — Identifying security vulnerabilities and logic errors in a multi-stage review pipeline.
3. **Meeting Scheduling** — Solving an over-constrained constraint satisfaction problem involving attendee preferences and mandatory mandates.

By providing **deterministic transitions**, **partial-credit grading**, and **stunningly clear state representations**, this workspace allows researchers to measure agent reasoning, the impact of prompt engineering, and the efficiency of tool-use in a production-ready package.

---

## 📋 Task Descriptions

### Task 1 — Email Triage System *(Easy)*
**Motivation:** Knowledge workers spend 28% of their workday on email. An AI agent that correctly triages an inbox saves hours and prevents critical messages from being missed.

**Setup:** Agent is presented 8 professional emails, each with varying urgency, sender rank, and content type (executive, client, incident, spam, newsletter, ops).

**Agent must:**
- Prioritize emails as `high`, `medium`, or `low`
- Choose the correct response template (`confirm_attendance`, `escalate_incident`, etc.)
- Delete spam emails without deleting real ones

**Scoring (per email, max 1.0):**
| Criterion | Points |
|---|---|
| Correct action (prioritize/delete/respond) | 0.40 |
| Correct priority level | 0.30 |
| Correct response type | 0.30 |
| Wrongly deleting a real email | −0.20 |

---

### Task 2 — Code Review Assistant *(Medium)*
**Motivation:** Code review is the #1 quality gate in software development. An AI that can detect bug types and suggest the right fix category reduces engineer cognitive load.

**Setup:** Agent reviews 6 Python code snippets, each with a specific, realistic bug (SQL injection, race condition, off-by-one, resource leak, mutable default, unclosed exception path).

**Agent must (per snippet, in 4 stages):**
1. `detect_bug` — Identify bug category: `logic_error`, `security_vulnerability`, `concurrency_issue`, `resource_leak`, `error_handling`
2. `suggest_fix` — Choose correct fix: `parameterized_query`, `add_lock_or_atomic`, `use_context_manager`, etc.
3. `set_severity` — Rate impact: `critical`, `major`, or `minor`
4. `verdict` — `approve_code` or `reject_code`

**Scoring (per snippet, max 1.0 × severity weight):**
| Criterion | Points |
|---|---|
| Correct bug category | 0.35 |
| Correct fix category | 0.35 |
| Correct severity | 0.20 |
| Correct verdict | 0.10 |
| Critical bugs | 2× weight |
| Major bugs | 1.5× weight |

---

### Task 3 — Meeting Scheduler *(Hard)*
**Motivation:** Scheduling across multiple stakeholders with competing preferences is an NP-hard constraint satisfaction problem in practice. Agents that resolve conflicts efficiently have direct business impact.

**Setup:** 10 meeting requests on a single workday (09:00–17:00, 30-min slots). Meetings have required attendees, preferred time windows, no-meeting restrictions, and mandatory/optional flags.

**Agent must:**
- Schedule meetings in valid 30-minute slot windows
- Avoid double-booking attendees (conflict-free scheduling)
- Respect preferred slots (preference alignment)
- Prioritise mandatory meetings over optional ones

**Scoring components:**
| Criterion | Weight |
|---|---|
| Conflict-free scheduling | 40% |
| Must-schedule compliance | 30% |
| Preference alignment | 20% |
| Priority handling | 10% |

---

## 🏆 Reward Design

### Step-level Rewards
| Event | Reward |
|---|---|
| Correct action | +0.10 to +0.40 |
| Partial credit (adjacent) | +0.05 to +0.20 |
| Invalid action (not in valid set) | −0.15 |
| Repeated action (seen 2+ times) | −0.10 |
| Efficiency bonus (done in <70% max steps) | up to +0.20 |

### Final Scores
All graders return a final `float` in `[0.0, 1.0]`:
- **Email Triage**: average per-email score
- **Code Review**: severity-weighted average per-snippet score
- **Meeting Scheduler**: composite calendar quality score

---

## ✨ Interactive Dashboard

The environment now includes a **premium web-based dashboard** for real-time visualization:
- **Task Visualization**: Dynamic views for Inbox (Emails), Code Editor (Review), and Calendar (Scheduler).
- **Live Feed**: See agent actions and rewards as they happen with micro-animations.
- **Manual Control**: Reset tasks or test actions manually directly from the UI.
- **Aesthetics**: Modern dark-mode glassmorphism design.

Access it at: `http://localhost:7860/` (or your HF Space URL).

---

## 🔧 Setup & Usage

### Local Development

```bash
# 1. Clone the repo
git clone <repo-url>
cd openenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
python main.py --mode server
# → Server running on http://localhost:7860

# 4. Run baseline agent only (no server)
python main.py --mode agent
```

### Docker

```bash
# Build
docker build -t openenv-ai .

# Run
docker run -p 7860:7860 openenv-ai

# Test
curl http://localhost:7860/
```

### Hugging Face Spaces Deployment
1. Push the entire project to a new HuggingFace Space repo
2. Set SDK: `docker` in the Space settings
3. Space auto-builds and serves on port 7860

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Environment info + status |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Reset environment `{"task_id": "email_triage"}` |
| `POST` | `/step` | Apply action `{"action_type": "...", "parameters": {}}` |
| `GET` | `/state` | Current state snapshot |
| `GET` | `/run-baseline` | Run baseline agent, return all scores |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🧩 Action & Observation Spaces

### Observation Space (Typed Pydantic Model)
| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Current active task |
| `step` | `int` | Current step count |
| `state_description` | `str` | Natural language description of the current UI/state |
| `available_actions` | `List[str]` | List of valid action strings for the current state |
| `context` | `Dict` | Structured data (e.g., email body, code snippet, calendar slots) |
| `done` | `bool` | Episode terminal state flag |

### Action Space (Task-Dependent)
| Task | Valid Actions | Parameters |
|---|---|---|
| **Email Triage** | `prioritize_[level]`, `delete_email`, `respond_[type]`, `skip_email` | `{}` |
| **Code Review** | `detect_[bug]`, `fix_[method]`, `severity_[level]`, `verdict`, `finish_review` | `{}` |
| **Meeting Scheduler** | `schedule_HHMM`, `cancel_meeting`, `finalize_schedule` | `{}` |

---

## 📊 Example Baseline Agent Output

```
============================================================
  OpenEnv Baseline Agent — Running All Tasks
============================================================

──────────────────────────────────────────────────────
  Task: EMAIL_TRIAGE
  Steps taken:   8
  Total reward:  4.9967
  Final score:   0.6062

──────────────────────────────────────────────────────
  Task: CODE_REVIEW
  Steps taken:   30
  Total reward:  0.2800
  Final score:   0.3289

──────────────────────────────────────────────────────
  Task: MEETING_SCHEDULER
  Steps taken:   30
  Total reward:  0.0700
  Final score:   0.6607

============================================================
  AVERAGE SCORE: 0.5319
============================================================
```

---

## 📁 Project Structure

```
openenv/
├── env/
│   ├── __init__.py
│   └── environment.py          # Core OpenEnv class: reset(), step(), state()
├── tasks/
│   ├── __init__.py
│   ├── email_triage.py         # Task 1: 8-email inbox triage
│   ├── code_review.py          # Task 2: 6-snippet code review pipeline
│   └── meeting_scheduler.py    # Task 3: 10-meeting conflict resolver
├── graders/
│   ├── __init__.py
│   ├── email_grader.py         # Partial-credit email scoring
│   ├── code_grader.py          # Severity-weighted code scoring
│   └── meeting_grader.py       # Calendar quality scoring
├── main.py                     # FastAPI server + baseline agent
├── requirements.txt
├── Dockerfile
├── openenv.yaml                # OpenEnv spec manifest
└── README.md
```

---

## 🧩 OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| `Observation` Pydantic model | ✅ |
| `Action` Pydantic model | ✅ |
| `Reward` Pydantic model | ✅ |
| `reset() → Observation` | ✅ |
| `step(action) → (obs, reward, done, info)` | ✅ |
| `state() → Dict` | ✅ |
| 3+ tasks with unique graders | ✅ |
| Deterministic transitions | ✅ |
| Partial-credit grading | ✅ |
| Score varies by agent behavior | ✅ |
| Baseline agent script | ✅ |
| Docker + HuggingFace ready | ✅ |
| FastAPI on port 7860 | ✅ |

---

## 🛡️ Anti-Disqualification Checklist

- [x] Environment never crashes — all invalid actions gracefully penalised
- [x] No hardcoded outputs — all rewards computed from agent behavior
- [x] No trivial/copied environment — original domain-specific logic
- [x] Grader scores vary by agent's actual decisions
- [x] Baseline agent produces reproducible, non-trivial scores
- [x] Dynamic state transitions at every step

---

## 📄 License

MIT License. See `LICENSE` for details.
