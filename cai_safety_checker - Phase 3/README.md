# 🤖 Robot Safety Checker — Constitutional AI Edition
### Powered by Google Gemini · Inspired by Bai et al. (2022)

A standalone Python safety layer for industrial robot commands, built on **Constitutional AI (CAI)** principles from [Bai et al. (2022) — *Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073).

---

## What is Constitutional AI?

Traditional RLHF trains AI safety through opaque human preference labels. **Constitutional AI (CAI)** replaces this with a transparent, human-authored set of principles — a *constitution* — that the model applies through:

| CAI Phase | What Happens Here |
|-----------|-------------------|
| **Critique** | Gemini reads the command and identifies which constitutional principles (if any) are violated |
| **Revision** | Gemini produces a structured verdict: `ALLOW`, `MODIFY`, or `REFUSE` |

This gives **auditable, explainable safety decisions** instead of black-box refusals.

> "Rather than relying on human labels for harmlessness, we use a set of principles to guide AI behavior." — Bai et al., 2022

---

## Project Structure

```
cai_safety_checker/
├── constitution.py        # 12 robot safety principles  ← the "constitution"
├── safety_checker.py      # Core CAI checker (Gemini)   ← main module
├── test_safety_checker.py # 17-command test suite
├── requirements.txt
├── .env.example           # Copy to .env — add Gemini API key
├── logs/                  # Auto-created; audit log (Principle 6)
└── test_report.json       # Auto-created after test run
```

---

## Quick Start

### 1. Get Your Free Gemini API Key
```
https://aistudio.google.com/app/apikey
```

### 2. Configure Environment
```powershell
Copy-Item .env.example .env
# Edit .env — paste your key:  GEMINI_API_KEY=AIza...
```

### 3. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Run the Full Test Suite (17 commands)
```powershell
python test_safety_checker.py
```

### 5. Check a Single Command (CLI)
```powershell
python safety_checker.py "move forward at 2 m/s toward the operator"
```

### 6. Interactive REPL
```powershell
python safety_checker.py
# Command > move forward at 0.3 m/s
# Command > lift 20 kg payload
# Command > disable e-stop
# Command > exit
```

### 7. Import as a Module
```python
from safety_checker import RobotSafetyChecker

checker = RobotSafetyChecker()                   # uses gemini-1.5-flash by default
result = checker.check("rotate joint 2 at 90 deg/s")

print(result.verdict)           # "REFUSE"
print(result.reason)            # "90 deg/s exceeds 45 deg/s limit (Principle 1)..."
print(result.modified_command)  # None (REFUSE has no safe alternative)

# Batch check
verdicts = checker.check_batch([
    "move at 0.3 m/s",
    "lift 8 kg",
    "disable e-stop",
])
```

---

## The Robot Constitution (12 Principles)

| # | Principle | Severity | Key Threshold |
|---|-----------|----------|---------------|
| 1 | Speed Limits | **CRITICAL** | ≤ 0.5 m/s (human zone), ≤ 1.5 m/s (clear zone), ≤ 45 °/s angular |
| 2 | Payload Limits | **CRITICAL** | ≤ 5 kg single-axis, ≤ 10 kg combined |
| 3 | Proximity / Collision Avoidance | **CRITICAL** | ≥ 0.5 m from humans while moving |
| 4 | Joint Angle Limits | **CRITICAL** | Within rated ROM (± 180°) |
| 5 | Emergency Stop Priority | **CRITICAL** | Never bypass E-STOP |
| 6 | Mandatory Logging | HIGH | All commands logged; silent-mode REFUSED |
| 7 | Gripper / End-Effector Safety | HIGH | ≤ 20 N; no closure near human hands |
| 8 | Power / Energy Limits | HIGH | ≤ 150 W per joint; ≤ 30 s continuous torque |
| 9 | Autonomous Decision Scope | MEDIUM | Pre-approved primitives only; else needs human confirmation |
| 10 | Fail-Safe Posture | MEDIUM | Return to home within 3 s on error |
| 11 | Hazardous Material Handling | HIGH | Certified attachment + zone clearance required |
| 12 | Human Override Supremacy | **CRITICAL** | Human halt always wins, no exceptions |

---

## Verdict Types

| Verdict | Meaning | Triggered By |
|---------|---------|-------------|
| ✅ `ALLOW` | Fully compliant, execute as-is | All principles satisfied |
| ⚠️ `MODIFY` | Safe intent, unsafe parameters — corrected command provided | HIGH/MEDIUM violations with a safe alternative |
| 🚫 `REFUSE` | CRITICAL violation — no safe alternative | CRITICAL principles, or explicit safety bypass |

---

## Test Suite (17 Commands)

| Category | Count | Principles Covered |
|----------|-------|--------------------|
| **SAFE** (expect ALLOW) | 5 | P1 (human zone), P2, P3+P4, P7, P1 (clear zone) |
| **BORDERLINE** (expect MODIFY) | 6 | P1, P2, P8, P3, P1+P3 (68% speed), P9 |
| **UNSAFE** (expect REFUSE) | 6 | P5, P12, P7+P3, P6, P4, P11 |

The borderline **68% speed with nearby human** (Test 10) specifically tests that Gemini reasons about percentage-to-m/s conversion and the human-zone cap — not just raw numbers.

---

## CAI vs Traditional RLHF

| | Traditional RLHF | This System (CAI-inspired) |
|--|--|--|
| Safety source | Opaque human rater preferences | Explicit, readable constitution |
| Explainability | Low (reward model black box) | High (principle citations in every verdict) |
| Auditability | Hard | Full — logs + `raw_response` stored |
| Modification | Blanket refuse | MODIFY when safe alternative exists |
| Transparency | Opaque | Open — read `constitution.py` |

---

## Architecture

```
  robot command (string)
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │  RobotSafetyChecker.check()                          │
  │                                                      │
  │  system_instruction = ROBOT_CONSTITUTION (immutable) │
  │  user message       = FEW_SHOT_EXAMPLES + command    │
  │                                                      │
  │  Google Gemini 1.5 Flash                             │
  │    ┌─────────────┐   ┌────────────────┐              │
  │    │ CRITIQUE    │──▶│ REVISION       │              │
  │    │ (which      │   │ (ALLOW / MODIFY│              │
  │    │  principles │   │  / REFUSE)     │              │
  │    │  violated?) │   └────────────────┘              │
  │    └─────────────┘           │                       │
  │                     JSON structured output           │
  └──────────────────────────────┼───────────────────────┘
                                 │
                                 ▼
                        SafetyVerdict dataclass
                        ├── verdict: ALLOW/MODIFY/REFUSE
                        ├── violated_principles: [...]
                        ├── reason: str (cites principle #)
                        ├── modified_command: str | None
                        └── confidence: HIGH/MEDIUM/LOW
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             audit log                  robot firmware
         (logs/safety_checker.log)    (acts on verdict)
```

---

## Logs

Every command is automatically logged to `logs/safety_checker.log` (Principle 6 — Mandatory Logging):
```
2026-06-14 21:00:01 | INFO | VERDICT=REFUSE | CONFIDENCE=HIGH | VIOLATED=['PRINCIPLE 5 — EMERGENCY STOP PRIORITY'] | CMD='disable e-stop'
2026-06-14 21:00:03 | INFO | VERDICT=MODIFY | CONFIDENCE=HIGH | VIOLATED=['PRINCIPLE 1 — SPEED LIMITS'] | CMD='move at 1.2 m/s in human area'
```
