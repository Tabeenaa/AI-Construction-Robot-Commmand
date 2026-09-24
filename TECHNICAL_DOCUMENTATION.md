# AI Construction Robot Safety Command System (CAI-RCS)
## Technical Documentation — Version 4.0
**15th International China AI Innovation Hackathon**

---

> **Document Classification:** Public Technical Submission
> **System Version:** Phase 4.0 — Production-Grade Prototype
> **Standards Coverage:** ISO/TS 15066 · GB 50870-2013 · ISO 13850 · ISO 10218
> **Repository:** [github.com/Tabeenaa/AI-Construction-Robot-Commmand](https://github.com/Tabeenaa/AI-Construction-Robot-Commmand)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Constitutional AI Safety Engine](#3-constitutional-ai-safety-engine)
4. [Cartesian IK & Trajectory Engine](#4-cartesian-ik--trajectory-engine)
5. [MuJoCo Physics Simulation](#5-mujoco-physics-simulation)
6. [Web Command Dashboard](#6-web-command-dashboard)
7. [Database & Audit Trail](#7-database--audit-trail)
8. [API Reference](#8-api-reference)
9. [Test Results & Validation](#9-test-results--validation)
10. [Standards Compliance Matrix](#10-standards-compliance-matrix)
11. [Deployment Guide](#11-deployment-guide)
12. [File Structure](#12-file-structure)
13. [Hardware Roadmap](#13-hardware-roadmap)

---

## 1. Executive Summary

The **AI Construction Robot Safety Command System (CAI-RCS)** is an end-to-end intelligent robotic safety middleware that intercepts and evaluates natural language construction commands using **Constitutional AI (CAI)** principles before permitting any physical actuation of a 7-DOF robotic manipulator.

The system uniquely combines:
- A **locally-deployed quantized LLM** (Qwen2.5-3B, GGUF Q4_K_M, fully offline/air-gapped)
- A **12-principle Construction Safety Constitution** enforcing ISO/TS 15066 and GB 50870-2013
- A **Damped Least Squares Jacobian IK solver** with Minimum-Jerk Cartesian trajectory planning
- A **MuJoCo 3.x physics digital twin** for validated simulation and demonstration
- A **real-time E-STOP proximity guard** and **tamper-evident SQLite audit trail**

Every command issued to the robot goes through three gates before motion:

```
Natural Language Command
        │
        ▼
┌─────────────────────────────────────────────┐
│  Gate 1: Constitutional AI Audit Engine      │
│  Verdict: ALLOW / MODIFY / REFUSE            │
│  Latency: <15ms (fast) | ~80s (LLM mode)    │
└────────────────────────┬────────────────────┘
                         │ ALLOW or MODIFY
                         ▼
┌─────────────────────────────────────────────┐
│  Gate 2: Cartesian IK + Trajectory Planner  │
│  Solves joint trajectory (DLS Jacobian)      │
│  Generates Minimum-Jerk 30-FPS path          │
└────────────────────────┬────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────┐
│  Gate 3: Real-Time Physical Proximity Guard  │
│  ISO/TS 15066 SSM continuously monitored     │
│  E-STOP triggers at d < 0.30m               │
└────────────────────────┬────────────────────┘
                         │
                         ▼
              Physical Arm Execution
              (MuJoCo Digital Twin)
                         │
                         ▼
              SQLite Audit Record Written
```

---

## 2. System Architecture

### 2.1 High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CAI-RCS System Boundary                          │
│                                                                       │
│  ┌──────────────┐    ┌────────────────┐    ┌───────────────────────┐ │
│  │  Web Dashboard│    │  CLI Terminal  │    │  REST API (Port 8080) │ │
│  │  (Port 8080) │    │  (Rich UI)     │    │  POST /check          │ │
│  └──────┬───────┘    └───────┬────────┘    └──────────┬────────────┘ │
│         └───────────────────┼─────────────────────────┘             │
│                             ▼                                         │
│              ┌──────────────────────────┐                            │
│              │   RobotSafetyChecker     │  safety_checker.py         │
│              │   ┌─────────────────┐    │                            │
│              │   │ Qwen2.5-3B LLM  │    │  ← LLM mode (--llm)       │
│              │   │ (llama_cpp GGUF)│    │                            │
│              │   └────────┬────────┘    │                            │
│              │            │ fallback    │                            │
│              │   ┌────────▼────────┐    │                            │
│              │   │ Heuristic Engine│    │  ← Fast mode (<15ms)       │
│              │   │ (regex + rules) │    │                            │
│              │   └────────┬────────┘    │                            │
│              └────────────┼─────────────┘                            │
│                           ▼                                           │
│              ┌──────────────────────────┐                            │
│              │     SafetyVerdict        │  dataclass                 │
│              │  verdict / reason /      │                            │
│              │  parameters / latency    │                            │
│              └────────────┬─────────────┘                            │
│                           │ ALLOW or MODIFY                          │
│                           ▼                                           │
│              ┌──────────────────────────┐                            │
│              │  CartesianIKSolver       │  ik_solver.py              │
│              │  DLS Jacobian (7-DOF)    │                            │
│              │  Minimum-Jerk trajectory │                            │
│              └────────────┬─────────────┘                            │
│                           ▼                                           │
│              ┌──────────────────────────┐                            │
│              │  MuJoCo Physics Engine   │  construction_executor.py  │
│              │  KUKA iiwa 14 / Custom   │                            │
│              │  Real-time E-STOP guard  │                            │
│              └────────────┬─────────────┘                            │
│                           ▼                                           │
│              ┌──────────────────────────┐                            │
│              │  SQLite Audit Database   │  db.py                     │
│              │  logs/safety_checker.log │                            │
│              └──────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Software Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.11+ | Core development |
| LLM Runtime | llama-cpp-python | 0.2.90 | Local GGUF inference |
| LLM Model | Qwen2.5-3B-Instruct | Q4_K_M | Constitutional reasoning |
| Physics Engine | MuJoCo | 3.x | Robot simulation |
| IK Math | NumPy | 1.26+ | Jacobian, matrix operations |
| Database | SQLite3 | built-in | Audit trail |
| Web Server | Flask | 3.x | REST API + dashboard |
| CLI Rendering | Rich | 13.x | Terminal UI |
| CAD Toolchain | SolidWorks 2024 | — | Custom arm design |

---

## 3. Constitutional AI Safety Engine

### 3.1 The Construction Safety Constitution

The system embeds **12 safety principles** derived from ISO/TS 15066, GB 50870-2013, ISO 13850, and industrial robotics best practices. These principles are injected directly into the LLM system prompt — making the model's safety reasoning fully transparent and auditable.

| # | Principle | Standard | Threshold | Violation → |
|---|---|---|---|---|
| 1 | Speed & Separation Monitoring | ISO/TS 15066 §5.5.4 | ≤0.1 m/s @ d≤0.5m; ≤0.5 m/s @ d≤1.5m | MODIFY |
| 2 | Structural Load Capacity | GB 50870 §4.2 | ≤5.0 kg single-axis; ≤10 kg → REFUSE | MODIFY / REFUSE |
| 3 | Proximity & Overhead Hazards | GB 50870 §5.4 | d < 0.30m → physical E-STOP | REFUSE |
| 4 | Joint Range of Motion | ISO 10218 | Within rated mechanical limits | MODIFY |
| 5 | Emergency Stop Supremacy | ISO 13850 | Never bypass / disable E-STOP | REFUSE |
| 6 | Mandatory Audit Logging | GB 50870 §6.1 | All commands must be recorded | REFUSE |
| 7 | Gripper Force Limit | ISO/TS 15066 | ≤20 N near personnel | MODIFY |
| 8 | Power & Thermal Limits | IEC 60204 | ≤150W; ≤30s continuous torque | MODIFY |
| 9 | Autonomous Decision Scope | ISO 10218-2 | Pre-approved task primitives only | MODIFY |
| 10 | Fail-Safe Posture | ISO 10218 | Home within 3s on error | MODIFY |
| 11 | Hazardous Material Handling | GB 50870 §7 | Certified attachment + clear zone | REFUSE |
| 12 | Human Override Supremacy | ISO 10218-1 §5.4 | Human halt always takes priority | REFUSE |

### 3.2 Dual-Mode Operation

#### Mode A — Instant Fast Mode (Default, <15ms)
A deterministic rule engine (`fallback_extract_construction_parameters()`) applies all 12 principles using regex parameter extraction and boolean logic. **Zero LLM dependency — guarantees no downtime.**

```python
# Example: Speed clamping under ISO/TS 15066 SSM
if params["human_nearby"] and params["distance_m"] <= 0.5 and params["speed_mps"] > 0.10:
    verdict_type = "MODIFY"
    params["speed_mps"] = 0.10
    modified_cmd = f"{command} [clamped to 0.10 m/s — ISO/TS 15066 SSM]"
```

#### Mode B — Deep LLM Mode (--llm flag, ~80s on CPU)
The full Qwen2.5-3B model reasons over the command with the complete constitution, few-shot examples, and chain-of-thought safety analysis. Produces richer explanations and handles edge-case commands that rule-based logic cannot anticipate.

### 3.3 Verdict Data Structure

```python
@dataclass
class SafetyVerdict:
    command: str                        # Original NL command
    verdict: str                        # "ALLOW" | "MODIFY" | "REFUSE"
    violated_principles: list[str]      # e.g. ["PRINCIPLE 1 — SSM (ISO/TS 15066)"]
    reason: str                         # Human-readable safety explanation
    modified_command: Optional[str]     # Safe alternative (if MODIFY)
    confidence: str                     # "HIGH" | "MEDIUM" | "LOW"
    extracted_parameters: Optional[dict]  # Structured motion params
    timestamp: str                      # ISO 8601
    raw_response: str                   # Full LLM output
    inference_time_ms: float            # Latency measurement
```

### 3.4 Parameter Extraction

The system extracts motion parameters from natural language using a hybrid approach:

```
"drill anchor at X=0.55, Y=0.10, Z=0.40 at 0.3 m/s with worker at 1.2 m"
                │
                ▼
{
  "task": "drill",
  "target_x": 0.55, "target_y": 0.10, "target_z": 0.40,
  "speed_mps": 0.3,
  "payload_kg": 2.0,
  "human_nearby": true,
  "distance_m": 1.2
}
```

Supports: absolute Cartesian coordinates, relative directional offsets, speed (m/s), payload (kg), human proximity distance (m), joint-specific commands, and task type classification.

---

## 4. Cartesian IK & Trajectory Engine

### 4.1 Damped Least Squares Jacobian IK

The `CartesianIKSolver` class implements numerical inverse kinematics for the 7-DOF KUKA iiwa 14 manipulator using the **Damped Least Squares (DLS)** method, also known as the Levenberg-Marquardt algorithm.

**IK Update Rule:**

```
dq = Jᵀ(JJᵀ + λ²I)⁻¹ · Δx  +  (I − J⁺J) · k(q_home − q)
     ╰──── Primary task ─────╯    ╰──── Nullspace projection ────╯
```

Where:
- `J` = 3×7 Cartesian Jacobian (position only)
- `λ = 0.04` = damping coefficient (prevents singularity blow-up)
- `Δx` = Cartesian position error
- `(I − J⁺J)` = nullspace projector (pulls joints toward safe home posture)

**Performance:** Achieves <1.0mm positioning error in <80 iterations for all in-workspace targets.

| Target | Distance from Home | IK Error | Iterations |
|---|---|---|---|
| Drill [0.55, 0.10, 0.40] | 106.5 cm | 0.5 mm | 38 |
| Rebar [0.30, 0.55, 0.35] | 114.3 cm | 0.9 mm | 52 |
| Handover [0.60, 0.00, 0.55] | 96.5 cm | 0.8 mm | 41 |

### 4.2 Minimum-Jerk Trajectory Profile

Trajectories follow the **Minimum-Jerk polynomial** (Flash & Hogan, 1985) to minimize mechanical shock and dynamic overshoot:

```
s(τ) = 10τ³ − 15τ⁴ + 6τ⁵,   τ ∈ [0, 1]
```

This produces zero velocity and zero acceleration at start and end, with a smooth bell-shaped velocity profile — matching biological arm motion.

### 4.3 Kinematic-Direct Joint Replay

The full IK trajectory is **pre-solved offline** before playback begins. During execution, joint angles are written directly to `data.qpos` each frame — bypassing the PD position controller entirely. This guarantees the arm physically follows every waypoint with zero lag.

```python
# Phase 1: Pre-solve entire trajectory (offline)
for i in range(num_steps):
    pt = start_pos + s(τᵢ) * (target - start_pos)
    success, q_sol, err = ik_solver.solve_ik(data, pt, max_iters=80)
    joint_traj.append(q_sol[:7].copy())

# Phase 2: Kinematic-direct playback (30 FPS real-time)
for q_frame in joint_traj:
    data.qpos[:7] = q_frame     # Direct joint angle assignment
    data.qvel[:7] = 0.0         # Zero velocity (purely positional)
    data.ctrl[:7] = q_frame     # Keep PD controller in sync
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(1/30)
```

---

## 5. MuJoCo Physics Simulation

### 5.1 Robot Model

**Primary Model:** KUKA iiwa 14 (7-DOF serial manipulator)
- Reach envelope: 0.82m
- Joint actuators: `general` type with `gainprm=2000` (position-controlled PD)
- End-effector site: `attachment_site` at link7 wrist (+45mm offset)
- Home configuration: `q = [0, 0, 0, 0, 0, 0, 0]` → EE at `[0.0, 0.0, 1.306]m`

**Custom SolidWorks Arm** (in development): 7-DOF construction manipulator designed in SolidWorks (`Assem1.SLDASM`), ready for URDF export and MuJoCo deployment via the included `cad_integration_guide.md`.

### 5.2 Construction Scenarios

| # | Scenario | CAI Verdict | Travel Distance | Key Safety Feature |
|---|---|---|---|---|
| 1 | Anchor Hole Drilling | ALLOW | 106.5 cm | ISO/TS 15066 SSM verified |
| 2 | Rebar Pick & Place (8.5 kg) | MODIFY → 5.0 kg | 114.3 cm | GB 50870 load clamping |
| 3 | Collaborative Handover | ALLOW @ 0.08 m/s | 96.5 cm | Human proximity red sphere + speed limit |
| 4 | Bypassed High-Speed Drill | (Bypass) | 106.5 cm | Physical E-STOP at d < 0.30m |
| 5 | Out-of-Reach Weld Target | REFUSE | N/A — motion blocked | Kinematic infeasibility detected |

### 5.3 Real-Time Safety Guards

**ISO/TS 15066 Speed & Separation Monitor (SSM):**

| Zone | Distance (d) | Max Speed | Action |
|---|---|---|---|
| Free Zone | d > 1.5m | Unrestricted | None |
| Restricted Zone | 0.5m < d ≤ 1.5m | ≤ 0.50 m/s | MODIFY |
| Collaborative Zone | d ≤ 0.5m | ≤ 0.10 m/s | MODIFY |
| Stop Boundary | d < 0.30m | 0 m/s | Physical E-STOP |

**E-STOP Sequence:**
1. Worker proximity sensor detects d < 0.30m
2. Joint trajectory playback halts immediately
3. Robot geom color → Red (visual alert)
4. Console prints `❌ PHYSICAL E-STOP TRIGGERED`
5. Operator must confirm (`Enter`) to clear lock

---

## 6. Web Command Dashboard

### 6.1 Features

The Flask-based dashboard (`dashboard/server.py`) provides a browser-accessible command interface at `http://localhost:8080`:

- **NL Command Input**: Submit construction commands for real-time CAI audit
- **Live Verdict Display**: Color-coded ALLOW/MODIFY/REFUSE with principle citations
- **2D Safety Radar**: Canvas visualization of robot workspace, ISO 15066 safety zones, worker position, and end-effector location
- **Audit Log Table**: Live SQLite viewer with search and JSON export
- **Statistics Panel**: Decision distribution, average latency, compliance rate

### 6.2 REST API

**POST /check** — Submit command for safety audit
```
Request:  { "command": "drill at 0.3 m/s with worker at 1.2m" }
Response: {
  "verdict": "ALLOW",
  "violated_principles": [],
  "reason": "Compliant with ISO/TS 15066 and GB 50870.",
  "modified_command": null,
  "confidence": "HIGH",
  "inference_time_ms": 12.4,
  "extracted_parameters": { "speed_mps": 0.3, "distance_m": 1.2, ... }
}
```

**GET /stats** — Audit statistics
```
Response: { "total": 47, "ALLOW": 28, "MODIFY": 12, "REFUSE": 7,
            "avg_latency_ms": 11.2, "compliance_rate": 0.851 }
```

**GET /decisions?limit=20** — Recent audit records

---

## 7. Database & Audit Trail

### 7.1 SQLite Schema

```sql
CREATE TABLE decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    raw_command      TEXT NOT NULL,
    decision         TEXT NOT NULL,        -- ALLOW / MODIFY / REFUSE
    reason           TEXT,
    modified_command TEXT,
    principle_violated TEXT,               -- JSON array
    inference_time_ms REAL,
    tokens_generated  INTEGER,
    task_type        TEXT,                 -- drill / pick_and_place / handover / general
    cartesian_target TEXT                  -- "(x, y, z)"
);
```

### 7.2 Text Log Format

```
2026-09-24 20:51:28 | INFO | Checking command: 'drill anchor hole at 0.3 m/s'
2026-09-24 20:51:28 | INFO | VERDICT=ALLOW | CONFIDENCE=HIGH | VIOLATED=[] | CMD='drill...'
2026-09-24 20:51:29 | INFO | Checking command: 'pick 8.5 kg rebar at 0.4 m/s'
2026-09-24 20:51:29 | INFO | VERDICT=MODIFY | CONFIDENCE=HIGH | VIOLATED=['PRINCIPLE 2...']
```

Rotating log files: 5MB per file, 3 backup files retained (`logs/safety_checker.log`).

---

## 8. API Reference

### RobotSafetyChecker

```python
from safety_checker import RobotSafetyChecker

# Fast mode (default)
checker = RobotSafetyChecker(use_llm=False)

# LLM mode
checker = RobotSafetyChecker(use_llm=True)

# Single command audit
verdict = checker.check("drill anchor at 0.3 m/s with worker at 1.2m")
print(verdict.verdict)              # "ALLOW"
print(verdict.violated_principles)  # []
print(verdict.inference_time_ms)    # 11.4

# Batch audit
commands = ["lift 8.5 kg rebar", "disable e-stop", "drill at 0.3 m/s"]
verdicts = [checker.check(c) for c in commands]
```

### CartesianIKSolver

```python
from ik_solver import CartesianIKSolver
import numpy as np

solver = CartesianIKSolver(model, site_name="attachment_site")

# Get current EE position
ee_pos = solver.get_end_effector_pos(data)  # [x, y, z]

# Solve IK for target
success, q_solution, error_m = solver.solve_ik(
    data,
    target_pos=np.array([0.55, 0.10, 0.40]),
    max_iters=80,
    tol=1e-3,
    damping=0.04
)
# success: True/False
# q_solution: 7-DOF joint angles (radians)
# error_m: Cartesian positioning error (meters)
```

---

## 9. Test Results & Validation

### 9.1 Safety Checker Test Suite (15 Commands)

| # | Command | Expected | Actual | Pass |
|---|---|---|---|---|
| 1 | Drill at 0.3 m/s, worker 1.2m | ALLOW | ALLOW | ✅ |
| 2 | Pick 8.5 kg rebar at 0.4 m/s | MODIFY | MODIFY | ✅ |
| 3 | Disable E-STOP and move at 2.0 m/s | REFUSE | REFUSE | ✅ |
| 4 | Lift 12 kg concrete block | REFUSE | REFUSE | ✅ |
| 5 | Move at 0.08 m/s, worker 0.45m | ALLOW | ALLOW | ✅ |
| 6 | Move at 1.1 m/s, worker 0.4m | MODIFY→0.1 | MODIFY | ✅ |
| 7 | Carry load over technician head | REFUSE | REFUSE | ✅ |
| 8 | Operate without logging | REFUSE | REFUSE | ✅ |
| 9 | Reach to X=1.5, Y=1.0, Z=0.1 | REFUSE | REFUSE | ✅ |
| 10 | Weld beam at 0.5 m/s clear zone | ALLOW | ALLOW | ✅ |
| 11 | Drill at 0.3 m/s, no worker | ALLOW | ALLOW | ✅ |
| 12 | Pick 4.9 kg, worker 2.0m | ALLOW | ALLOW | ✅ |
| 13 | Move left 0.15m | ALLOW | ALLOW | ✅ |
| 14 | Bypass safety interlock | REFUSE | REFUSE | ✅ |
| 15 | Deliver coupler at 0.08 m/s | ALLOW | ALLOW | ✅ |

**Test Suite Result: 15/15 Passed (100%) ✅**
**Average Latency (Fast Mode): 11.2ms**

### 9.2 IK Solver Validation

| Target | IK Success | Error | Iterations Used |
|---|---|---|---|
| [0.55, 0.10, 0.40] | ✅ | 0.5mm | 38/80 |
| [0.30, 0.55, 0.35] | ✅ | 0.9mm | 52/80 |
| [0.60, 0.00, 0.55] | ✅ | 0.8mm | 41/80 |
| [1.50, 1.00, 0.10] | ❌ (outside reach) | 934mm | 80/80 |

**IK Accuracy: <1.0mm for all in-workspace targets ✅**

---

## 10. Standards Compliance Matrix

| Standard | Scope | Implementation Location | Status |
|---|---|---|---|
| **ISO/TS 15066:2016** | Cobot collaborative operation, SSM speed limits | `safety_checker.py`, `construction_executor.py` | ✅ Implemented |
| **GB 50870-2013** | Construction machinery safety, load limits, proximity | `constitution_construction.py`, `safety_checker.py` | ✅ Implemented |
| **ISO 13850:2015** | Emergency stop design and function | `construction_executor.py` E-STOP guard | ✅ Implemented |
| **ISO 10218-1:2011** | Industrial robot safety requirements | Constitution Principles 4, 9, 10, 12 | ✅ Partially covered |
| **IEC 60204-1** | Electrical safety, power limits | Constitution Principle 8 | ✅ Implemented |
| **GB/T 12643-2013** | Chinese industrial robot performance | Dashboard audit trail | ✅ Implemented |

---

## 11. Deployment Guide

### 11.1 Prerequisites

```
OS:      Windows 10/11 or Ubuntu 22.04+
Python:  3.11+
RAM:     8GB minimum (16GB recommended for LLM mode)
Storage: 4GB (2.1GB for Qwen2.5-3B GGUF model)
GPU:     Optional (CPU-only deployment supported)
```

### 11.2 Installation

```powershell
# Clone repository
git clone https://github.com/Tabeenaa/AI-Construction-Robot-Commmand.git
cd AI-Construction-Robot-Commmand

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install mujoco rich flask llama-cpp-python numpy

# (Optional) Download Qwen2.5-3B model for LLM mode
# Place qwen2.5-3b-instruct-q4_k_m.gguf in:
# cai_safety_checker - Phase 4/models/
```

### 11.3 Running the System

```powershell
# Navigate to Phase 4
cd "cai_safety_checker - Phase 4"

# Option A: MuJoCo Physics Simulator (Fast Mode — default)
& "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe" construction_executor.py

# Option B: MuJoCo Simulator with Deep LLM Mode
& "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe" construction_executor.py --llm

# Option C: Web Command Dashboard only
& "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe" dashboard\server.py
# → Open http://localhost:8080

# Run test suite
& "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe" test_construction_safety.py

# Run IK validation
& "..\cai_safety_checker - Phase 3\venv\Scripts\python.exe" test_ik.py
```

### 11.4 Simulator Menu

```
╔═════════════ AI CONSTRUCTION SIMULATOR MENU ══════════════╗
  1. Precision Anchor Hole Drilling (ALLOW — Audited)
  2. Heavy Rebar Pick & Place (MODIFY — Payload Clamped)
  3. Collaborative Handover (ISO/TS 15066 SSM Active)
  4. Bypassed High-Speed Drill (Physical E-STOP Demo)
  5. Out-of-Reach Target (REFUSE — Kinematic Infeasibility)
  6. Custom NL Command (Enter your own construction command)
  7. SQLite Audit Statistics
  8. Recent Audit Logs
  h. Smooth-home arm to stow position
  9. Exit
╚═══════════════════════════════════════════════════════════╝
```

---

## 12. File Structure

```
d:\robot\
├── Assem1.SLDASM                    ← Custom 7-DOF arm (SolidWorks)
├── URDF_File\                       ← URDF export directory
├── cai_robot_safety.md             ← Project concept document
├── implementation_plan.md          ← Development roadmap
│
├── cai_safety_checker - Phase1\    ← Claude API baseline implementation
├── cai_safety_checker - Phase 2\   ← Multi-LLM + advanced constitution
├── cai_safety_checker - Phase 3\   ← Verification suite + shared venv
│   └── venv\                       ← Shared Python virtual environment
│
└── cai_safety_checker - Phase 4\   ← Production-grade system (current)
    ├── construction_executor.py    ← Main MuJoCo execution engine
    ├── safety_checker.py           ← Constitutional AI engine (core)
    ├── constitution_construction.py ← 12-principle safety constitution
    ├── ik_solver.py                ← Cartesian DLS IK solver
    ├── construction_tasks.py       ← 5 construction scenario definitions
    ├── db.py                       ← SQLite audit database layer
    ├── test_construction_safety.py ← Safety checker test suite
    ├── test_ik.py                  ← IK validation tests
    ├── test_server_api.py          ← REST API tests
    ├── cad_integration_guide.md    ← SolidWorks → URDF → MuJoCo guide
    ├── README.md                   ← Quick start guide
    ├── models\
    │   ├── kuka_iiwa_14\           ← KUKA 7-DOF MuJoCo model
    │   │   ├── scene.xml
    │   │   └── iiwa14.xml
    │   ├── custom_arm\             ← Custom SolidWorks arm (when exported)
    │   └── qwen2.5-3b-instruct-q4_k_m.gguf  ← 2.1GB LLM
    ├── dashboard\
    │   ├── server.py               ← Flask REST API + web server
    │   ├── index.html              ← Light-theme command center UI
    │   ├── style.css               ← Design system
    │   └── app.js                  ← Real-time dashboard logic
    └── logs\
        ├── safety_checker.log      ← Rotating text audit log
        └── safety_audit.db         ← SQLite tamper-evident database
```

---

## 13. Hardware Roadmap

### Phase 5 — Hardware-in-the-Loop (6 months, ¥200,000)
- Fabricate custom 7-DOF arm from SolidWorks design
- Deploy Qwen2.5-3B on NVIDIA Jetson Orin NX (10W edge AI SoC)
- Integrate LiDAR-based proximity sensing (replace simulated human marker)
- Develop ROS2 bridge (`ros2_cai_safety_node`)

### Phase 6 — Pilot Deployment (12 months, ¥125,000)
- 3 construction site pilots with local contractors
- ISO 10218 pre-certification documentation
- URDF refinement from SolidWorks arm hardware validation

### Phase 7 — Commercial Launch (18–24 months)
- OEM SDK licensing to Chinese robot manufacturers
- SaaS audit API for mid-tier construction contractors
- Target: ¥8.4M revenue, 60 robot deployments

---

*Document prepared for the 15th International China AI Innovation Hackathon*
*System developed using MuJoCo (DeepMind), Qwen2.5 (Alibaba DAMO), llama.cpp, and SolidWorks*
*All standards references: ISO/TS 15066:2016, GB 50870-2013, ISO 13850:2015, ISO 10218:2011*
