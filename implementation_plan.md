# Implementation Plan: Contest-Grade Autonomous Construction Robotic Safety System (Phase 4)

Upgrade the existing Constitutional AI robot safety checker into an autonomous, standards-compliant, contest-ready system for the **15th China AI Innovation in AI Construction** competition.

This plan integrates **Cartesian Inverse Kinematics (IK)**, **ISO/TS 15066 & GB 50870 standards-based safety rules**, **construction task primitives**, and a **real-time Web Command Center**.

---

## User Review Required

> [!IMPORTANT]
> **Robotics Architecture Choice**:
> - We will implement **Cartesian Inverse Kinematics (IK)** on the 7-DOF manipulator so the robot accepts 3D task-space goals $(X, Y, Z, \text{orientation})$ instead of just 1-DOF joint rotations.
> - For environment awareness, we frame the perception system as **Dynamic 3D Workspace Perception & Speed-Separation-Monitoring (SSM)** rather than mobile SLAM, ensuring strict alignment with industrial robotics standards.
> - We will create a clean new workspace directory `d:\robot\cai_safety_checker - Phase 4` so Phases 1, 2, and 3 remain untouched as reference milestones.

---

## Proposed Changes

### Component 1: Cartesian Inverse Kinematics (IK) Engine

We will build a numerical IK solver leveraging MuJoCo's analytical Jacobian calculation (`mujoco.mj_jac`) with Damped Least Squares (DLS) and nullspace joint-limit avoidance.

#### [NEW] [ik_solver.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/ik_solver.py)
- **Functions**:
  - `solve_ik_cartesian(model, data, target_pos, target_quat=None, max_steps=50, tol=1e-3, damping=0.05)`: Computes target joint configuration $q^*$ for a given 3D end-effector goal.
  - `plan_cartesian_trajectory(start_pos, goal_pos, duration, dt, max_cartesian_vel)`: Generates a smooth minimum-jerk or trapezoidal Cartesian path.
  - Nullspace projection to keep joints near comfortable mid-ranges and avoid mechanical limits.

---

### Component 2: Construction Safety Constitution V2 (ISO 15066 & GB 50870)

Upgrade the safety constitution from general laboratory thresholds to accredited civil construction and collaborative robot safety standards.

#### [NEW] [constitution_construction.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/constitution_construction.py)
- Injects formal citations and quantitative curves:
  - **ISO/TS 15066 Clause 5.5.4 (Speed and Separation Monitoring - SSM)**: Calculates dynamic protective separation distance:
    $$S_p = (v_h \cdot T_r) + (v_r \cdot T_r) + B_r + C$$
    Where robot speed $v_r$ must scale down dynamically as worker distance decreases.
  - **ISO/TS 15066 Clause 5.5.5 (Power and Force Limiting - PFL)**: End-effector pressure & force clamping ($\le 140\text{ N}$ transient, $\le 70\text{ N}$ quasi-static).
  - **GB 50870-2013 (Construction Machinery Safety)**: Crane drop-zone avoidance, overhead collision bounds, mandatory acoustic/visual safety signaling.
  - **GB 11291.2-2013**: Industrial robot collaborative interlocks and human override supremacy.

---

### Component 3: Construction Task Scenarios & Physics Execution

#### [NEW] [construction_tasks.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/construction_tasks.py)
- Implementation of 3 specific AI Construction tasks:
  1. **Precision Anchor Hole Drilling / Surface Finishing**: Cartesian surface approach normal to a concrete wall surface, regulated feed rate, automatic pushback on unexpected torque.
  2. **Rebar / Prefab Component Pick-and-Place**: Coordinated 3D trajectory from supply bin to structural assembly point with payload verification.
  3. **Collaborative Human-Robot Material Handover**: Worker enters zone; CAI detects approach, applies SSM speed scaling, safely pauses arm when worker is within handover range ($0.3\text{ m}$).
- Comparative execution modes:
  - **Audited (CAI Safe)**: Automatically modifies speeds and paths to guarantee zero incident.
  - **Bypassed (Unsafe Near-Miss)**: Demonstrates what happens without CAI (near-miss collision, mechanical stop impact, triggering emergency physical stop).

#### [NEW] [construction_executor.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/construction_executor.py)
- Integrated execution loop combining MuJoCo, IK solver, local Qwen GGUF CAI checker, and SQLite logging.
- Real-time visual feedback:
  - Green / Cyan: Normal compliant Cartesian motion.
  - Yellow: ISO 15066 Speed-Separation deceleration active.
  - Red: Safety Refusal / E-STOP active.

---

### Component 4: Real-time Web Command Center & Telemetry Cockpit

Replace the terminal CLI with a web application to impress competition judges.

#### [NEW] [dashboard/server.py](file:///d:/robot/cai_safety_checker%20-%20Phase%204/dashboard/server.py)
- Lightweight Python HTTP/WebSocket server streaming real-time metrics at 20 Hz:
  - Current Cartesian EE coordinates $(X, Y, Z)$ and linear speed $(\text{m/s})$.
  - Worker proximity distance and dynamic ISO 15066 safe speed cap.
  - Live CAI Critique & Revision verdicts with principle citations.
  - SQLite historical audit log feed.

#### [NEW] [dashboard/index.html](file:///d:/robot/cai_safety_checker%20-%20Phase%204/dashboard/index.html) & [dashboard/app.js](file:///d:/robot/cai_safety_checker%20-%20Phase%204/dashboard/app.js)
- Industrial dark mode interface featuring:
  - **Live Telemetry Gauges**: Cartesian velocity, payload, worker separation distance.
  - **CAI Live Audit Terminal**: Live streaming of LLM critique, violated principles, and revision.
  - **Interactive Natural Language Dispatcher**: Send commands directly from the browser.
  - **Safety Zone Radar**: 2D top-down visualizer of the robot and human position.

---

### Component 5: Custom SolidWorks CAD Integration Guide

#### [NEW] [cad_integration_guide.md](file:///d:/robot/cai_safety_checker%20-%20Phase%204/cad_integration_guide.md)
- Complete guide tailored for Member 3 on the team:
  - Converting the SolidWorks files (`Assem1.SLDASM`, `rotor.SLDPRT`, etc.) to STL / URDF using the SolidWorks to URDF Exporter plugin.
  - Defining joint coordinate frames, mass matrices, and collision geometries.
  - Swapping the generic KUKA model with your custom arm in `construction_executor.py`.

---

## Verification Plan

### Automated & Physics Verification
1. **IK Accuracy & Convergence**:
   - Run `test_ik_solver.py` across 100 random reachable Cartesian points in the workspace.
   - Verify position error $< 1\text{ mm}$ and no joint limit violations.
2. **Constitutional Safety Compliance**:
   - Test suite of 20 construction commands verifying ALLOW, MODIFY (speed/force clamping), and REFUSE verdicts against ISO/GB standards.
3. **MuJoCo Simulation Run**:
   - Run `python construction_executor.py` verifying all 3 construction task scenarios execute smoothly without jitter.
4. **Dashboard Connectivity**:
   - Launch `python dashboard/server.py`, open browser at `http://localhost:8080`, verify live telemetry updates and command dispatch.
