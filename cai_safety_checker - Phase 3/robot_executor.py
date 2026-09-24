"""
robot_executor.py
=================
MuJoCo-based Robot Command Executor with Constitutional AI (CAI) safety audits.

Key Features:
1. Passive rendering viewer (non-blocking) using mujoco.viewer.launch_passive.
2. Implements real-time physical safety monitoring (joint range constraints & proximity checks).
3. Real-time visual feedback: changes robot color to red when REFUSED or E-STOPPED.
4. Simulates payload mass on the end-effector (link7) dynamically using mj_setConst.
5. Fully integrated with Local Qwen-based Constitutional AI safety checker and SQLite logging.
"""

import os
import re
import sys
import time
import json
import math
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Prevent glfw warning/noise on startup
os.environ["PYBULLET_SYSTEM_ENCODING"] = "utf8"

# Add parent directory to path to ensure modules are importable
sys.path.append(str(Path(__file__).parent))

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:
    print("[ERROR] Failed to import MuJoCo. Please run 'pip install mujoco' first.")
    print("Error details:", e)
    sys.exit(1)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import safety checker and database modules
from safety_checker import RobotSafetyChecker, SafetyVerdict
from db import log_decision, get_stats, get_decisions

console = Console()

# ─── Configuration ────────────────────────────────────────────────────────────
XML_PATH = Path(__file__).parent / "models" / "kuka_iiwa_14" / "scene.xml"
DEFAULT_QPOS = [0, 0.785398, 0, -1.5708, 0, 0, 0]

# ─── Smoothing / Anti-glitch parameters ───────────────────────────────────────
# Dead-zone: ignore actuator error smaller than this (radians) to kill micro-jitter
DEAD_ZONE_RAD   = 0.005   # ~0.3 degrees — below this, hold still
# Rolling-average window for qpos readings (anti-noise)
AVG_WINDOW      = 5
# Smooth-home: number of seconds to glide back to DEFAULT_QPOS
SMOOTH_HOME_DUR = 1.5     # seconds

# Persistent human state: remembered across all NL commands in a session
# Updated whenever the user mentions a human and/or specifies a distance.
HUMAN_STATE = {
    "human_nearby": False,
    "distance_m": 1.2,   # default 1.2 m when human is placed
}

# Define command schema example
# {"joint_id": 3, "velocity": 0.75, "payload_kg": 8.0, "human_nearby": True, "distance_m": 1.2, "zone": "restricted"}

# ─── Helper Functions ──────────────────────────────────────────────────────────

def command_to_text(cmd: dict) -> str:
    """Convert a JSON command dict into a natural-language representation for the CAI checker."""
    joint_num = cmd["joint_id"] + 1  # 0-indexed to 1-indexed for operators
    vel_rad = cmd["velocity"]
    vel_deg = round(vel_rad * 180.0 / math.pi, 1)
    payload = cmd["payload_kg"]
    human = cmd["human_nearby"]
    dist = cmd["distance_m"]
    zone = cmd["zone"]
    
    parts = [
        f"rotate joint {joint_num} at {vel_deg} deg/s",
        f"with a payload of {payload} kg"
    ]
    
    # If the command is driving a joint past its mechanical limit, we represent this in text
    if "target_angle" in cmd:
        target_deg = round(cmd["target_angle"] * 180.0 / math.pi, 1)
        parts[0] = f"rotate joint {joint_num} to target angle {target_deg} degrees at {vel_deg} deg/s"
        
    if human:
        parts.append(f"in the {zone} zone with a human present at a distance of {dist} m")
    else:
        parts.append(f"in the {zone} zone with no humans present")
        
    return ", ".join(parts)


def set_robot_color(model, original_colors, color_rgba=None):
    """Change the color of all robot geoms to show visual safety feedback (e.g. Red for Locked/Refused)."""
    if color_rgba is None:
        # Restore original colors
        model.geom_rgba[:] = original_colors
    else:
        # Set all robot links to target color
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" not in geom_name and "human" not in geom_name:
                model.geom_rgba[geom_id] = color_rgba


# ─── Rolling average buffer (anti-noise for qpos readings) ────────────────────
_qpos_buffer: list = []  # filled lazily; each entry is a 7-element array


def _averaged_qpos(data) -> np.ndarray:
    """Return a rolling-average of the last AVG_WINDOW qpos readings.
    This dampens sensor noise that would otherwise cause micro-jitter in ctrl.
    """
    global _qpos_buffer
    _qpos_buffer.append(np.array(data.qpos[:7]))
    if len(_qpos_buffer) > AVG_WINDOW:
        _qpos_buffer.pop(0)
    return np.mean(_qpos_buffer, axis=0)


def _apply_dead_zone(ctrl: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    """Snap ctrl to qpos for joints whose error is within the dead-zone.
    This prevents the actuators fighting tiny numerical drift and causing glitch.
    """
    out = ctrl.copy()
    for i in range(len(out)):
        if abs(out[i] - qpos[i]) < DEAD_ZONE_RAD:
            out[i] = qpos[i]   # lock this joint in place
    return out


def reset_robot_state(model, data, payload_kg, human_nearby, distance_m, original_colors):
    """Reset the robot qpos, velocity, target payload, and human marker position.
    Ctrl is pinned to DEFAULT_QPOS so the position actuators are at rest
    (no error → no torque → no glitch).
    """
    global _qpos_buffer
    _qpos_buffer.clear()   # flush stale averages after a hard reset

    # Restore normal colors
    set_robot_color(model, original_colors, None)
    
    # Update payload mass at end-effector (link7)
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    default_link7_mass = 1.2  # Defined in iiwa14.xml
    model.body_mass[link7_id] = default_link7_mass + payload_kg
    
    # Position human marker mocap body
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
    mocap_idx = model.body_mocapid[mocap_id]
    if human_nearby:
        data.mocap_pos[mocap_idx] = [distance_m, 0.0, 0.5]
    else:
        data.mocap_pos[mocap_idx] = [0.0, 0.0, -10.0]
        
    # Recompute inertias / constraints
    mujoco.mj_setConst(model, data)
    
    # Hard-set joint positions, zero velocity, pin ctrl == qpos (zero actuator error)
    data.qpos[:7] = DEFAULT_QPOS
    data.qvel[:]  = 0
    data.qacc[:]  = 0
    data.ctrl[:7] = DEFAULT_QPOS   # ctrl == qpos → zero position error → zero torque
    
    mujoco.mj_forward(model, data)


def smooth_home_robot(model, data, viewer, original_colors):
    """Smoothly interpolate all joints back to DEFAULT_QPOS over SMOOTH_HOME_DUR seconds.

    Algorithm
    ---------
    * At each physics step, the *control target* is moved a fraction of the
      remaining error (proportional approach → exponential decay).
    * A dead-zone suppresses the last tiny residual so the robot snaps cleanly
      to the home pose without buzzing.
    * A rolling average of qpos is used for the error calculation to reject
      sensor/integration noise.
    """
    global _qpos_buffer
    _qpos_buffer.clear()

    home = np.array(DEFAULT_QPOS, dtype=np.float64)
    dt   = model.opt.timestep
    # Alpha determines how quickly we converge: alpha ≈ dt / (SMOOTH_HOME_DUR / 3)
    # A lower alpha = slower, smoother motion.
    alpha = dt / (SMOOTH_HOME_DUR / 3.0)
    alpha = float(np.clip(alpha, 0.001, 0.15))  # safety clamp

    # Current control target starts from wherever the robot is now
    ctrl_target = np.array(data.ctrl[:7], dtype=np.float64)

    total_steps = int(SMOOTH_HOME_DUR / dt) + 1
    console.print(f"[bold cyan]🏠 Smooth homing to DEFAULT_QPOS over {SMOOTH_HOME_DUR}s ...[/bold cyan]")

    for _ in range(total_steps):
        if not viewer.is_running():
            break

        step_start = time.time()

        # Averaged current position (noise-filtered)
        avg_qpos = _averaged_qpos(data)

        # Step the control target toward home
        error = home - ctrl_target
        ctrl_target += alpha * error

        # Apply dead-zone: freeze joints that are close enough
        ctrl_target = _apply_dead_zone(ctrl_target, home)

        data.ctrl[:7] = ctrl_target
        mujoco.mj_step(model, data)
        viewer.sync()

        # If every joint is within dead-zone of home, we're done
        if np.all(np.abs(data.qpos[:7] - home) < DEAD_ZONE_RAD * 2):
            break

        elapsed = time.time() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Final hard-lock to home
    data.qpos[:7] = home
    data.qvel[:]  = 0
    data.ctrl[:7] = home
    mujoco.mj_forward(model, data)
    _qpos_buffer.clear()
    viewer.sync()
    console.print("[bold green]✔ Robot returned to home position.[/bold green]")


def run_mujoco_execution(model, data, viewer, cmd: dict, bypass_audit: bool = False, original_colors=None, checker=None):
    """Execute the command in the MuJoCo simulator, performing safety audits and real-time monitoring."""
    joint_id = cmd["joint_id"]
    velocity = cmd["velocity"]
    payload = cmd["payload_kg"]
    human_nearby = cmd["human_nearby"]
    distance = cmd["distance_m"]
    zone = cmd["zone"]
    target_angle = cmd.get("target_angle", None)

    # 1. Convert to natural language command
    cmd_text = command_to_text(cmd)
    
    console.print(Panel(
        f"[bold white]Target Joint  :[/bold white] Joint {joint_id + 1}\n"
        f"[bold white]Velocity      :[/bold white] {velocity:.3f} rad/s ({round(velocity * 180.0 / math.pi, 1)} °/s)\n"
        f"[bold white]Payload Mass  :[/bold white] {payload:.1f} kg\n"
        f"[bold white]Human Nearby  :[/bold white] {human_nearby} (Distance: {distance:.2f} m, Zone: '{zone}')\n"
        f"[bold white]Translated NL :[/bold white] [cyan]\"{cmd_text}\"[/cyan]",
        title="[bold yellow]1. Command Dispatch[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED
    ))

    # 2. Constitutional AI safety audit
    verdict_type = "ALLOW"
    violated = []
    reason = "Bypassed by operator. Unsafe direct run."
    modified_command = None
    exec_payload = payload
    exec_velocity = velocity

    if not bypass_audit:
        console.print("[dim]Querying Local Constitutional AI Safety Checker (Gemini/Qwen GGUF)...[/dim]")
        start_time = time.perf_counter()
        verdict = checker.check(cmd_text)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        verdict_type = verdict.verdict
        violated = verdict.violated_principles
        reason = verdict.reason
        modified_command = verdict.modified_command
        
        v_color = {"ALLOW": "green", "MODIFY": "yellow", "REFUSE": "red"}.get(verdict_type, "white")
        console.print(Panel(
            f"[bold white]Verdict   :[/bold white] [{v_color}]{verdict_type}[/{v_color}]\n"
            f"[bold white]Violated  :[/bold white] {violated if violated else 'None'}\n"
            f"[bold white]Reason    :[/bold white] {reason}\n"
            f"[bold white]Modified  :[/bold white] {modified_command if modified_command else 'N/A'}\n"
            f"[bold white]Latency   :[/bold white] {latency_ms:.1f} ms",
            title="[bold yellow]2. Constitutional AI Audit Verdict[/bold yellow]",
            border_style=v_color,
            box=box.ROUNDED
        ))
        
        if verdict_type == "REFUSE":
            console.print("[bold red]🚫 COMMAND REFUSED. Locking robot arm for safety.[/bold red]")
            # Visual feedback: Turn robot red and keep locked
            set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
            reset_robot_state(model, data, 0.0, human_nearby, distance, original_colors)
            set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
            
            console.print("[bold yellow]👀 Look at the MuJoCo simulator window to see the RED locked state.[/bold yellow]")
            viewer.sync()
            input("Press Enter in this terminal to release safety lock and return to menu...")
            set_robot_color(model, original_colors, None) # restore
            return
            
        elif verdict_type == "MODIFY":
            console.print("[bold yellow]⚠️ COMMAND MODIFIED. Capping parameters to constitutional safe limits.[/bold yellow]")
            # Apply constitutional caps
            if "PRINCIPLE 2" in str(violated):
                exec_payload = 5.0
                console.print(f"  [dim]↳ Payload capped: {payload} kg -> 5.0 kg[/dim]")
            if "PRINCIPLE 1" in str(violated):
                # Cap velocity to 45 deg/s = 0.785 rad/s
                sign = 1.0 if velocity >= 0 else -1.0
                exec_velocity = sign * 0.785
                console.print(f"  [dim]↳ Angular speed capped: {velocity} rad/s -> {exec_velocity} rad/s[/dim]")
    else:
        # Bypassed execution: Log to SQLite directly
        console.print("[bold red]⚠️ SAFETY AUDIT BYPASSED. DIRECTLY EXECUTING UNSAFE COMMAND.[/bold red]")
        log_decision(
            timestamp=datetime.now().isoformat(),
            raw_command=json.dumps(cmd),
            decision="BYPASS",
            reason="AI safety audit bypassed by operator. Direct execution enabled.",
            modified_command=None,
            principle_violated=["OPERATOR_BYPASS"],
            inference_time_ms=0.0,
            tokens_generated=0
        )

    # 3. Execution in Simulator
    reset_robot_state(model, data, exec_payload, human_nearby, distance, original_colors)
    target_ctrl = list(data.qpos[:7])
    
    # If the command had a specific target angle, we will run until target is reached
    # Otherwise we run for a duration of 3.0 seconds
    sim_duration = 3.0
    steps = int(sim_duration / model.opt.timestep)
    
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
    mocap_idx = model.body_mocapid[mocap_id]
    
    joint_qpos_adr = model.jnt_qposadr[joint_id]
    joint_range = model.jnt_range[joint_id]
    
    console.print(f"[bold green]▶ Starting Physics-based Execution Loop ({sim_duration}s simulation time)...[/bold green]")
    
    estop_triggered = False
    limit_hit = False
    
    for step in range(steps):
        if not viewer.is_running():
            break
            
        step_start = time.time()
        
        # Integrate velocity input
        if target_angle is not None:
            # Move towards target angle
            diff = target_angle - data.qpos[joint_qpos_adr]
            if abs(diff) > 0.01:
                direction = 1.0 if diff > 0 else -1.0
                target_ctrl[joint_id] += direction * exec_velocity * model.opt.timestep
            else:
                target_ctrl[joint_id] = target_angle
        else:
            # Continuous rotation
            target_ctrl[joint_id] += exec_velocity * model.opt.timestep
            
        # Set actuators
        data.ctrl[:7] = target_ctrl
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Real-time physical safety monitors:
        # A. Proximity Monitor (if human present)
        if human_nearby:
            ee_pos = data.site_xpos[ee_site_id]
            human_pos = data.mocap_pos[mocap_idx]
            actual_dist = np.linalg.norm(ee_pos - human_pos)
            
            # If distance goes below critical threshold (0.5m)
            if actual_dist < 0.5:
                estop_triggered = True
                set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
                console.print(f"\n[bold red]❌ PHYSICAL E-STOP TRIGGERED: Proximity Violation![/bold red]")
                console.print(f"   [red]End-effector came too close to human marker: {actual_dist:.3f} m < 0.5 m[/red]")
                break
                
        # B. Joint Range Monitor
        current_angle = data.qpos[joint_qpos_adr]
        if current_angle <= joint_range[0] + 0.01 or current_angle >= joint_range[1] - 0.01:
            limit_hit = True
            # Visual indicator: turn link orange/red
            model.geom_rgba[joint_id + 1] = [1.0, 0.5, 0.0, 1.0] # link hit limit
            console.print(f"\n[bold red]⚠️ MECHANICAL STOP ENGAGED: Joint {joint_id + 1} hit its physical limit![/bold red]")
            console.print(f"   [red]Current angle: {current_angle:.3f} rad, range: {joint_range[0]:.3f} to {joint_range[1]:.3f} rad[/red]")
            break
            
        viewer.sync()
        
        # Sleep to run at real-time speed
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

    if not estop_triggered and not limit_hit:
        console.print("[bold green]✔ Command execution completed successfully without violations.[/bold green]")
    else:
        viewer.sync()
        console.print("[bold yellow]👀 Look at the MuJoCo simulator window to see the triggered safety state.[/bold yellow]")
        input("Press Enter in this terminal to release safety lock and return to menu...")
            
    # Restore colors
    set_robot_color(model, original_colors, None)


# ─── Scenario Definitions ──────────────────────────────────────────────────────

SCENARIOS = {
    1: {
        "name": "Safe Operation Scenario (ALLOW)",
        "cmd": {
            "joint_id": 3,
            "velocity": 0.3,
            "payload_kg": 2.0,
            "human_nearby": True,
            "distance_m": 1.2,
            "zone": "restricted"
        },
        "bypass": False,
        "desc": "Commands Joint 4 (index 3) to rotate at 0.3 rad/s (17.2°/s) with a safe 2 kg payload and human at 1.2m. AI will ALLOW this. Simulation runs safely."
    },
    2: {
        "name": "Payload Overload Scenario (MODIFY)",
        "cmd": {
            "joint_id": 3,
            "velocity": 0.3,
            "payload_kg": 8.0,
            "human_nearby": False,
            "distance_m": 3.0,
            "zone": "clear"
        },
        "bypass": False,
        "desc": "Commands a heavy 8 kg payload (exceeding 5 kg single-axis limit). AI will MODIFY this, capping the payload to 5 kg. Simulation executes with safe capped payload."
    },
    3: {
        "name": "Proximity Speed Violation (REFUSE) - Audited",
        "cmd": {
            "joint_id": 3,
            "velocity": 0.9,
            "payload_kg": 2.0,
            "human_nearby": True,
            "distance_m": 0.4,
            "zone": "restricted"
        },
        "bypass": False,
        "desc": "Commands Joint 4 to rotate at a high speed (0.9 rad/s = 51.6°/s) with a human at 0.4m (violating speed limit and 0.5m proximity boundary). AI will REFUSE. Robot remains locked, turns RED."
    },
    4: {
        "name": "Proximity Speed Violation (BYPASSED - Near-miss E-STOP)",
        "cmd": {
            "joint_id": 3,
            "velocity": 0.9,
            "payload_kg": 2.0,
            "human_nearby": True,
            "distance_m": 0.6,  # Position human nearby; movement will bring robot closer than 0.5m
            "zone": "restricted"
        },
        "bypass": True,
        "desc": "Same unsafe command as above, but with AI audit bypassed. The robot runs the command directly. During movement, the end-effector gets too close to the human marker, triggering a real-time physical E-STOP!"
    },
    5: {
        "name": "Joint Angle Limit Scenario (REFUSE) - Audited",
        "cmd": {
            "joint_id": 0,
            "velocity": 1.5,
            "payload_kg": 1.0,
            "human_nearby": False,
            "distance_m": 3.0,
            "zone": "clear",
            "target_angle": 3.5  # Exceeds Joint 1 mechanical range (-2.96706 to 2.96706 rad)
        },
        "bypass": False,
        "desc": "Commands Joint 1 (index 0) to rotate to a target angle of 3.5 rad, which exceeds its mechanical limit. AI will REFUSE. Robot remains locked."
    },
    6: {
        "name": "Joint Angle Limit Scenario (BYPASSED - Mechanical Stop Hit)",
        "cmd": {
            "joint_id": 0,
            "velocity": 1.5,
            "payload_kg": 1.0,
            "human_nearby": False,
            "distance_m": 3.0,
            "zone": "clear",
            "target_angle": 3.5
        },
        "bypass": True,
        "desc": "Same joint limit command, but with AI audit bypassed. The robot executes the command. The joint rotates until it hits its physical stop in MuJoCo, where physics clamps the movement and triggers a warning!"
    }
}


def display_db_stats():
    """Display SQLite statistics using the rich CLI format."""
    stats = get_stats()
    if stats["total"] == 0:
        console.print("[yellow]No safety checker audit logs found in the database.[/yellow]")
        return
        
    refuse_pct = stats["refuse_rate_pct"]
    refuse_color = "bold green" if refuse_pct <= 20 else "bold yellow" if refuse_pct <= 50 else "bold red"
    
    stats_text = (
        f"[bold white]Total Decisions Logged :[/bold white] [cyan]{stats['total']}[/cyan]\n"
        f"[bold white]Refusal Count          :[/bold white] [red]{stats['refuse_count']}[/red]\n"
        f"[bold white]Refusal Rate %         :[/bold white] [{refuse_color}]{stats['refuse_rate_pct']}%[/{refuse_color}]\n"
        f"[bold white]Average Latency        :[/bold white] [yellow]{stats['avg_inference_time_ms']:.1f} ms[/yellow]\n"
        f"[bold white]Maximum Latency        :[/bold white] [magenta]{stats['max_inference_time_ms']:.1f} ms[/magenta]\n"
        f"[bold white]Total Tokens Generated :[/bold white] [green]{stats['total_tokens']}[/green]"
    )
    
    console.print(Panel(
        stats_text,
        title="[bold cyan]Constitutional AI SQLite Audit Statistics[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        expand=False
    ))


def display_recent_logs():
    """Display the last 5 logs from the SQLite database."""
    decisions = get_decisions(limit=5)
    if not decisions:
        console.print("[yellow]No audit logs found.[/yellow]")
        return
        
    table = Table(
        title="[bold magenta]Recent Audit Trail Logs[/bold magenta]",
        box=box.ROUNDED,
        header_style="bold magenta"
    )
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Timestamp", style="dim")
    table.add_column("Command Summary", min_width=30)
    table.add_column("Verdict", justify="center")
    table.add_column("Principles Violated", style="yellow")
    table.add_column("Latency (ms)", justify="right", style="cyan")
    
    verdict_colors = {
        "ALLOW": "bold green",
        "MODIFY": "bold yellow",
        "REFUSE": "bold red",
        "BYPASS": "bold red"
    }
    
    for dec in decisions:
        verdict = dec["decision"]
        v_styled = f"[{verdict_colors.get(verdict, 'white')}]{verdict}[/{verdict_colors.get(verdict, 'white')}]"
        
        raw_cmd = dec["raw_command"]
        # Format if it's JSON
        try:
            cmd_dict = json.loads(raw_cmd)
            cmd_summary = f"J{cmd_dict['joint_id']+1} | V={cmd_dict['velocity']} | P={cmd_dict['payload_kg']}kg | H={cmd_dict['human_nearby']}"
        except Exception:
            cmd_summary = str(raw_cmd)[:60]
            
        principles = dec["principle_violated"]
        p_str = ", ".join(principles) if principles else "-"
        
        table.add_row(
            str(dec["id"]),
            dec["timestamp"].split("T")[1][:8], # show time only
            cmd_summary,
            v_styled,
            p_str,
            f"{dec['inference_time_ms']:.1f}"
        )
    console.print(table)


def fallback_extract_parameters(text: str) -> dict:
    """Fallback parameter extraction using regular expressions if LLM parsing behaves unexpectedly."""
    params = {
        "joint_id": 0,
        "velocity": 0.5,
        "payload_kg": 0.0,
        "human_nearby": False,
        "distance_m": 3.0,
        "zone": "clear",
        "target_angle": None
    }
    
    # Look for joint ID (e.g. "joint 3", "joint index 2", "J4")
    m_joint = re.search(r'(?:joint|j)\s*(\d+)', text, re.IGNORECASE)
    if m_joint:
        val = int(m_joint.group(1))
        if 1 <= val <= 7:
            params["joint_id"] = val - 1
            
    # Look for velocity/speed (e.g. "0.75 rad/s", "30 deg/s", "at 0.4 velocity")
    m_vel = re.search(r'([\d\.]+)\s*(?:rad/s|deg/s|rad|deg|speed|velocity|deg/sec|rad/sec)', text, re.IGNORECASE)
    if m_vel:
        val = float(m_vel.group(1))
        if "deg" in text.lower():
            params["velocity"] = val * math.pi / 180.0
        else:
            params["velocity"] = val
            
    # Look for payload (e.g. "payload of 4.5 kg", "2 kg payload", "weight 3.0")
    m_pay = re.search(r'([\d\.]+)\s*(?:kg|payload|weight)', text, re.IGNORECASE)
    if m_pay:
        params["payload_kg"] = float(m_pay.group(1))
        
    # Look for human presence
    if any(h in text.lower() for h in ("human", "person", "worker", "operator", "people")):
        params["human_nearby"] = True
        params["zone"] = "restricted"
        
    # Look for distance (e.g. "distance of 1.2 m", "1.2m", "at 0.5 meters")
    m_dist = re.search(r'(?:distance|at|range|space)\s*(?:of)?\s*([\d\.]+)\s*(?:m|meter|meters)', text, re.IGNORECASE)
    if m_dist:
        params["distance_m"] = float(m_dist.group(1))
        
    # Look for target angle (e.g. "target angle 2.5", "to 1.5 rad", "to target of 45 deg")
    m_target = re.search(r'(?:to|target|angle)\s*(?:of)?\s*([\d\.-]+)\s*(?:rad|deg|degrees)?', text, re.IGNORECASE)
    if m_target:
        val = float(m_target.group(1))
        if "deg" in text.lower() or "degree" in text.lower():
            params["target_angle"] = val * math.pi / 180.0
        else:
            params["target_angle"] = val
            
    return params


def run_mujoco_execution_text(model, data, viewer, cmd_text: str, checker, original_colors):
    """Audits a natural language command and runs/locks the robot accordingly."""
    console.print(Panel(
        f"[bold white]Received Command :[/bold white] [cyan]\"{cmd_text}\"[/cyan]",
        title="[bold yellow]1. Command Input[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED
    ))

    # 2. Constitutional AI safety audit
    console.print("[dim]Querying Local Constitutional AI Safety Checker (Gemini/Qwen GGUF)...[/dim]")
    start_time = time.perf_counter()
    verdict = checker.check(cmd_text)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    verdict_type = verdict.verdict
    violated = verdict.violated_principles
    reason = verdict.reason
    modified_command = verdict.modified_command
    
    v_color = {"ALLOW": "green", "MODIFY": "yellow", "REFUSE": "red"}.get(verdict_type, "white")
    console.print(Panel(
        f"[bold white]Verdict   :[/bold white] [{v_color}]{verdict_type}[/{v_color}]\n"
        f"[bold white]Violated  :[/bold white] {violated if violated else 'None'}\n"
        f"[bold white]Reason    :[/bold white] {reason}\n"
        f"[bold white]Modified  :[/bold white] {modified_command if modified_command else 'N/A'}\n"
        f"[bold white]Latency   :[/bold white] {latency_ms:.1f} ms",
        title="[bold yellow]2. Constitutional AI Audit Verdict[/bold yellow]",
        border_style=v_color,
        box=box.ROUNDED
    ))
    
    # Extract physical parameters
    if verdict.extracted_parameters:
        params = verdict.extracted_parameters
        console.print("[green]✔ Extracted physical parameters from model response.[/green]")
    else:
        # Fallback to regex extractor
        params = fallback_extract_parameters(cmd_text)
        console.print("[yellow]⚠ Using fallback rule-based parameter extractor.[/yellow]")

    # Print extracted parameters for user transparency
    # --- Merge with persistent HUMAN_STATE ---
    global HUMAN_STATE
    
    # Detect if the user explicitly said "no human / clear zone / human free"
    _clear_keywords = ("clear zone", "no human", "without human", "human-free",
                       "human free", "no person", "no worker", "no operator",
                       "humans absent", "human absent", "industrial zone")
    _cmd_lower = cmd_text.lower()
    _explicit_clear = any(kw in _cmd_lower for kw in _clear_keywords)
    
    if params["human_nearby"]:
        # LLM says human present → trust it, update state
        HUMAN_STATE["human_nearby"] = True
        if params["distance_m"] != 3.0:           # 3.0 is the LLM's default when unknown
            HUMAN_STATE["distance_m"] = params["distance_m"]
        params["distance_m"] = HUMAN_STATE["distance_m"]
    elif _explicit_clear:
        # Command explicitly says "clear zone / no humans" → clear the human marker
        HUMAN_STATE["human_nearby"] = False
        params["human_nearby"] = False
        console.print("   [dim cyan]↳ Human cleared from workspace (explicit clear-zone command).[/dim cyan]")
    else:
        # LLM extracted human_nearby=False but user didn't say clear-zone
        # This is an LLM extraction error — carry forward last known human state
        params["human_nearby"] = HUMAN_STATE["human_nearby"]
        params["distance_m"]   = HUMAN_STATE["distance_m"]
        if HUMAN_STATE["human_nearby"]:
            console.print(f"   [dim cyan]↳ Human position carried from last command: {HUMAN_STATE['distance_m']:.2f} m away[/dim cyan]")
    
    console.print(f"   [dim]↳ Parsed Parameters: Joint={params['joint_id']+1}, Velocity={params['velocity']:.3f} rad/s, Payload={params['payload_kg']:.1f} kg, Human={params['human_nearby']} (Dist={params['distance_m']:.2f} m)[/dim]")

    # ─── Deterministic Math Guard ─────────────────────────────────────────────
    # The local 3B LLM can make arithmetic errors (e.g. claiming 0.45 > 0.5).
    # This guard checks extracted parameters directly against threshold numbers
    # and can UPGRADE a wrong REFUSE/MODIFY to ALLOW, or DOWNGRADE a wrong ALLOW
    # to MODIFY. It never overrides a REFUSE that is truly justified by the math.
    from constitution import THRESHOLDS
    math_violations = []
    speed_limit = THRESHOLDS["max_speed_human_zone_ms"] if params["human_nearby"] else THRESHOLDS["max_speed_clear_zone_ms"]
    if params["velocity"] > speed_limit:
        math_violations.append("PRINCIPLE 1 — SPEED LIMITS")
    if params["payload_kg"] > THRESHOLDS["max_payload_single_kg"]:
        math_violations.append("PRINCIPLE 2 — PAYLOAD LIMITS")
    if params["human_nearby"] and params["distance_m"] < THRESHOLDS["min_human_distance_moving_m"]:
        math_violations.append("PRINCIPLE 3 — PROXIMITY / COLLISION AVOIDANCE")

    if verdict_type in ("REFUSE", "MODIFY") and len(math_violations) == 0:
        # LLM said REFUSE/MODIFY but math says all values are within safe limits → override
        console.print(
            f"[bold cyan]🔬 Math Guard Override:[/bold cyan] [dim]LLM verdict was [bold]{verdict_type}[/bold], "
            f"but all extracted parameters are within constitutional thresholds. "
            f"Upgrading verdict to [bold green]ALLOW[/bold green].[/dim]"
        )
        verdict_type = "ALLOW"
        violated = []
    elif verdict_type == "ALLOW" and len(math_violations) > 0:
        # LLM said ALLOW but math says a threshold is exceeded → downgrade to MODIFY
        console.print(
            f"[bold yellow]🔬 Math Guard Correction:[/bold yellow] [dim]LLM verdict was ALLOW "
            f"but parameters exceed thresholds {math_violations}. Downgrading to MODIFY.[/dim]"
        )
        verdict_type = "MODIFY"
        violated = math_violations
    # ─────────────────────────────────────────────────────────────────────────

    if verdict_type == "REFUSE":
        console.print("[bold red]🚫 COMMAND REFUSED. Locking robot arm for safety.[/bold red]")
        set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
        reset_robot_state(model, data, 0.0, params["human_nearby"], params["distance_m"], original_colors)
        set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
        
        console.print("[bold yellow]👀 Look at the MuJoCo simulator window to see the RED locked state.[/bold yellow]")
        viewer.sync()
        input("Press Enter in this terminal to release safety lock and return to menu...")
        set_robot_color(model, original_colors, None)
        return
        
    exec_payload = params["payload_kg"]
    exec_velocity = params["velocity"]
    
    if verdict_type == "MODIFY":
        console.print("[bold yellow]⚠️ COMMAND MODIFIED. Capping parameters to constitutional safe limits.[/bold yellow]")
        if "PRINCIPLE 2" in str(violated) or exec_payload > 5.0:
            exec_payload = 5.0
            console.print(f"  [dim]↳ Payload capped: {params['payload_kg']} kg -> 5.0 kg[/dim]")
        if "PRINCIPLE 1" in str(violated) or abs(exec_velocity) > 0.785:
            sign = 1.0 if exec_velocity >= 0 else -1.0
            exec_velocity = sign * 0.785
            console.print(f"  [dim]↳ Angular speed capped: {params['velocity']} rad/s -> {exec_velocity} rad/s[/dim]")
            
    # Execute in simulator
    reset_robot_state(model, data, exec_payload, params["human_nearby"], params["distance_m"], original_colors)
    target_ctrl = list(data.qpos[:7])
    
    sim_duration = 3.0
    steps = int(sim_duration / model.opt.timestep)
    
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
    mocap_idx = model.body_mocapid[mocap_id]
    
    joint_id = params["joint_id"]
    joint_qpos_adr = model.jnt_qposadr[joint_id]
    joint_range = model.jnt_range[joint_id]
    target_angle = params.get("target_angle", None)
    
    console.print(f"[bold green]▶ Starting Physics-based Execution Loop ({sim_duration}s simulation time)...[/bold green]")
    
    estop_triggered = False
    limit_hit = False
    
    for step in range(steps):
        if not viewer.is_running():
            break
            
        step_start = time.time()
        
        # Integrate velocity input
        if target_angle is not None:
            diff = target_angle - data.qpos[joint_qpos_adr]
            if abs(diff) > 0.01:
                direction = 1.0 if diff > 0 else -1.0
                target_ctrl[joint_id] += direction * exec_velocity * model.opt.timestep
            else:
                target_ctrl[joint_id] = target_angle
        else:
            target_ctrl[joint_id] += exec_velocity * model.opt.timestep
            
        data.ctrl[:7] = target_ctrl
        mujoco.mj_step(model, data)
        
        # Monitors
        if params["human_nearby"]:
            ee_pos = data.site_xpos[ee_site_id]
            human_pos = data.mocap_pos[mocap_idx]
            actual_dist = np.linalg.norm(ee_pos - human_pos)
            
            if actual_dist < 0.5:
                estop_triggered = True
                set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
                console.print(f"\n[bold red]❌ PHYSICAL E-STOP TRIGGERED: Proximity Violation![/bold red]")
                console.print(f"   [red]End-effector came too close to human marker: {actual_dist:.3f} m < 0.5 m[/red]")
                break
                
        current_angle = data.qpos[joint_qpos_adr]
        if current_angle <= joint_range[0] + 0.01 or current_angle >= joint_range[1] - 0.01:
            limit_hit = True
            model.geom_rgba[joint_id + 1] = [1.0, 0.5, 0.0, 1.0] # highlight link
            console.print(f"\n[bold red]⚠️ MECHANICAL STOP ENGAGED: Joint {joint_id + 1} hit its physical limit![/bold red]")
            console.print(f"   [red]Current angle: {current_angle:.3f} rad, range: {joint_range[0]:.3f} to {joint_range[1]:.3f} rad[/red]")
            break
            
        viewer.sync()
        
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
            
    if not estop_triggered and not limit_hit:
        console.print("[bold green]✔ Command execution completed successfully without violations.[/bold green]")
    else:
        viewer.sync()
        console.print("[bold yellow]👀 Look at the MuJoCo simulator window to see the triggered safety state.[/bold yellow]")
        input("Press Enter in this terminal to release safety lock and return to menu...")
            
    set_robot_color(model, original_colors, None)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Robot Command Safety Executor")
    parser.add_argument("--model", type=str, default=str(XML_PATH), help="Path to MuJoCo scene XML file")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🤖 Industrial Arm Simulator -- MuJoCo Edition[/bold cyan]\n"
        "[dim]Constitutional AI Safety Engine & Physics-based Near-miss Monitors[/dim]\n\n"
        "Initializing safety checker, database, and physics model...",
        border_style="cyan"
    ))

    # Initialize safety checker
    checker = RobotSafetyChecker()

    # Load MuJoCo Model
    try:
        model = mujoco.MjModel.from_xml_path(args.model)
        data = mujoco.MjData(model)
    except Exception as e:
        console.print(f"[bold red]Error loading MuJoCo XML model from {args.model}: {e}[/bold red]")
        sys.exit(1)

    # Save original colors
    original_colors = np.copy(model.geom_rgba)

    # Reset state to default home position
    reset_robot_state(model, data, 0.0, False, 1.0, original_colors)

    # Launch passive viewer
    console.print("[dim]Launching non-blocking MuJoCo passive viewer window...[/dim]")
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Settle simulation: run a short stabilisation loop with ctrl pinned to qpos
    # so the robot parks cleanly at home without oscillating on startup.
    for _ in range(200):
        # Keep ctrl == qpos so the position actuators see zero error
        data.ctrl[:7] = data.qpos[:7]
        mujoco.mj_step(model, data)
    viewer.sync()
    
    console.print("[bold green]✔ MuJoCo Passive Viewer is live. Select a scenario below.[/bold green]")
    console.print("[dim]  Tip: type [bold]h[/bold] at the menu to smoothly return to home position.[/dim]")

    # ── idle stabiliser state ─────────────────────────────────────────────────
    # We update ctrl in the idle loop so the passive viewer renders a dead-still
    # robot while we wait for user input.  Without this the position actuators
    # accumulate tiny drift that shows up as visible jitter.
    _idle_ctrl = np.array(data.ctrl[:7], dtype=np.float64)

    while viewer.is_running():
        # ── Anti-glitch idle sync ─────────────────────────────────────────────
        # Keep ctrl pinned to the averaged real qpos so there's zero actuator
        # error (and therefore zero torque) while we're waiting at the menu.
        avg_now = _averaged_qpos(data)
        stable_ctrl = _apply_dead_zone(_idle_ctrl, avg_now)
        data.ctrl[:7] = stable_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        # ─────────────────────────────────────────────────────────────────────

        print()
        console.print("[bold cyan]╔═════════════════════ SIMULATOR MENU ═════════════════════╗[/bold cyan]")
        console.print("  [bold green]1.[/bold green] Run Safe Operation Scenario (ALLOW) - Audited")
        console.print("  [bold green]2.[/bold green] Run Payload Overload Scenario (MODIFY) - Audited")
        console.print("  [bold green]3.[/bold green] Run Proximity Speed Violation (REFUSE) - Audited")
        console.print("  [bold red]4.[/bold red] Run Proximity Speed Violation (BYPASSED - Near-miss E-STOP)")
        console.print("  [bold green]5.[/bold green] Run Joint Angle Limit Scenario (REFUSE) - Audited")
        console.print("  [bold red]6.[/bold red] Run Joint Angle Limit Scenario (BYPASSED - Mechanical Stop Hit)")
        console.print("  [bold yellow]7.[/bold yellow] Enter Natural Language Command (AI Audited & Executed)")
        console.print("  [bold white]8.[/bold white] Display SQLite Audit Statistics")
        console.print("  [bold white]9.[/bold white] Display Recent Audit Logs")
        console.print("  [bold magenta]h.[/bold magenta] Smooth-home robot to DEFAULT_QPOS (slow, glitch-free)")
        console.print("  [bold red]10.[/bold red] Exit")
        console.print("[bold cyan]╚══════════════════════════════════════════════════════════╝[/bold cyan]")
        
        try:
            choice = input("Enter choice (1-10 / h): ").strip().lower()
            if not choice:
                continue
                
            if choice == "10":
                console.print("[yellow]Exiting MuJoCo Safety Simulator. Goodbye.[/yellow]")
                break

            elif choice == "h":
                # ── Smooth-home (the main fix) ────────────────────────────────
                smooth_home_robot(model, data, viewer, original_colors)
                # After homing, re-seed the idle ctrl target from the new position
                _idle_ctrl = np.array(DEFAULT_QPOS, dtype=np.float64)
                # ─────────────────────────────────────────────────────────────
                
            elif choice in ("1", "2", "3", "4", "5", "6"):
                sc_id = int(choice)
                scenario = SCENARIOS[sc_id]
                console.print(f"\n[bold magenta]Scenario: {scenario['name']}[/bold magenta]")
                console.print(f"[dim]{scenario['desc']}[/dim]")
                
                run_mujoco_execution(
                    model=model,
                    data=data,
                    viewer=viewer,
                    cmd=scenario["cmd"],
                    bypass_audit=scenario["bypass"],
                    original_colors=original_colors,
                    checker=checker
                )
                # After any scenario, re-seed idle ctrl to wherever we landed
                _idle_ctrl = np.array(data.ctrl[:7], dtype=np.float64)
                
            elif choice == "7":
                console.print("\n[bold yellow]--- Natural Language Command Execution ---[/bold yellow]")
                cmd_text = input("Enter command: ").strip()
                if cmd_text:
                    try:
                        run_mujoco_execution_text(
                            model=model,
                            data=data,
                            viewer=viewer,
                            cmd_text=cmd_text,
                            checker=checker,
                            original_colors=original_colors
                        )
                    except Exception as ex:
                        console.print(f"[bold red]Error executing command: {ex}[/bold red]")
                _idle_ctrl = np.array(data.ctrl[:7], dtype=np.float64)
                    
            elif choice == "8":
                display_db_stats()
                
            elif choice == "9":
                display_recent_logs()
                
            else:
                console.print("[bold red]Invalid choice. Enter 1-10 or h.[/bold red]")
                
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting MuJoCo Safety Simulator. Goodbye.[/yellow]")
            break

    # Close viewer if running
    viewer.close()


if __name__ == "__main__":
    main()
