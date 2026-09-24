"""
construction_executor.py
========================
MuJoCo Physics Execution Engine for AI Construction Manipulator.
Integrates:
  1. Cartesian Inverse Kinematics (IK) with Minimum-Jerk Trajectory Planning.
  2. ISO/TS 15066 Speed and Separation Monitoring (SSM) & Real-time Collision Guards.
  3. Constitutional AI Safety Checker (Local Qwen2.5-3B).
  4. Non-blocking MuJoCo passive viewer with visual feedback.
"""

import os
import sys
import time
import json
import math
import numpy as np
import argparse
from pathlib import Path

# Add directory to sys.path
sys.path.append(str(Path(__file__).parent))

import mujoco
import mujoco.viewer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ik_solver import CartesianIKSolver, generate_cartesian_trajectory, DEFAULT_HOME_QPOS
from safety_checker import RobotSafetyChecker, SafetyVerdict
from construction_tasks import CONSTRUCTION_SCENARIOS
from db import log_decision, get_stats, get_decisions

console = Console()

import argparse

# Default to the custom robot arm from SolidWorks if available, fallback to KUKA
CUSTOM_XML = Path(__file__).parent / "models" / "custom_arm" / "scene.xml"
KUKA_XML   = Path(__file__).parent / "models" / "kuka_iiwa_14" / "scene.xml"
XML_PATH   = CUSTOM_XML if CUSTOM_XML.exists() else KUKA_XML

SMOOTH_HOME_DUR = 1.5
DEAD_ZONE_RAD = 0.005


def set_robot_color(model, original_colors, color_rgba=None):
    """Change link colors for real-time visual safety state."""
    if color_rgba is None:
        model.geom_rgba[:] = original_colors
    else:
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "floor" not in geom_name and "human" not in geom_name:
                model.geom_rgba[geom_id] = color_rgba


def reset_robot_state(model, data, payload_kg: float, human_nearby: bool, distance_m: float, original_colors):
    """Reset the robot state, set payload mass, and position human marker."""
    set_robot_color(model, original_colors, None)
    
    # Update payload mass at end-effector (link7)
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    default_link7_mass = 1.2
    model.body_mass[link7_id] = default_link7_mass + payload_kg
    
    # Position human marker mocap body
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
    mocap_idx = model.body_mocapid[mocap_id]
    human_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "human_sphere")
    if human_nearby:
        # Place human marker at the correct distance in front of robot
        data.mocap_pos[mocap_idx] = [distance_m, 0.0, 0.5]
        if human_geom_id >= 0:
            model.geom_rgba[human_geom_id] = [1.0, 0.2, 0.2, 0.7]   # visible red sphere
    else:
        # Completely hide the marker — move far underground AND make transparent
        data.mocap_pos[mocap_idx] = [0.0, 0.0, -100.0]
        if human_geom_id >= 0:
            model.geom_rgba[human_geom_id] = [0.0, 0.0, 0.0, 0.0]   # fully invisible

    data.qpos[:7] = DEFAULT_HOME_QPOS
    data.qvel[:] = 0
    data.ctrl[:7] = DEFAULT_HOME_QPOS
    mujoco.mj_forward(model, data)


def smooth_home_robot(model, data, viewer, original_colors):
    """Smoothly glide arm back to DEFAULT_HOME_QPOS using kinematic-direct interpolation."""
    home = np.array(DEFAULT_HOME_QPOS, dtype=np.float64)
    q_start = np.array(data.qpos[:7], dtype=np.float64)
    num_steps = 90   # 3 seconds at 30 FPS
    step_delay = 3.0 / num_steps

    console.print(f"[bold cyan]🏠 Smooth homing over 3.0s ...[/bold cyan]")
    for i in range(num_steps):
        if not viewer.is_running():
            break
        tau = i / max(num_steps - 1, 1)
        s = 10*(tau**3) - 15*(tau**4) + 6*(tau**5)   # Minimum-Jerk
        q_frame = q_start + s * (home - q_start)
        data.qpos[:7] = q_frame
        data.qvel[:7] = 0.0
        data.ctrl[:7] = q_frame
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(step_delay)

    data.qpos[:7] = home
    data.qvel[:] = 0
    data.ctrl[:7] = home
    mujoco.mj_forward(model, data)
    set_robot_color(model, original_colors, None)
    viewer.sync()
    console.print("[bold green]✔ Robot safely parked at home position.[/bold green]")


def execute_cartesian_trajectory(
    model,
    data,
    viewer,
    ik_solver: CartesianIKSolver,
    target_pos: np.ndarray,
    speed_mps: float,
    original_colors,
    human_nearby: bool,
    task_name: str = "Construction Task"
) -> bool:
    """
    Execute smooth Cartesian trajectory using kinematic-direct joint replay.
    
    Strategy: Pre-solve ALL IK waypoints into a full joint-space trajectory,
    then write data.qpos directly each frame (bypassing PD controller lag).
    This guarantees the arm physically follows every waypoint.
    """
    start_pos = ik_solver.get_end_effector_pos(data)
    dist = float(np.linalg.norm(target_pos - start_pos))
    
    # Observable duration: slow enough to see clearly (3s minimum)
    speed_mps = max(0.02, speed_mps)
    duration = max(3.0, min(6.0, dist / speed_mps))
    
    num_steps = max(int(duration * 30), 90)  # 30 FPS
    
    # --- Phase 1: Pre-solve complete joint trajectory (IK for all waypoints) ---
    console.print(f"[bold cyan]▶ PLANNING:[/bold cyan] {task_name}")
    console.print(f"  Start  → [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}] m")
    console.print(f"  Target → [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] m  (dist={dist*100:.1f} cm)")
    console.print(f"  Motion → {duration:.1f}s @ {speed_mps:.2f} m/s  ({num_steps} frames)")

    joint_traj = []  # list of q arrays, one per frame
    q_prev = np.array(data.qpos[:7], dtype=np.float64)

    for i in range(num_steps):
        tau = i / max(num_steps - 1, 1)
        s = 10*(tau**3) - 15*(tau**4) + 6*(tau**5)   # Minimum-Jerk
        pt = start_pos + s * (target_pos - start_pos)

        # Warm-start IK from previous solution for smooth inter-frame continuity
        data.qpos[:7] = q_prev
        mujoco.mj_forward(model, data)
        success, q_sol, err = ik_solver.solve_ik(data, pt, max_iters=80)
        if success:
            q_prev = q_sol[:7].copy()
            joint_traj.append(q_sol[:7].copy())
        else:
            joint_traj.append(q_prev.copy())   # hold last good solution

    # Restore starting joint state before playback
    data.qpos[:7] = np.array(joint_traj[0], dtype=np.float64)
    data.qvel[:] = 0
    data.ctrl[:7] = data.qpos[:7]
    mujoco.mj_forward(model, data)
    viewer.sync()

    console.print(f"[green]  IK solved {len(joint_traj)} frames — starting playback...[/green]")

    ee_site_id = ik_solver.site_id
    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
    mocap_idx = model.body_mocapid[mocap_id]
    
    step_delay = duration / num_steps
    estop_triggered = False
    milestones_printed = set()

    # --- Phase 2: Kinematic-direct playback (write qpos every frame) ---
    for idx, q_frame in enumerate(joint_traj):
        if not viewer.is_running():
            break

        # Log progress at 25% milestones
        pct = int((idx / max(num_steps - 1, 1)) * 100)
        milestone = (pct // 25) * 25
        if milestone in (25, 50, 75) and milestone not in milestones_printed:
            milestones_printed.add(milestone)
            ee_now = data.site_xpos[ee_site_id]
            console.print(f"  [dim]{milestone}% → EE=[{ee_now[0]:.3f}, {ee_now[1]:.3f}, {ee_now[2]:.3f}][/dim]")

        # --- KINEMATIC-DIRECT: write joint angles straight to qpos ---
        # This bypasses PD controller lag entirely — arm moves to exact IK solution
        data.qpos[:7] = q_frame
        data.qvel[:7] = 0.0          # zero velocity (purely positional)
        data.ctrl[:7] = q_frame      # keep ctrl in sync so controller doesn't fight us
        mujoco.mj_forward(model, data)

        # Real-time Physical Proximity Sensor Guard
        if human_nearby:
            ee_pos = data.site_xpos[ee_site_id]
            human_pos = data.mocap_pos[mocap_idx]
            dist_to_human = np.linalg.norm(ee_pos - human_pos)
            if dist_to_human < 0.30:  # Absolute stop boundary
                estop_triggered = True
                set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.9])
                console.print(f"\n[bold red]❌ PHYSICAL E-STOP TRIGGERED: Worker proximity breach ({dist_to_human:.2f} m < 0.30 m)![/bold red]")
                break

        viewer.sync()
        time.sleep(step_delay)

    if not estop_triggered:
        final_pos = ik_solver.get_end_effector_pos(data)
        final_err = np.linalg.norm(target_pos - final_pos)
        console.print(f"[bold green]✔ {task_name} reached target: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}] (Error: {final_err*1000:.1f} mm)[/bold green]")
        return True
    else:
        input("Press Enter in terminal to clear E-STOP lock...")
        set_robot_color(model, original_colors, None)
        return False


def run_construction_scenario(model, data, viewer, ik_solver, scenario: dict, checker: RobotSafetyChecker, original_colors):
    """Audit and execute a construction task scenario."""
    cmd_text = scenario["cmd_text"]
    bypass = scenario.get("bypass", False)
    target_pos = np.array(scenario["target_pos"], dtype=np.float64)
    speed = scenario["speed_mps"]
    payload = scenario["payload_kg"]
    human = scenario["human_nearby"]
    dist = scenario["distance_m"]

    # NOTE: Use scenario's actual target positions — they are within the KUKA/custom arm workspace.
    # No override needed; each scenario has distinct, realistic Cartesian coordinates.

    console.print(Panel(
        f"[bold white]Task Name      :[/bold white] {scenario['name']}\n"
        f"[bold white]Command Text   :[/bold white] [cyan]\"{cmd_text}\"[/cyan]\n"
        f"[bold white]Target Pose    :[/bold white] X={target_pos[0]:.2f}, Y={target_pos[1]:.2f}, Z={target_pos[2]:.2f}\n"
        f"[bold white]Command Speed  :[/bold white] {speed:.2f} m/s\n"
        f"[bold white]Payload Mass   :[/bold white] {payload:.1f} kg\n"
        f"[bold white]Worker State   :[/bold white] Nearby={human} (Distance: {dist:.2f} m)",
        title="[bold yellow]1. Construction Task Dispatch[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED
    ))

    # Reset physical simulation state
    reset_robot_state(model, data, payload, human, dist, original_colors)

    if bypass:
        console.print("[bold red]⚠ SAFETY AUDIT BYPASSED BY OPERATOR — Direct Physical Execution[/bold red]")
        execute_cartesian_trajectory(model, data, viewer, ik_solver, target_pos, speed, original_colors, human, scenario["name"])
        return

    # Constitutional AI Audit
    console.print("[dim]Querying Constitutional AI Compliance Engine (ISO 15066 & GB 50870)...[/dim]")
    verdict = checker.check(cmd_text)
    
    v_color = {"ALLOW": "green", "MODIFY": "yellow", "REFUSE": "red"}.get(verdict.verdict, "white")
    console.print(Panel(
        f"[bold white]Verdict   :[/bold white] [{v_color}]{verdict.verdict}[/{v_color}]\n"
        f"[bold white]Violated  :[/bold white] {verdict.violated_principles if verdict.violated_principles else 'None'}\n"
        f"[bold white]Reason    :[/bold white] {verdict.reason}\n"
        f"[bold white]Modified  :[/bold white] {verdict.modified_command if verdict.modified_command else 'N/A'}\n"
        f"[bold white]Latency   :[/bold white] {verdict.inference_time_ms:.1f} ms",
        title="[bold yellow]2. Constitutional AI Safety Verdict[/bold yellow]",
        border_style=v_color,
        box=box.ROUNDED
    ))

    if verdict.verdict == "REFUSE":
        console.print("[bold red]🚫 COMMAND REFUSED. Manipulator locked at current pose for site safety.[/bold red]")
        set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
        time.sleep(1.5)
        set_robot_color(model, original_colors, None)
        return

    # If modified, apply safe parameters
    exec_speed = speed
    exec_payload = payload
    if verdict.verdict == "MODIFY" and verdict.extracted_parameters:
        exec_speed = verdict.extracted_parameters.get("speed_mps", speed)
        exec_payload = verdict.extracted_parameters.get("payload_kg", payload)
        console.print(f"[bold yellow]⚡ Applying Safe Regulated Parameters: Speed={exec_speed:.2f} m/s, Payload={exec_payload:.1f} kg[/bold yellow]")
        set_robot_color(model, original_colors, [1.0, 0.9, 0.2, 0.8])  # Yellow for regulated SSM
        reset_robot_state(model, data, exec_payload, human, dist, original_colors)

    execute_cartesian_trajectory(model, data, viewer, ik_solver, target_pos, exec_speed, original_colors, human, scenario["name"])
    set_robot_color(model, original_colors, None)


def main():
    parser = argparse.ArgumentParser(description="AI Construction Manipulator Safety Executor")
    parser.add_argument("--model", type=str, default=str(XML_PATH), help="Path to MuJoCo scene XML")
    parser.add_argument("--llm", action="store_true", help="Enable deep offline LLM reasoning (default: instant fast rule mode)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🏗 AI Construction Autonomous Manipulator — MuJoCo Edition[/bold cyan]\n"
        "[dim]Constitutional AI Safety Engine (ISO/TS 15066 & GB 50870) + Cartesian IK Engine[/dim]\n\n"
        "Initializing safety checker, database, kinematics solver, and physics...",
        border_style="cyan"
    ))

    checker = RobotSafetyChecker(use_llm=args.llm)
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    original_colors = np.copy(model.geom_rgba)
    ik_solver = CartesianIKSolver(model, site_name="attachment_site")

    reset_robot_state(model, data, 0.0, False, 1.0, original_colors)
    
    # Print actual home EE position for confirmation
    home_ee = ik_solver.get_end_effector_pos(data)
    console.print(f"[bold green]✔ Arm homed. EE at [{home_ee[0]:.3f}, {home_ee[1]:.3f}, {home_ee[2]:.3f}] m[/bold green]")
    
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.sync()

    console.print("[bold green]✔ Simulation live. Select a construction scenario below.[/bold green]")

    while viewer.is_running():
        # Keep viewer alive while menu is shown — minimal physics hold
        data.ctrl[:7] = data.qpos[:7]
        mujoco.mj_forward(model, data)
        if viewer.is_running():
            viewer.sync()

        print()
        console.print("[bold cyan]╔═══════════════════ AI CONSTRUCTION SIMULATOR MENU ═══════════════════╗[/bold cyan]")
        console.print("  [bold green]1.[/bold green] Precision Anchor Hole Drilling (ALLOW - Audited)")
        console.print("  [bold yellow]2.[/bold yellow] Prefab Rebar Pick & Place Overload (MODIFY - Clamped Payload)")
        console.print("  [bold green]3.[/bold green] Cobot Material Handover (ISO 15066 SSM Speed Scaling)")
        console.print("  [bold red]4.[/bold red] High-Speed Drilling Near Worker (BYPASSED - Physical Proximity E-STOP)")
        console.print("  [bold red]5.[/bold red] Out-of-Reach Structural Weld (REFUSE - Workspace Limit)")
        console.print("  [bold yellow]6.[/bold yellow] Enter Custom Construction NL Command (Audited & Executed via IK)")
        console.print("  [bold white]7.[/bold white] Display SQLite Compliance Audit Statistics")
        console.print("  [bold white]8.[/bold white] Display Recent Construction Audit Logs")
        console.print("  [bold magenta]h.[/bold magenta] Smooth-home manipulator to Stow Position")
        console.print("  [bold red]9.[/bold red] Exit")
        console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]")

        try:
            choice = input("Select scenario (1-9 / h): ").strip().lower()
            if not choice:
                continue
            if choice == "9":
                break
            elif choice == "h":
                smooth_home_robot(model, data, viewer, original_colors)
            elif choice in ("1", "2", "3", "4", "5"):
                sc_id = int(choice)
                run_construction_scenario(model, data, viewer, ik_solver, CONSTRUCTION_SCENARIOS[sc_id], checker, original_colors)
            elif choice == "6":
                cmd = input("Enter construction command (e.g. 'move left 0.1m', 'drill hole', 'lift rebar', 'reach up'): ").strip()
                if cmd:
                    verdict = checker.check(cmd)
                    console.print(f"[bold]Verdict:[/bold] {verdict.verdict} | [bold]Reason:[/bold] {verdict.reason}")
                    if verdict.verdict != "REFUSE":
                        p = verdict.extracted_parameters
                        curr_ee = ik_solver.get_end_effector_pos(data)
                        
                        # Determine Cartesian target from custom command
                        if p.get("target_x") is not None and p.get("target_y") is not None and p.get("target_z") is not None:
                            t_pos = np.array([p["target_x"], p["target_y"], p["target_z"]], dtype=np.float64)
                        else:
                            # Scale relative deltas to 15 cm for clearly observable motion
                            dx = p.get("delta_x", 0.0) * 2.5
                            dy = p.get("delta_y", 0.0) * 2.5
                            dz = p.get("delta_z", 0.0) * 2.5
                            if dx == 0 and dy == 0 and dz == 0:
                                dx = 0.15  # Default: extend forward 15 cm
                            t_pos = curr_ee + np.array([dx, dy, dz], dtype=np.float64)
                        
                        exec_spd = p.get("speed_mps", 0.3)
                        exec_pay = p.get("payload_kg", 1.0)
                        human = p.get("human_nearby", False)
                        dist = p.get("distance_m", 2.0)
                        
                        reset_robot_state(model, data, exec_pay, human, dist, original_colors)
                        execute_cartesian_trajectory(model, data, viewer, ik_solver, t_pos, exec_spd, original_colors, human, f"Custom: {cmd[:25]}")
            elif choice == "7":
                stats = get_stats()
                console.print(Panel(json.dumps(stats, indent=2), title="Audit Statistics", border_style="cyan"))
            elif choice == "8":
                logs = get_decisions(limit=5)
                for l in logs:
                    console.print(f"[{l['decision']}] {l['raw_command']} | Violated: {l['principle_violated']} ({l['inference_time_ms']:.1f}ms)")
        except (KeyboardInterrupt, EOFError):
            break

    viewer.close()


if __name__ == "__main__":
    main()
