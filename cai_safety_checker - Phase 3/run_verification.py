import time
import json
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from robot_executor import command_to_text, reset_robot_state, set_robot_color
from safety_checker import RobotSafetyChecker
from db import log_decision, get_stats

XML_PATH = Path("models/kuka_iiwa_14/scene.xml")

def run_auto_verification():
    print("======================================================================")
    print("            AUTOMATED SIMULATOR VERIFICATION RUNNER")
    print("======================================================================")
    
    # 1. Initialize checker
    print("Initializing safety checker...")
    checker = RobotSafetyChecker()
    
    # 2. Load model
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    original_colors = np.copy(model.geom_rgba)
    
    # 3. Open passive viewer
    print("Opening passive viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Settle simulation
    for _ in range(100):
        mujoco.mj_step(model, data)
    viewer.sync()
    
    # Define test scenarios
    # Test case 1: Safe Audited ALLOW
    # Test case 2: Unaudited Proximity Violation (Near-miss E-STOP)
    # Test case 3: Unaudited Joint Range Violation (Mechanical Stop Hit)
    scenarios = [
        {
            "name": "TEST CASE 1: Safe Audited Operation (ALLOW)",
            "cmd": {
                "joint_id": 3,
                "velocity": 0.3,
                "payload_kg": 2.0,
                "human_nearby": True,
                "distance_m": 1.2,
                "zone": "restricted"
            },
            "bypass": False
        },
        {
            "name": "TEST CASE 2: Proximity Proximity Speed Violation (BYPASSED - Physical E-STOP)",
            "cmd": {
                "joint_id": 3,
                "velocity": 0.9,
                "payload_kg": 2.0,
                "human_nearby": True,
                "distance_m": 0.6,
                "zone": "restricted"
            },
            "bypass": True
        },
        {
            "name": "TEST CASE 3: Joint Angle Limit Violation (BYPASSED - Mechanical Stop Hit)",
            "cmd": {
                "joint_id": 0,
                "velocity": 1.5,
                "payload_kg": 1.0,
                "human_nearby": False,
                "distance_m": 3.0,
                "zone": "clear",
                "target_angle": 3.5
            },
            "bypass": True
        }
    ]
    
    for i, sc in enumerate(scenarios, 1):
        print(f"\n--- Running {sc['name']} ---")
        cmd = sc["cmd"]
        bypass = sc["bypass"]
        
        # Reset state
        reset_robot_state(model, data, cmd["payload_kg"], cmd["human_nearby"], cmd["distance_m"], original_colors)
        target_ctrl = list(data.qpos[:7])
        
        # Audit
        if not bypass:
            cmd_text = command_to_text(cmd)
            print(f"Auditing command: \"{cmd_text}\"")
            verdict = checker.check(cmd_text)
            print(f"Audit Verdict: {verdict.verdict}")
            print(f"Reason: {verdict.reason}")
            
            if verdict.verdict == "REFUSE":
                print("🚫 Locked by AI Audit.")
                set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
                time.sleep(2.0)
                set_robot_color(model, original_colors, None)
                continue
        else:
            print("⚠️ Bypassing audit, running direct execution...")
            log_decision(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                raw_command=json.dumps(cmd),
                decision="BYPASS",
                reason="Automatic verification run - bypass safety checker.",
                modified_command=None,
                principle_violated=["VERIFICATION_BYPASS"],
                inference_time_ms=0.0,
                tokens_generated=0
            )
            
        # Run simulation
        joint_id = cmd["joint_id"]
        exec_velocity = cmd["velocity"]
        target_angle = cmd.get("target_angle", None)
        
        sim_duration = 3.0
        steps = int(sim_duration / model.opt.timestep)
        
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_marker")
        mocap_idx = model.body_mocapid[mocap_id]
        
        joint_qpos_adr = model.jnt_qposadr[joint_id]
        joint_range = model.jnt_range[joint_id]
        
        estop_triggered = False
        limit_hit = False
        
        for step in range(steps):
            if not viewer.is_running():
                break
                
            step_start = time.time()
            
            # Control input
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
            
            # Physics step
            mujoco.mj_step(model, data)
            
            # Monitors
            if cmd["human_nearby"]:
                ee_pos = data.site_xpos[ee_site_id]
                human_pos = data.mocap_pos[mocap_idx]
                actual_dist = np.linalg.norm(ee_pos - human_pos)
                
                if actual_dist < 0.5:
                    estop_triggered = True
                    set_robot_color(model, original_colors, [1.0, 0.0, 0.0, 0.8])
                    print(f"❌ PHYSICAL E-STOP TRIGGERED! Proximity violation: {actual_dist:.3f} m < 0.5 m")
                    break
                    
            current_angle = data.qpos[joint_qpos_adr]
            if current_angle <= joint_range[0] + 0.01 or current_angle >= joint_range[1] - 0.01:
                limit_hit = True
                model.geom_rgba[joint_id + 1] = [1.0, 0.5, 0.0, 1.0] # highlight joint
                print(f"⚠️ MECHANICAL STOP ENGAGED! Joint {joint_id + 1} hit limit: {current_angle:.3f} rad")
                break
                
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
                
        if not estop_triggered and not limit_hit:
            print("✔ Execution finished safely.")
        else:
            time.sleep(1.5) # keep visual showing red/orange
            
        set_robot_color(model, original_colors, None)
        
    print("\nVerification runner completed all test cases.")
    viewer.close()

if __name__ == "__main__":
    run_auto_verification()
