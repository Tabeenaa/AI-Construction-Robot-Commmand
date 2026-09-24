import numpy as np
import mujoco
from pathlib import Path
from ik_solver import CartesianIKSolver, generate_cartesian_trajectory

XML_PATH = Path("models/kuka_iiwa_14/scene.xml")

def test_ik():
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Set home position
    data.qpos[:7] = [0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    
    solver = CartesianIKSolver(model, site_name="attachment_site")
    start_pos = solver.get_end_effector_pos(data)
    print(f"Initial End-Effector Position: {start_pos}")
    
    test_targets = [
        start_pos + np.array([0.1, 0.1, -0.1]),
        start_pos + np.array([-0.15, 0.05, 0.1]),
        np.array([0.45, 0.20, 0.35]),
        np.array([0.35, -0.25, 0.40]),
    ]
    
    for i, target in enumerate(test_targets, 1):
        print(f"\n--- Testing Target {i}: {target} ---")
        success, q_sol, err = solver.solve_ik(data, target, max_iters=120)
        print(f"Result: Success={success}, Error={err*1000:.2f} mm")
        assert err < 0.01, f"Error too large: {err*1000:.2f} mm"
        
        # Test trajectory generation
        waypoints, duration = generate_cartesian_trajectory(start_pos, target, speed_mps=0.3)
        print(f"Trajectory: {len(waypoints)} waypoints over {duration:.2f} seconds")
        
    print("\n>>> ALL IK TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_ik()
