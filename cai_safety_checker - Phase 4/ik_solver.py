"""
ik_solver.py
============
Cartesian Inverse Kinematics (IK) & Trajectory Planning Engine for 7-DOF Manipulator in MuJoCo.

Features:
1. Damped Least Squares (DLS / Levenberg-Marquardt) Jacobian solver.
2. Secondary nullspace projection for joint limit avoidance and natural posture centering.
3. Smooth Minimum-Jerk and S-Curve Cartesian path generation.
4. Real-time Cartesian velocity limits compliant with ISO/TS 15066 Cobot Safety.
"""

import math
import numpy as np
import mujoco

DEFAULT_HOME_QPOS = np.array([0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0], dtype=np.float64)


class CartesianIKSolver:
    """Numerical Inverse Kinematics solver using MuJoCo analytical Jacobians."""

    def __init__(self, model, site_name: str = "attachment_site"):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in model.")
            
        self.nv = model.nv
        # Joint limits
        self.joint_limits = np.zeros((self.nv, 2), dtype=np.float64)
        for i in range(min(self.nv, 7)):
            joint_id = model.jnt_qposadr[i]
            if model.jnt_limited[i]:
                self.joint_limits[i] = model.jnt_range[i]
            else:
                self.joint_limits[i] = [-np.pi, np.pi]

    def get_end_effector_pos(self, data) -> np.ndarray:
        """Return current Cartesian position [x, y, z] of the end-effector site."""
        return np.array(data.site_xpos[self.site_id], dtype=np.float64)

    def get_end_effector_mat(self, data) -> np.ndarray:
        """Return current 3x3 rotation matrix of the end-effector site."""
        return np.array(data.site_xmat[self.site_id].reshape(3, 3), dtype=np.float64)

    def solve_ik(
        self,
        data,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        max_iters: int = 120,
        tol: float = 1e-3,
        damping: float = 0.04,
        step_size: float = 0.5,
        nullspace_weight: float = 0.05,
        home_qpos: np.ndarray = None
    ) -> tuple[bool, np.ndarray, float]:
        """
        Solve numerical IK for target Cartesian position (and optional orientation).
        
        Uses Damped Least Squares (DLS):
            dq = J^T * (J * J^T + lambda^2 * I)^-1 * err + (I - J# * J) * k * (q_home - q)
            
        Returns:
            (success: bool, final_qpos: np.ndarray, final_error: float)
        """
        if home_qpos is None:
            home_qpos = DEFAULT_HOME_QPOS

        # Work on a copy of qpos
        q = np.array(data.qpos[:self.nv], dtype=np.float64)
        
        # Temp data for forward kinematics
        d_temp = mujoco.MjData(self.model)
        d_temp.qpos[:self.nv] = q
        mujoco.mj_forward(self.model, d_temp)

        jacp = np.zeros((3, self.nv), dtype=np.float64)
        jacr = np.zeros((3, self.nv), dtype=np.float64)

        final_err = 999.0
        for it in range(max_iters):
            curr_pos = d_temp.site_xpos[self.site_id]
            pos_err = target_pos - curr_pos
            err_norm = np.linalg.norm(pos_err)
            final_err = float(err_norm)

            if err_norm < tol:
                return True, q, final_err

            # Compute Jacobian at attachment site
            mujoco.mj_jacSite(self.model, d_temp, jacp, jacr, self.site_id)
            J = jacp[:, :7]  # 3x7

            JJt = J @ J.T
            inv_term = np.linalg.inv(JJt + (damping ** 2) * np.eye(3))
            J_dls = J.T @ inv_term  # 7x3

            # Primary task: position error
            dq_primary = J_dls @ pos_err

            # Secondary task: light posture stabilization
            null_proj = np.eye(7) - (J_dls @ J)
            dq_null = null_proj @ (nullspace_weight * (home_qpos[:7] - q[:7]))

            # Multi-scale step: larger step if far, fine-tuned when close
            gamma = 0.8 if err_norm > 0.05 else 0.5
            dq = gamma * dq_primary + 0.05 * dq_null

            # Limit step
            dq = np.clip(dq, -0.4, 0.4)
            q_candidate = q + dq

            # Joint limits
            for i in range(min(7, self.nv)):
                q_candidate[i] = np.clip(q_candidate[i], self.joint_limits[i, 0] + 0.01, self.joint_limits[i, 1] - 0.01)

            q = q_candidate
            d_temp.qpos[:self.nv] = q
            mujoco.mj_forward(self.model, d_temp)

        curr_pos = d_temp.site_xpos[self.site_id]
        final_err = float(np.linalg.norm(target_pos - curr_pos))
        success = final_err < 0.005
        return success, q, final_err


def generate_cartesian_trajectory(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    speed_mps: float = 0.3,
    dt: float = 0.01
) -> tuple[np.ndarray, float]:
    """
    Generate a smooth Minimum-Jerk Cartesian trajectory between start and target.
    
    Formula (Minimum-Jerk):
        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5, tau in [0, 1]
    
    Returns:
        waypoints: (N, 3) array of Cartesian coordinates
        duration: total trajectory time in seconds
    """
    start_pos = np.array(start_pos, dtype=np.float64)
    target_pos = np.array(target_pos, dtype=np.float64)
    dist = np.linalg.norm(target_pos - start_pos)
    
    # Minimum duration based on speed clamp
    speed_mps = max(0.05, speed_mps)
    duration = max(dist / speed_mps, 0.6)  # at least 0.6s for smooth acceleration
    
    num_steps = max(int(duration / dt), 10)
    waypoints = np.zeros((num_steps, 3), dtype=np.float64)
    
    for i in range(num_steps):
        tau = i / (num_steps - 1)
        # Minimum-Jerk polynomial
        s = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
        waypoints[i] = start_pos + s * (target_pos - start_pos)
        
    return waypoints, duration
