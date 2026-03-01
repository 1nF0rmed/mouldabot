"""IK solver, motion execution, and pick-and-place primitives for the SO-101 arm."""

import numpy as np
import mujoco

# 5 reaching joints (gripper excluded)
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_ACTUATOR = "gripper"
EE_SITE = "gripperframe"

# Pick/place height offsets (metres)
APPROACH_HEIGHT = 0.08  # hover above target
GRASP_HEIGHT = 0.005    # lower to grasp (slight offset above block centre)

# Simulation stepping
SIM_STEPS_PER_TICK = 5   # mj_step calls per control tick
SETTLE_STEPS = 300       # steps for gripper open/close to settle


def _get_arm_joint_ids(model):
    """Return (joint_ids, dof_indices, actuator_ids) for the 5 arm joints."""
    jnt_ids = []
    dof_ids = []
    act_ids = []
    for name in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jnt_ids.append(jid)
        dof_ids.append(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        act_ids.append(aid)
    return np.array(jnt_ids), np.array(dof_ids), np.array(act_ids)


def solve_ik(model, data, target_pos, max_iter=100, tol=1e-3, damping=0.05):
    """Damped least-squares IK targeting the gripperframe site.

    Operates on the 5 arm joints only.
    Returns (qpos_arm, success) where qpos_arm has shape (5,).
    """
    jnt_ids, dof_ids, act_ids = _get_arm_joint_ids(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    # Work on a copy so we don't mutate the live state
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos[:]
    d.qvel[:] = 0
    mujoco.mj_forward(model, d)

    jacp = np.zeros((3, model.nv))

    for _ in range(max_iter):
        ee_pos = d.site_xpos[site_id]
        err = target_pos - ee_pos
        if np.linalg.norm(err) < tol:
            qpos_arm = np.array([d.qpos[model.jnt_qposadr[j]] for j in jnt_ids])
            return qpos_arm, True

        # Compute Jacobian (position only)
        mujoco.mj_jacSite(model, d, jacp, None, site_id)
        J = jacp[:, dof_ids]  # (3, 5) — only arm DOFs

        # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} err
        JJT = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Apply delta and clamp to joint limits
        for i, jid in enumerate(jnt_ids):
            qadr = model.jnt_qposadr[jid]
            d.qpos[qadr] += dq[i]
            lo, hi = model.jnt_range[jid]
            d.qpos[qadr] = np.clip(d.qpos[qadr], lo, hi)

        mujoco.mj_forward(model, d)

    qpos_arm = np.array([d.qpos[model.jnt_qposadr[j]] for j in jnt_ids])
    return qpos_arm, False


def move_to(model, data, target_pos, viewer=None, max_steps=3000):
    """Solve IK for target_pos, then drive the arm there in simulation.

    Returns True if converged, False on timeout.
    """
    qpos_target, ok = solve_ik(model, data, target_pos)
    if not ok:
        print(f"[IK] Warning: did not converge for target {target_pos}")

    _, _, act_ids = _get_arm_joint_ids(model)
    jnt_ids, dof_ids, _ = _get_arm_joint_ids(model)

    # Set arm actuator targets
    for i, aid in enumerate(act_ids):
        data.ctrl[aid] = qpos_target[i]

    # Step simulation until joints converge
    for step in range(max_steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

        # Check convergence every 50 steps
        if step % 50 == 49:
            current = np.array([data.qpos[model.jnt_qposadr[j]] for j in jnt_ids])
            if np.allclose(current, qpos_target, atol=0.02):
                return True

    return False


def open_gripper(model, data, viewer=None):
    """Open the gripper to its max range."""
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    data.ctrl[grip_id] = 1.74  # near max open
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def close_gripper(model, data, viewer=None):
    """Close the gripper to its min range (clamp on block)."""
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
    data.ctrl[grip_id] = -0.17  # near min closed
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def go_home(model, data, viewer=None):
    """Return all joints (including gripper) to the home position."""
    from sim import HOME_JOINTS
    for name, val in HOME_JOINTS.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[aid] = val
    for _ in range(600):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def pick_up(model, data, block_name, viewer=None):
    """Pick up a named block (e.g. 'red_block')."""
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    block_pos = data.xpos[block_id].copy()
    print(f"[pick] Block '{block_name}' at {block_pos}")

    above = block_pos.copy()
    above[2] += APPROACH_HEIGHT
    grasp = block_pos.copy()
    grasp[2] += GRASP_HEIGHT

    print("[pick] Moving above block...")
    move_to(model, data, above, viewer)

    print("[pick] Opening gripper...")
    open_gripper(model, data, viewer)

    print("[pick] Lowering to grasp...")
    move_to(model, data, grasp, viewer)

    print("[pick] Closing gripper...")
    close_gripper(model, data, viewer)

    print("[pick] Lifting...")
    move_to(model, data, above, viewer)
    print("[pick] Done.")


def place(model, data, target_pos, viewer=None):
    """Place the held block at target_pos (x, y on table surface)."""
    # Table surface is at z≈0.21
    table_z = 0.22
    target = np.array([target_pos[0], target_pos[1], table_z])

    above = target.copy()
    above[2] += APPROACH_HEIGHT

    print(f"[place] Moving above target {target}...")
    move_to(model, data, above, viewer)

    print("[place] Lowering...")
    move_to(model, data, target, viewer)

    print("[place] Opening gripper...")
    open_gripper(model, data, viewer)

    print("[place] Retracting...")
    move_to(model, data, above, viewer)
    print("[place] Done.")
