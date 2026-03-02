# Robot Control Algorithms

Provides expert guidance on trajectory planning, inverse kinematics, PID tuning, and multi-phase motion control for robotic manipulation. Grounded in this project's SO-101 arm control patterns.

---

## Activation

- **Auto-invoked** when the user asks about: trajectory planning, motion planning, interpolation, inverse kinematics (IK), PID tuning, actuator gains, control algorithms, safe trajectories, pick-and-place motion, or joint-space / task-space control.
- **User-invocable** via `/robot-control`

---

## Project Context

The SO-101 arm uses **position actuators** — you command desired joint angles and the built-in PD controller drives to them. The project currently uses joint-space interpolation with a multi-phase safe trajectory pattern.

### Key Constants (sim.py)

```python
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

HOME_JOINTS = {
    "shoulder_pan": 0, "shoulder_lift": -1.75, "elbow_flex": 1.69,
    "wrist_flex": 1.08, "wrist_roll": 0, "gripper": 0,
}

RAISED_JOINTS = {
    "shoulder_pan": 0, "shoulder_lift": -1.0, "elbow_flex": 1.2,
    "wrist_flex": 0.8, "wrist_roll": 0, "gripper": 0,
}

GRASP_CONFIGS = {
    "red_block":   {"shoulder_pan": -0.4992, "shoulder_lift": -0.0173, ...},
    "green_block": {"shoulder_pan": -0.0,    "shoulder_lift":  0.1921, ...},
    "blue_block":  {"shoulder_pan":  0.4992, "shoulder_lift": -0.0173, ...},
}
```

---

## Trajectory Planning

### Linear Interpolation (Current Implementation)

```python
# sim.py — make_motion_steps()
def make_motion_steps(current: dict, target: dict, steps: int = 200) -> list[dict]:
    """Linearly interpolate from current to target joint config."""
    trajectory = []
    for i in range(1, steps + 1):
        alpha = i / steps
        frame = {}
        for name in JOINT_NAMES:
            c = current.get(name, 0.0)
            t = target.get(name, 0.0)
            frame[name] = c + alpha * (t - c)
        trajectory.append(frame)
    return trajectory
```

### Cubic Spline Interpolation (Smoother Motion)

```python
import numpy as np
from scipy.interpolate import CubicSpline

def make_cubic_trajectory(current, target, steps=200):
    """Cubic spline with zero velocity at endpoints."""
    t_knots = np.array([0.0, 1.0])
    trajectory = []
    splines = {}
    for name in JOINT_NAMES:
        c = current.get(name, 0.0)
        t = target.get(name, 0.0)
        # bc_type=clamped → zero velocity at start and end
        splines[name] = CubicSpline(t_knots, [c, t], bc_type="clamped")

    for i in range(1, steps + 1):
        alpha = i / steps
        frame = {name: float(splines[name](alpha)) for name in JOINT_NAMES}
        trajectory.append(frame)
    return trajectory
```

### Trapezoidal Velocity Profile (Acceleration-Limited)

```python
def make_trapezoidal_trajectory(current, target, steps=200, accel_frac=0.25):
    """Trapezoidal velocity profile — constant acceleration, cruise, deceleration."""
    trajectory = []
    accel_steps = int(steps * accel_frac)
    decel_start = steps - accel_steps

    for i in range(1, steps + 1):
        if i <= accel_steps:
            # Acceleration phase (quadratic position)
            alpha = 0.5 * (i / accel_steps) ** 2 * (accel_steps / steps) * 2
        elif i >= decel_start:
            # Deceleration phase
            remaining = (steps - i) / accel_steps
            alpha = 1.0 - 0.5 * remaining ** 2 * (accel_steps / steps) * 2
        else:
            # Cruise phase (linear)
            alpha = (i - accel_steps / 2) / (steps - accel_steps)
        alpha = np.clip(alpha, 0, 1)
        frame = {}
        for name in JOINT_NAMES:
            c = current.get(name, 0.0)
            t = target.get(name, 0.0)
            frame[name] = c + alpha * (t - c)
        trajectory.append(frame)
    return trajectory
```

---

## Multi-Phase Safe Trajectory (This Project's Pattern)

The `make_safe_trajectory()` function in `sim.py` prevents collisions by moving the arm through safe intermediate waypoints:

```
Phase 1a: RETRACT  — Pull wrist/elbow to safe angles
Phase 1b: LIFT     — Raise shoulder to RAISED_JOINTS height
Phase 2:  PAN      — Rotate shoulder_pan to target while raised
Phase 3:  LOWER    — Descend to pre-grasp (blocks) or target (home)
Phase 4:  DESCEND  — Final approach from pre-grasp to grasp (blocks only)
```

### Pre-Grasp Derivation

```python
# sim.py — derive approach-from-above configuration
def make_pre_grasp(grasp: dict) -> dict:
    pre = dict(grasp)
    pre["shoulder_lift"] = grasp["shoulder_lift"] - 0.3   # lift shoulder ~17 deg
    pre["elbow_flex"]    = grasp["elbow_flex"]    + 0.15   # bend elbow slightly
    return pre
```

### Key Design Decisions
- **Why retract first?** Prevents sweeping through objects on the table
- **Why lift before pan?** Keeps arm above table height during rotation
- **Why pre-grasp?** Approaches blocks from above to avoid lateral collisions
- **`is_arm_raised()`** checks `shoulder_lift < -0.6` to determine if arm is already safe

---

## Inverse Kinematics

### Jacobian-Based IK with MuJoCo

```python
import mujoco
import numpy as np

def ik_solve(model, data, target_pos, site_name="gripperframe",
             joint_names=None, max_iter=100, tol=1e-3, damping=1e-3):
    """Damped least-squares IK to reach target_pos with specified site."""
    if joint_names is None:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll"]  # exclude gripper

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    dof_ids = [model.jnt_dofadr[j] for j in jnt_ids]
    n_dof = len(dof_ids)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.site_xpos[site_id]
        error = target_pos - current_pos

        if np.linalg.norm(error) < tol:
            break

        # Compute full Jacobian
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # Extract columns for our joints only
        J = jacp[:, dof_ids]

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Apply joint updates
        for i, jid in enumerate(jnt_ids):
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] += dq[i]
            # Clip to joint limits
            data.qpos[adr] = np.clip(data.qpos[adr],
                                      model.jnt_range[jid, 0],
                                      model.jnt_range[jid, 1])

    return {name: data.qpos[model.jnt_qposadr[jid]]
            for name, jid in zip(joint_names, jnt_ids)}
```

### Using `mj_jac` for Body Jacobians

```python
# Jacobian for a body (instead of site)
jacp = np.zeros((3, model.nv))  # position Jacobian
jacr = np.zeros((3, model.nv))  # rotation Jacobian
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
```

---

## PID / Actuator Tuning

### SO-101 Actuator Configuration (sts3215 class)

```xml
<default class="sts3215">
  <joint  damping="0.60" frictionloss="0.052" armature="0.028"/>
  <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
</default>
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `kp` | 998.22 | Proportional gain — stiffness of position tracking |
| `kv` | 2.731 | Derivative gain — velocity damping |
| `damping` | 0.60 | Joint viscous damping (passive) |
| `frictionloss` | 0.052 | Dry friction torque |
| `armature` | 0.028 | Rotor inertia (stabilizes simulation) |
| `forcerange` | [-2.94, 2.94] Nm | Actuator torque limits (some joints use [-3.35, 3.35]) |

### Tuning Guidelines

- **Oscillation/overshoot** → increase `kv` or `damping`, or decrease `kp`
- **Sluggish response** → increase `kp`, decrease `damping`
- **High-frequency vibration** → increase `armature` (adds rotor inertia)
- **Unrealistic snapping** → reduce `forcerange` to match real servo torque
- **Gripper slipping** → increase friction on block geoms, tune `solref`/`solimp`

### Modifying Gains at Runtime

```python
# Change actuator gain via model (requires recompile for MjSpec, but
# for MjModel you can modify model.actuator_gainprm directly)
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_lift")
model.actuator_gainprm[act_id, 0] = new_kp  # gainprm[0] = kp for position actuator
model.actuator_biasprm[act_id, 1] = -new_kp  # biasprm[1] = -kp
model.actuator_biasprm[act_id, 2] = -new_kv  # biasprm[2] = -kv
```

---

## Operational Space Control

```python
def operational_space_control(model, data, target_pos, target_vel=None,
                               site_name="gripperframe", kp=100, kd=20):
    """Task-space PD control with dynamics compensation."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_forward(model, data)

    # Position error
    current_pos = data.site_xpos[site_id]
    pos_error = target_pos - current_pos

    # Jacobian
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    # Task-space force
    if target_vel is None:
        target_vel = np.zeros(3)
    current_vel = jacp @ data.qvel
    f_task = kp * pos_error + kd * (target_vel - current_vel)

    # Map to joint torques
    tau = jacp.T @ f_task

    # Apply (for motor actuators, set ctrl = tau)
    data.ctrl[:model.nv] = tau[:model.nu]
```

> **Note:** The SO-101 uses position actuators, not motor/torque actuators. To use operational space control, you'd need to change actuator types to `motor` in the MJCF.

---

## Collision Avoidance Strategies

1. **Waypoint-based** (current approach): Pre-computed safe waypoints via `make_safe_trajectory()`
2. **Joint-limit margins**: Keep joints away from limits by a safety margin
3. **Workspace boundary**: Check gripper site position against table bounds before executing
4. **Potential field**: Add repulsive force from obstacles:
   ```python
   def check_collision_risk(model, data, threshold=0.05):
       """Check if any contacts have penetration > threshold."""
       for i in range(data.ncon):
           if data.contact[i].dist < -threshold:
               return True
       return False
   ```

---

## Executing Trajectories (This Project's Pattern)

```python
# sim.py — how trajectories are executed in the viewer loop
trajectory = make_safe_trajectory(current_joints, target_joints,
                                  target_name="red_block", steps_per_seg=80)
traj_idx = 0

# Inside the viewer while loop:
if trajectory and traj_idx < len(trajectory):
    apply_joints(model, data, trajectory[traj_idx])
    traj_idx += 1
mujoco.mj_step(model, data)
viewer.sync()
```
