# Debugging Robotics Simulations

Provides expert guidance on diagnosing and fixing MuJoCo simulation issues — instability, contact problems, joint violations, actuator saturation, performance profiling, and visualization tools.

---

## Activation

- **Auto-invoked** when the user asks about: simulation instability, NaN values, simulation exploding, contact issues, penetration, joint limits, actuator saturation, debugging, profiling, visualization of forces, or common MuJoCo errors.
- **User-invocable** via `/robot-debug`

---

## Project Context

| File | Relevance |
|------|-----------|
| `sim.py` | Main simulation loop, trajectory execution |
| `SO101/so101_new_calib.xml` | Robot model (joints, actuators, collision) |
| `SO101/scene_blocks.xml` | Scene with table + blocks |

### Key Parameters to Watch

- Timestep: default 0.002s (in `<option>`)
- Actuator kp: 998.22, kv: 2.731 (sts3215 class)
- Forcerange: [-2.94, 2.94] Nm per joint (some [-3.35, 3.35])
- Block mass: 0.05 kg, size: 0.02³ m

---

## Simulation Instability Diagnosis

### Symptoms and Causes

| Symptom | Likely Cause | Quick Fix |
|---------|-------------|-----------|
| Objects fly away | Timestep too large | Reduce `timestep` to 0.001 or smaller |
| NaN in qpos/qvel | Divergent dynamics | Check for zero-mass bodies, reduce gains |
| Robot vibrates rapidly | kp too high, kv too low | Increase kv or decrease kp |
| Blocks jitter on table | Contact parameters too stiff | Tune `solref`, `solimp` |
| Slow drift | Accumulated numerical error | Use `integrator="implicit"` or RK4 |
| Gripper passes through block | Missing collision | Check `contype`/`conaffinity` |

### NaN Detection Script

```python
import mujoco
import numpy as np

def check_simulation_health(model, data):
    """Check for common simulation issues. Call after mj_step."""
    issues = []

    # NaN check
    if np.any(np.isnan(data.qpos)):
        issues.append(f"NaN in qpos: indices {np.where(np.isnan(data.qpos))[0]}")
    if np.any(np.isnan(data.qvel)):
        issues.append(f"NaN in qvel: indices {np.where(np.isnan(data.qvel))[0]}")
    if np.any(np.isnan(data.qacc)):
        issues.append(f"NaN in qacc: indices {np.where(np.isnan(data.qacc))[0]}")

    # Velocity explosion check
    max_vel = np.max(np.abs(data.qvel))
    if max_vel > 100:
        issues.append(f"Extreme velocity: max |qvel| = {max_vel:.1f}")

    # Energy check (if increasing rapidly, simulation is diverging)
    ke = 0.5 * data.qvel @ (model.dof_M0 * data.qvel)  # approximate KE
    if ke > 1000:
        issues.append(f"High kinetic energy: {ke:.1f}")

    return issues
```

### Stability Fixes

```xml
<!-- More stable integrator -->
<option timestep="0.001" integrator="implicit"/>

<!-- Or increase solver iterations -->
<option solver="Newton" iterations="100" tolerance="1e-12"/>

<!-- Add damping to all joints as safety net -->
<default>
  <joint damping="2.0"/>
</default>
```

---

## Contact Debugging

### Inspecting Active Contacts

```python
def print_contacts(model, data, verbose=True):
    """Print all active contacts in the simulation."""
    print(f"Active contacts: {data.ncon}")
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        print(f"  [{i}] {geom1_name} <-> {geom2_name}")
        if verbose:
            print(f"       pos: {contact.pos}")
            print(f"       dist: {contact.dist:.6f}")  # negative = penetration
            print(f"       frame: {contact.frame[:3]}")  # contact normal
```

### Checking Penetration Depth

```python
def check_penetration(model, data, threshold=0.005):
    """Warn about excessive contact penetration."""
    for i in range(data.ncon):
        if data.contact[i].dist < -threshold:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2)
            print(f"WARNING: Penetration {data.contact[i].dist:.4f}m between {g1} and {g2}")
```

### Contact Parameter Tuning

```xml
<!-- Softer contacts (less bouncy, more stable) -->
<geom solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>

<!-- More friction for grasping -->
<geom friction="1.5 0.01 0.001"/>
```

| `solref` param | Meaning | Effect of increasing |
|----------------|---------|---------------------|
| solref[0] | Time constant | Softer contact (more penetration) |
| solref[1] | Damping ratio | Less bouncy (>1 = overdamped) |

| `solimp` param | Meaning | Range |
|----------------|---------|-------|
| solimp[0] | Min impedance | [0, 1] |
| solimp[1] | Max impedance | [0, 1] |
| solimp[2] | Width | >0, transition width |

---

## Joint Limit Violations

### Detecting Violations

```python
def check_joint_limits(model, data):
    """Check if any joints are at or beyond their limits."""
    for i in range(model.njnt):
        if not model.jnt_limited[i]:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        adr = model.jnt_qposadr[i]
        pos = data.qpos[adr]
        lo, hi = model.jnt_range[i]
        margin = 0.01  # 0.01 rad margin

        if pos <= lo + margin:
            print(f"WARNING: {name} at LOWER limit ({pos:.4f} <= {lo:.4f})")
        elif pos >= hi - margin:
            print(f"WARNING: {name} at UPPER limit ({pos:.4f} >= {hi:.4f})")
```

### SO-101 Joint Limits

| Joint | Min (rad) | Max (rad) | Common Issue |
|-------|-----------|-----------|--------------|
| shoulder_pan | -1.9198 | 1.9198 | Hit during wide sweeps |
| shoulder_lift | -1.7453 | 1.7453 | Hit at extreme tilt |
| elbow_flex | -1.69 | 1.69 | Hit when fully extended/folded |
| wrist_flex | -1.6580 | 1.6580 | Hit during aggressive wrist motion |
| wrist_roll | -2.7438 | 2.8412 | Widest range, rarely hit |
| gripper | -0.1745 | 1.7453 | Lower limit = closed, upper = open |

---

## Actuator Saturation

### Detecting Saturated Actuators

```python
def check_actuator_saturation(model, data):
    """Check if actuators are hitting their force limits."""
    mujoco.mj_forward(model, data)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        force = data.actuator_force[i]
        lo = model.actuator_forcerange[i, 0]
        hi = model.actuator_forcerange[i, 1]

        if lo != 0 or hi != 0:  # forcerange is set
            if abs(force - lo) < 0.01 or abs(force - hi) < 0.01:
                print(f"SATURATED: {name} force={force:.3f} range=[{lo:.2f}, {hi:.2f}]")
            elif abs(force) > 0.8 * hi:
                print(f"NEAR LIMIT: {name} force={force:.3f} ({abs(force/hi)*100:.0f}% of max)")
```

### Common Saturation Causes

- **Gravity load on shoulder_lift** — arm weight exceeds torque limit
- **Fast trajectory** — demanding large accelerations
- **Grasping heavy objects** — gripper force insufficient
- **Fix:** Increase `forcerange`, reduce trajectory speed, add counterweight

---

## Performance Profiling

### Timing mj_step

```python
import time

def profile_simulation(model, data, n_steps=1000):
    """Profile simulation step performance."""
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        mujoco.mj_step(model, data)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # ms
    print(f"mj_step timing over {n_steps} steps:")
    print(f"  Mean:   {times.mean():.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Max:    {times.max():.3f} ms")
    print(f"  Real-time factor: {model.opt.timestep * 1000 / times.mean():.1f}x")
```

### Breakdown by Component

```python
# MuJoCo provides timer data
mujoco.mj_step(model, data)
print("Timer breakdown (microseconds):")
timer_names = ["step", "forward", "inverse", "position", "velocity",
               "actuation", "acceleration", "constraint", "advance"]
for i, name in enumerate(timer_names):
    if i < len(data.timer):
        print(f"  {name}: {data.timer[i]:.1f}")
```

### Collision Overhead

If simulation is slow, check collision count:

```python
print(f"Geom pairs checked: {data.ncon}")
print(f"Total geoms: {model.ngeom}")
# Reduce by using contype/conaffinity filtering or simplifying collision geoms
```

---

## Common MuJoCo Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `XML parse error` | Malformed MJCF | Check XML syntax, attribute names |
| `mesh file not found` | Wrong `meshdir` path | Verify `compiler meshdir="assets"` and file exists |
| `joint has no DOF` | Joint inside static body | Ensure body is in a proper kinematic chain |
| `actuator references invalid joint` | Typo in joint name | Check actuator `joint=` attribute matches joint `name=` |
| `qpos has wrong size` | Changed model without updating keyframe | Recompile keyframe with correct qpos size |
| `unstable simulation` | See instability section | Reduce timestep, check masses/gains |
| `contact too deep` | Fast-moving objects or large timestep | Reduce timestep, increase solver iterations |
| `body has zero mass` | No geom with mass or density | Add `mass` to geom or `<inertial>` to body |

---

## Visualization Tools

### Show Contact Forces in Viewer

```python
# Enable contact force visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Toggle visualization flags
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Available Visualization Flags

```python
# Contact visualization
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True   # contact force arrows
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True   # contact points
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True     # constraint forces

# Joint/body visualization
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True          # joint axes
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True            # center of mass
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True        # inertia ellipsoids

# Geom groups (toggle collision geoms)
viewer.opt.geomgroup[0] = True   # group 0 (visual geoms)
viewer.opt.geomgroup[3] = True   # group 3 (collision geoms)
```

### Comprehensive Debug Printout

```python
def debug_state(model, data):
    """Print comprehensive simulation state for debugging."""
    print("=== SIMULATION STATE ===")
    print(f"Time: {data.time:.4f}s")

    print("\nJoint positions:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            adr = model.jnt_qposadr[i]
            lo, hi = model.jnt_range[i]
            pos = data.qpos[adr]
            vel = data.qvel[model.jnt_dofadr[i]]
            print(f"  {name:20s}: pos={pos:+.4f} vel={vel:+.4f} range=[{lo:.2f}, {hi:.2f}]")

    print("\nActuator forces:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl = data.ctrl[i]
        force = data.actuator_force[i]
        lo, hi = model.actuator_forcerange[i]
        print(f"  {name:20s}: ctrl={ctrl:+.4f} force={force:+.4f} range=[{lo:.2f}, {hi:.2f}]")

    print(f"\nContacts: {data.ncon}")
    for i in range(min(data.ncon, 10)):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
        print(f"  [{i}] {g1} <-> {g2}, dist={c.dist:.5f}")

    # Health check
    issues = check_simulation_health(model, data)
    if issues:
        print("\n!!! ISSUES DETECTED !!!")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nSimulation healthy.")
```
