# MuJoCo Simulation Core

Provides expert guidance on MuJoCo simulation — MJCF XML authoring, Python API usage, simulation loops, physics tuning, and data access patterns. Grounded in this project's SO-101 robotic arm pick-and-place simulation.

---

## Activation

- **Auto-invoked** when the user asks about: MuJoCo, MJCF XML, simulation setup, scene composition, physics parameters, `mujoco` Python API, adding objects/bodies to the scene, simulation stepping, or viewer usage.
- **User-invocable** via `/mujoco-sim`

---

## Project Context

This project simulates an SO-101 6-DOF robotic arm with pick-and-place manipulation of colored blocks on a table.

### Key Files

| File | Purpose |
|------|---------|
| `sim.py` | Main simulation — model loading, trajectory execution, interactive viewer |
| `SO101/scene_blocks.xml` | Scene with table + 3 colored blocks (includes robot model) |
| `SO101/scene.xml` | Minimal scene (robot on ground plane, no objects) |
| `SO101/so101_new_calib.xml` | SO-101 robot MJCF model (joints, actuators, meshes) |
| `SO101/joints_properties.xml` | Default joint/actuator classes |
| `SO101/assets/*.stl` | 13 mesh files for robot links |

### How the Scene is Built

`sim.py` generates `scene_blocks.xml` from the `SCENE_BLOCKS_XML` constant, then loads it via `load_model()`:

```python
# sim.py — load_model() pattern
spec = mujoco.MjSpec.from_file(str(scene_path))

# Attach robot arm to world
arm = spec.worldbody.add_body(name="so101_arm", pos=ARM_POS, quat=ARM_QUAT)
child_spec = mujoco.MjSpec.from_file(str(LOCAL_DIR / "so101_new_calib.xml"))
arm.attach(child_spec)

# Create keyframe for home position
key = spec.add_key(name="home")
model = spec.compile()
# ... set keyframe qpos for each joint ...
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
```

---

## MJCF XML Authoring

### Scene Composition Pattern (scene_blocks.xml)

```xml
<mujoco model="scene_blocks">
  <include file="so101_new_calib.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.45 0.30 0.18 1"/>
    <material name="red_block"  rgba="0.9 0.15 0.15 1"/>
    <material name="green_block" rgba="0.15 0.8 0.15 1"/>
    <material name="blue_block" rgba="0.15 0.15 0.9 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

    <!-- Table -->
    <body name="table" pos="0.2 0 0.1">
      <geom type="box" size="0.2 0.2 0.01" pos="0 0 0.1" material="table_mat"/>
      <!-- 4 legs at corners -->
    </body>

    <!-- Free-floating blocks -->
    <body name="red_block" pos="0.1 0.0 0.23">
      <freejoint name="red_block_joint"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05" material="red_block"/>
    </body>
    <!-- green_block, blue_block similar -->

    <camera name="top_down" pos="0.2 0 0.8" xyaxes="1 0 0 0 1 0" fovy="60"/>
  </worldbody>
</mujoco>
```

### Adding New Objects to the Scene

To add a new object, edit the `SCENE_BLOCKS_XML` string in `sim.py` (it gets written to `SO101/scene_blocks.xml` by `generate_scene()`):

1. **Static object** — add a `<body>` with geometry under `<worldbody>`:
   ```xml
   <body name="cylinder_obj" pos="0.15 0.1 0.23">
     <geom type="cylinder" size="0.015 0.02" mass="0.03" rgba="1 0.5 0 1"/>
   </body>
   ```

2. **Dynamic (graspable) object** — add `<freejoint>` so it has 6-DOF:
   ```xml
   <body name="yellow_block" pos="0.25 -0.05 0.23">
     <freejoint name="yellow_block_joint"/>
     <geom type="box" size="0.02 0.02 0.02" mass="0.05" rgba="1 1 0 1"/>
   </body>
   ```

3. **With material** — define in `<asset>` and reference:
   ```xml
   <!-- In <asset> -->
   <material name="yellow_block" rgba="1 1 0 1"/>
   <!-- In <worldbody> -->
   <geom ... material="yellow_block"/>
   ```

### Compiler Settings (from so101_new_calib.xml)

```xml
<compiler angle="radian" meshdir="assets" autolimits="true"/>
```

- Always use **radians** (not degrees) for joint angles and limits
- Mesh files are relative to `meshdir="assets"`
- `autolimits="true"` means joint ranges auto-create limited flags

---

## Python API Reference

### MjSpec — Model Building API

```python
import mujoco

# Load from XML file
spec = mujoco.MjSpec.from_file("SO101/scene_blocks.xml")

# Add body to worldbody
body = spec.worldbody.add_body(name="my_body", pos=[0.1, 0, 0.3])
body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02])

# Attach another model file (how SO-101 is loaded)
child_spec = mujoco.MjSpec.from_file("SO101/so101_new_calib.xml")
parent_body = spec.worldbody.add_body(name="robot", pos=[0.2, -0.18, 0.21])
parent_body.attach(child_spec)

# Add keyframe
key = spec.add_key(name="home")
model = spec.compile()
# After compile, set key.qpos values, then recompile
key.qpos[joint_id] = value
model = spec.compile()
```

### MjModel — Compiled Model (Read-Only)

```python
model = spec.compile()

# Name-to-ID lookups
jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_pan")

# Joint address in qpos (for multi-DOF joints)
qpos_adr = model.jnt_qposadr[jnt_id]

# Model dimensions
model.nq   # number of qpos elements
model.nv   # number of velocity DOFs
model.nu   # number of actuators (6 for SO-101)
model.nbody  # number of bodies
```

### MjData — Simulation State (Read-Write)

```python
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # reset to "home" keyframe

# Joint positions — this project reads joint angles from qpos:
# sim.py get_current_joints() pattern:
for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    adr = model.jnt_qposadr[jid]
    value = data.qpos[adr]

# Set control (actuator commands)
data.ctrl[act_id] = target_angle

# Body positions (Cartesian) — after mj_forward/mj_step
data.xpos[body_id]    # 3D position
data.xquat[body_id]   # quaternion orientation

# Site positions
data.site_xpos[site_id]  # 3D position of site

# Sensor data
data.sensordata  # all sensor readings

# Contact information
data.ncon  # number of active contacts
data.contact[i].geom1, data.contact[i].geom2  # contacting geom IDs
```

### apply_joints() — How This Project Sets Joint Commands

```python
# sim.py pattern — sets BOTH qpos and ctrl for each joint
def apply_joints(model, data, joint_dict):
    for name, val in joint_dict.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = val
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        data.ctrl[aid] = val
```

---

## Simulation Loop Patterns

### Interactive Viewer (this project's pattern)

```python
import mujoco.viewer

model, data = load_model(scene_path)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Headless Stepping (for RL training / batch sim)

```python
model = mujoco.MjModel.from_xml_path("SO101/scene_blocks.xml")
data = mujoco.MjData(model)

for _ in range(1000):
    data.ctrl[:] = action_vector
    mujoco.mj_step(model, data)
    # Read state from data.qpos, data.xpos, etc.
```

### Forward Kinematics Only (no dynamics)

```python
mujoco.mj_forward(model, data)  # compute positions without stepping physics
```

---

## Physics Parameters

### Timestep and Integrator

```xml
<option timestep="0.002" integrator="Euler"/>
<!-- or for more accuracy -->
<option timestep="0.001" integrator="RK4"/>
```

- Default timestep: 0.002s (500 Hz)
- For stable grasping, consider smaller timestep (0.001s) or implicit integrator

### Solver Tuning

```xml
<option solver="Newton" iterations="50" tolerance="1e-10"/>
```

### Contact Parameters

```xml
<!-- Global defaults -->
<option cone="elliptic" impratio="10"/>

<!-- Per-geom contact properties -->
<geom friction="1 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
```

- `friction`: sliding, torsional, rolling
- `solref`: contact time-constant and damping ratio
- `solimp`: impedance parameters (dmin, dmax, width)

### Gravity

```xml
<option gravity="0 0 -9.81"/>
```

---

## Common Patterns and Tips

### Reading Block Positions at Runtime

```python
block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
block_pos = data.xpos[block_id]  # [x, y, z] array
```

### Checking Gripper Tip Positions

```python
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
gripper_pos = data.site_xpos[site_id]
```

### Resetting to Home

```python
mujoco.mj_resetDataKeyframe(model, data, 0)  # keyframe 0 = "home"
```

### Stepping with Substeps (finer physics)

```python
n_substeps = 4
for _ in range(n_substeps):
    mujoco.mj_step(model, data)
viewer.sync()  # only sync once per render frame
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError` on joint name | Joint name doesn't match MJCF | Check `so101_new_calib.xml` — joints: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` |
| Model won't load | Missing mesh files | Run `download_assets()` or check `SO101/assets/` directory |
| Objects fall through table | Contact filtering | Check `contype`/`conaffinity` or add collision geoms |
| Simulation explodes | Timestep too large or bad gains | Reduce timestep, check actuator `forcerange` |
| Viewer doesn't open | Display not available | Use headless mode or set `MUJOCO_GL=egl` |
