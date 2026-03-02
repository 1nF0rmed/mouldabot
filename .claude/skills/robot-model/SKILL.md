# MJCF/URDF Robot Models

Provides expert guidance on robot model structure — kinematic chains, joint configuration, actuator types, collision classes, mesh integration, and the SO-101 model specifics.

---

## Activation

- **Auto-invoked** when the user asks about: robot model, MJCF structure, URDF, kinematic chain, joints, actuators, collision geoms, meshes, servo parameters, body hierarchy, joint limits, or the SO-101 model XML.
- **User-invocable** via `/robot-model`

---

## Project Context

The SO-101 is a 6-DOF robotic arm using STS3215 servos, modeled in `SO101/so101_new_calib.xml`. It was generated from Onshape CAD via `onshape-to-robot`.

---

## SO-101 Kinematic Chain

```
worldbody
  └─ so101_arm (attached at pos=[0.2, -0.18, 0.21], quat=[0.707, 0, 0, 0.707])
     └─ base (root body — no joint, fixed to world)
        └─ shoulder (joint: shoulder_pan, hinge, axis=[0,0,1])
           └─ upper_arm (joint: shoulder_lift, hinge)
              └─ lower_arm (joint: elbow_flex, hinge)
                 └─ wrist (joint: wrist_flex, hinge)
                    └─ gripper (joint: wrist_roll, hinge)
                       ├─ site: gripperframe
                       ├─ site: fixed_jaw_tip
                       └─ moving_jaw_so101_v1 (joint: gripper, hinge)
                          └─ site: moving_jaw_tip
```

### Joint Specifications

| Joint | Type | Axis | Range (rad) | Range (deg) | Parent Body | Child Body |
|-------|------|------|-------------|-------------|-------------|------------|
| `shoulder_pan` | hinge | [0,0,1] | [-1.9198, 1.9198] | [-110, 110] | base | shoulder |
| `shoulder_lift` | hinge | [0,0,1] | [-1.7453, 1.7453] | [-100, 100] | shoulder | upper_arm |
| `elbow_flex` | hinge | [0,0,1] | [-1.69, 1.69] | [-96.8, 96.8] | upper_arm | lower_arm |
| `wrist_flex` | hinge | [0,0,1] | [-1.6580, 1.6580] | [-95, 95] | lower_arm | wrist |
| `wrist_roll` | hinge | [0,0,1] | [-2.7438, 2.8412] | [-157, 163] | wrist | gripper |
| `gripper` | hinge | [0,0,1] | [-0.1745, 1.7453] | [-10, 100] | gripper | moving_jaw |

### Sites (End-Effector Reference Points)

| Site | Body | Position | Use |
|------|------|----------|-----|
| `baseframe` | base | origin | Robot base reference |
| `gripperframe` | gripper | [-0.0079, -0.0002, -0.0981] | Gripper center point |
| `fixed_jaw_tip` | gripper | [-0.0079, -0.0002, -0.0981] | Fixed jaw contact |
| `moving_jaw_tip` | moving_jaw | [0.0, -0.058, 0.019] | Moving jaw contact |

---

## MJCF Body-Joint-Body Nesting

In MJCF, the kinematic chain is built by nesting `<body>` elements. Each child body can have a `<joint>` that defines its motion relative to its parent:

```xml
<body name="parent">
  <!-- parent geoms here -->
  <body name="child" pos="0 0 0.1">
    <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
    <!-- child geoms here -->
    <body name="grandchild" pos="0 0 0.1">
      <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <!-- grandchild geoms here -->
    </body>
  </body>
</body>
```

### Joint Types

| Type | DOF | Description |
|------|-----|-------------|
| `hinge` | 1 | Rotation about axis (all SO-101 joints) |
| `slide` | 1 | Translation along axis |
| `ball` | 3 | 3-DOF spherical joint (qpos = quaternion) |
| `free` | 6 | 6-DOF floating (qpos = pos + quaternion, used for blocks) |

---

## Actuator Types and Configuration

### Position Actuator (SO-101 uses this)

Commands a desired position; internal PD controller computes torque:

```xml
<actuator>
  <position name="shoulder_pan" joint="shoulder_pan" class="sts3215"
            forcerange="-3.35 3.35" ctrlrange="-1.91986 1.91986"/>
</actuator>
```

**Control law:** `torque = kp * (ctrl - qpos) - kv * qvel`

### Velocity Actuator

Commands a desired velocity:

```xml
<velocity name="joint1_vel" joint="joint1" kv="10" forcerange="-5 5"/>
```

### Motor (Torque) Actuator

Direct torque control:

```xml
<motor name="joint1_motor" joint="joint1" forcerange="-5 5"/>
```

Use motor actuators for operational-space control or custom control laws.

### Comparing Actuator Types

| Type | Input | Internal Control | Best For |
|------|-------|-----------------|----------|
| `position` | Target angle | PD controller | Position tracking (servos) |
| `velocity` | Target velocity | P controller on velocity | Speed control |
| `motor` | Torque | None (direct) | Custom controllers, RL |

---

## STS3215 Servo Properties

The SO-101 uses Feetech STS3215 serial bus servos. Parameters in `so101_new_calib.xml`:

```xml
<default class="sts3215">
  <joint  damping="0.60" frictionloss="0.052" armature="0.028"/>
  <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
</default>
```

**Real servo specs:**
- Stall torque: ~3.5 Nm (6V), modeled as `forcerange="-2.94 2.94"` (conservative)
- Operating voltage: 6-8.4V
- Baudrate: 1,000,000 bps (serial bus)
- Resolution: 4096 steps/revolution (~0.088 deg)

**Note:** `joints_properties.xml` has an alternative `sts3215` class with `kp=17.8` (much softer). The main model uses `kp=998.22` from `so101_new_calib.xml`.

---

## Collision Classes

The SO-101 model uses two geom groups to separate visual rendering from physics collision:

```xml
<default class="visual">
  <geom type="mesh" contype="0" conaffinity="0" group="0"/>
</default>

<default class="collision">
  <geom group="3"/>
</default>
```

### How Collision Filtering Works

- `contype` and `conaffinity` are bitmasks
- Contact occurs when `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity)`
- Visual geoms: `contype="0" conaffinity="0"` → never collide
- Collision geoms: default `contype="1" conaffinity="1"` → collide with everything

### Adding a New Collision Pair

To make blocks only collide with the gripper and table (not the arm links):

```xml
<!-- Blocks: contype=2 -->
<geom type="box" size="0.02 0.02 0.02" contype="2" conaffinity="3"/>
<!-- Gripper collision geom: conaffinity includes bit 2 -->
<geom class="collision" contype="1" conaffinity="3"/>
<!-- Arm links: only conaffinity=1, won't detect contype=2 -->
```

---

## Mesh Integration

### Directory Setup

```xml
<compiler meshdir="assets"/>  <!-- relative to XML file -->
```

### Declaring Mesh Assets

```xml
<asset>
  <mesh name="base_so101_v2" file="base_so101_v2.stl"/>
  <mesh name="sts3215_03a_v1" file="sts3215_03a_v1.stl"/>
  <!-- ... 13 total meshes -->
</asset>
```

### Using Meshes in Bodies

```xml
<body name="base">
  <!-- Visual mesh (rendered but no collision) -->
  <geom class="visual" mesh="base_so101_v2"/>
  <!-- Collision approximation (simplified shape) -->
  <geom class="collision" type="box" size="0.03 0.03 0.02" pos="0 0 0.01"/>
</body>
```

### Mesh Tips

- STL files should have units in meters
- MuJoCo auto-computes inertia from mesh geometry
- For faster collision, use primitive shapes (box, cylinder, capsule) as collision geoms instead of mesh
- Scale meshes: `<mesh name="..." file="..." scale="0.001 0.001 0.001"/>` (mm → m)

---

## Adding a New Robot Model

### From URDF

```python
# MuJoCo can load URDF directly
model = mujoco.MjModel.from_xml_path("robot.urdf")

# Or convert URDF to MJCF for more control
# Use: python -m mujoco.mjcf_from_urdf robot.urdf robot.xml
```

### From Onshape (how SO-101 was created)

1. Install `onshape-to-robot`: `pip install onshape-to-robot`
2. Configure with Onshape document URL
3. Export MJCF with meshes
4. Tune joint properties, actuators, and collision manually

### Building from Scratch

```xml
<mujoco model="my_robot">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <default>
    <joint damping="1" armature="0.01"/>
    <position kp="100" kv="5"/>
  </default>

  <asset>
    <mesh name="link1" file="link1.stl"/>
  </asset>

  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.05 0.02" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="mesh" mesh="link1"/>
        <!-- continue chain... -->
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="joint1" joint="joint1"/>
  </actuator>
</mujoco>
```

---

## Debugging Model Issues

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Joint moves wrong direction | Axis orientation | Check `axis` attribute, try negating |
| Joint range violated at start | Initial qpos outside range | Set valid initial qpos in keyframe |
| Mesh looks wrong | Scale or orientation | Check STL units, add `scale` to `<mesh>` |
| Self-collision | Collision geoms overlap at rest | Adjust collision geom sizes or use `contype`/`conaffinity` filtering |
| Inertia warnings | Auto-inertia from mesh too small | Add explicit `<inertial>` or increase `armature` |
