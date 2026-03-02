# Sim-to-Real Transfer

Provides expert guidance on transferring learned policies and control from MuJoCo simulation to the physical SO-101 robot — domain randomization, system identification, action space matching, policy export, and calibration.

---

## Activation

- **Auto-invoked** when the user asks about: sim-to-real, domain randomization, system identification, transferring to real robot, policy deployment, servo calibration, ONNX export, real-world deployment, or matching simulation to hardware.
- **User-invocable** via `/sim-to-real`

---

## Project Context

- **Simulated robot:** SO-101 6-DOF arm in MuJoCo (`SO101/so101_new_calib.xml`)
- **Real robot:** SO-101 with Feetech STS3215 serial bus servos
- **Actuators:** Position-controlled (command angle → servo PD controller)
- **Key params:** kp=998.22, kv=2.731, damping=0.60, armature=0.028, forcerange=[-2.94, 2.94]

---

## Domain Randomization

Randomize simulation parameters during training so the learned policy is robust to real-world variations.

### Physics Randomization

```python
import mujoco
import numpy as np

def randomize_physics(model, rng):
    """Randomize physics parameters for domain randomization."""

    # Joint damping (±30%)
    for i in range(model.njnt):
        base_damping = 0.60
        model.dof_damping[i] = base_damping * rng.uniform(0.7, 1.3)

    # Joint friction (±50%)
    for i in range(model.njnt):
        base_friction = 0.052
        model.dof_frictionloss[i] = base_friction * rng.uniform(0.5, 1.5)

    # Armature / rotor inertia (±30%)
    for i in range(model.njnt):
        base_armature = 0.028
        model.dof_armature[i] = base_armature * rng.uniform(0.7, 1.3)

    # Actuator gains (±20%)
    for i in range(model.nu):
        base_kp = 998.22
        new_kp = base_kp * rng.uniform(0.8, 1.2)
        model.actuator_gainprm[i, 0] = new_kp
        model.actuator_biasprm[i, 1] = -new_kp
        model.actuator_biasprm[i, 2] = -2.731 * rng.uniform(0.8, 1.2)

    # Gravity (±5%)
    model.opt.gravity[2] = -9.81 * rng.uniform(0.95, 1.05)

    # Object mass (±50%)
    for name in ["red_block", "green_block", "blue_block"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            model.body_mass[bid] = 0.05 * rng.uniform(0.5, 1.5)

    # Object friction
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            model.geom_friction[i, 0] = rng.uniform(0.5, 1.5)  # sliding

    return model
```

### Visual Randomization

```python
def randomize_visual(model, rng):
    """Randomize visual properties (for vision-based policies)."""

    # Light position and intensity
    for i in range(model.nlight):
        model.light_pos[i] += rng.uniform(-0.3, 0.3, size=3)
        model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)

    # Object colors (slight variation)
    for i in range(model.ngeom):
        noise = rng.uniform(-0.1, 0.1, size=4)
        noise[3] = 0  # keep alpha
        model.geom_rgba[i] = np.clip(model.geom_rgba[i] + noise, 0, 1)

    # Camera position noise
    for i in range(model.ncam):
        model.cam_pos[i] += rng.uniform(-0.02, 0.02, size=3)

    return model
```

### Dynamics Randomization

```python
def randomize_dynamics(model, rng):
    """Randomize contact/solver parameters."""

    # Contact stiffness and damping
    model.opt.o_solref[0] = 0.02 * rng.uniform(0.5, 2.0)  # time constant
    model.opt.o_solref[1] = 1.0 * rng.uniform(0.5, 2.0)   # damping ratio

    # Timestep variation (simulate varying control frequencies)
    model.opt.timestep = 0.002 * rng.uniform(0.8, 1.2)

    return model
```

### Integration into RL Environment

```python
class SO101PickPlaceEnv(gym.Env):
    def __init__(self, domain_randomize=True):
        self.domain_randomize = domain_randomize
        # ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.domain_randomize:
            randomize_physics(self.model, self.np_random)
            randomize_dynamics(self.model, self.np_random)

        # Randomize block position
        # ...
        return self._get_obs(), {}
```

---

## System Identification

Match simulation parameters to the real robot by measuring actual servo behavior.

### Servo Response Measurement

```python
def measure_servo_response(real_robot, joint_name, target_angle, dt=0.01):
    """
    Command a step input on the real robot and record position over time.
    Returns time and position arrays for fitting kp/kv/damping.
    """
    positions = []
    times = []
    real_robot.set_position(joint_name, target_angle)
    for i in range(500):  # 5 seconds at 100Hz
        pos = real_robot.read_position(joint_name)
        positions.append(pos)
        times.append(i * dt)
    return np.array(times), np.array(positions)
```

### Fitting PD Gains

```python
from scipy.optimize import minimize

def fit_pd_params(times, positions, target):
    """Fit kp, kv, damping to match real servo step response."""

    def simulate_pd(params):
        kp, kv, damping = params
        dt = times[1] - times[0]
        pos, vel = positions[0], 0.0
        sim_pos = [pos]
        for _ in range(len(times) - 1):
            torque = kp * (target - pos) - kv * vel - damping * vel
            vel += torque * dt
            pos += vel * dt
            sim_pos.append(pos)
        return np.array(sim_pos)

    def loss(params):
        sim_pos = simulate_pd(params)
        return np.mean((sim_pos - positions) ** 2)

    result = minimize(loss, x0=[500, 5, 1], bounds=[(1, 2000), (0.1, 50), (0, 10)])
    return {"kp": result.x[0], "kv": result.x[1], "damping": result.x[2]}
```

### Updating MJCF from Identified Parameters

```python
def update_model_params(spec, params_per_joint):
    """Update MjSpec with identified servo parameters."""
    model = spec.compile()
    for jname, params in params_per_joint.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)

        model.actuator_gainprm[aid, 0] = params["kp"]
        model.actuator_biasprm[aid, 1] = -params["kp"]
        model.actuator_biasprm[aid, 2] = -params["kv"]
        model.dof_damping[model.jnt_dofadr[jid]] = params["damping"]
    return model
```

---

## Action Space Matching

### Simulation ↔ Real Servo Mapping

| Aspect | Simulation | Real STS3215 |
|--------|-----------|--------------|
| Units | Radians | 0–4095 (12-bit) |
| Convention | 0 = center for most joints | 2048 = center |
| Control | `data.ctrl[i] = angle_rad` | `servo.write(id, position_raw)` |
| Frequency | 500 Hz (dt=0.002) | ~50-100 Hz (bus limited) |

### Conversion Functions

```python
import math

def rad_to_servo(angle_rad, offset=2048, scale=4096 / (2 * math.pi)):
    """Convert radians to STS3215 raw position."""
    return int(offset + angle_rad * scale)

def servo_to_rad(raw_pos, offset=2048, scale=4096 / (2 * math.pi)):
    """Convert STS3215 raw position to radians."""
    return (raw_pos - offset) / scale

# Joint-specific offsets (calibrated per robot)
JOINT_OFFSETS = {
    "shoulder_pan":  2048,
    "shoulder_lift": 2048,
    "elbow_flex":    2048,
    "wrist_flex":    2048,
    "wrist_roll":    2048,
    "gripper":       2048,
}
```

### Control Frequency Matching

The simulation runs at 500 Hz but the real bus servo is ~50–100 Hz:

```python
class RealRobotController:
    def __init__(self, control_freq=50):
        self.dt = 1.0 / control_freq

    def execute_policy(self, policy, obs_fn):
        """Run policy at real control frequency."""
        while True:
            obs = obs_fn()
            action = policy.predict(obs, deterministic=True)
            self.send_joint_commands(action)
            time.sleep(self.dt)

    def send_joint_commands(self, target_angles):
        """Send to STS3215 servos via serial bus."""
        for name, angle in zip(JOINT_NAMES, target_angles):
            raw = rad_to_servo(angle, offset=JOINT_OFFSETS[name])
            self.bus.write_position(SERVO_IDS[name], raw)
```

---

## Policy Export

### ONNX Export (Stable-Baselines3)

```python
import torch
from stable_baselines3 import PPO

model = PPO.load("ppo_so101_pick")
policy = model.policy

# Get a dummy observation
dummy_obs = torch.zeros(1, model.observation_space.shape[0])

# Export actor network
torch.onnx.export(
    policy.mlp_extractor,
    dummy_obs,
    "so101_policy.onnx",
    input_names=["observation"],
    output_names=["features"],
    dynamic_axes={"observation": {0: "batch"}, "features": {0: "batch"}},
)
```

### TorchScript Export

```python
class PolicyWrapper(torch.nn.Module):
    def __init__(self, sb3_model):
        super().__init__()
        self.features = sb3_model.policy.mlp_extractor
        self.action_net = sb3_model.policy.action_net

    def forward(self, obs):
        features = self.features(obs)
        return self.action_net(features[0])  # pi features

wrapper = PolicyWrapper(model)
scripted = torch.jit.script(wrapper)
scripted.save("so101_policy.pt")
```

### NumPy-Only Inference (No PyTorch on Robot)

```python
import numpy as np

def extract_weights(sb3_model):
    """Extract MLP weights for pure NumPy inference."""
    params = sb3_model.policy.state_dict()
    weights = []
    for key in sorted(params.keys()):
        if "weight" in key or "bias" in key:
            weights.append(params[key].numpy())
    return weights

def numpy_inference(obs, weights):
    """Forward pass with ReLU activations."""
    x = obs
    for i in range(0, len(weights) - 2, 2):
        x = x @ weights[i].T + weights[i + 1]
        x = np.maximum(x, 0)  # ReLU
    x = x @ weights[-2].T + weights[-1]  # output (no activation)
    return x
```

---

## Calibration Workflow

### Step-by-Step Calibration

1. **Home position calibration** — Command all servos to center (2048) and record physical joint angles
2. **Joint direction** — Verify positive rotation direction matches simulation convention
3. **Joint limits** — Slowly sweep each joint to physical limits, record raw values
4. **Offset mapping** — Compute per-joint offset: `offset = raw_center - rad_to_servo(sim_center)`
5. **Step response test** — Command step inputs, measure rise time and overshoot
6. **Gravity compensation** — Hold arm at various poses, measure steady-state error
7. **Payload test** — Grasp block, verify no drift in holding position

### Validation Script

```python
def validate_calibration(real_robot, sim_model, sim_data, test_configs):
    """Compare sim and real joint positions for validation configs."""
    errors = {}
    for name, config in test_configs.items():
        # Set sim to config
        for jname, angle in config.items():
            jid = mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            sim_data.qpos[sim_model.jnt_qposadr[jid]] = angle
        mujoco.mj_forward(sim_model, sim_data)
        sim_gripper = sim_data.site_xpos[gripper_site_id].copy()

        # Set real to config
        for jname, angle in config.items():
            real_robot.set_position(jname, angle)
        time.sleep(2)  # wait for settling
        real_gripper = real_robot.get_gripper_position()  # from external tracker

        errors[name] = np.linalg.norm(sim_gripper - real_gripper)
        print(f"{name}: sim={sim_gripper}, real={real_gripper}, error={errors[name]:.4f}m")

    return errors
```

---

## Tips for Successful Sim-to-Real

1. **Start with position control** — Match what the real servos do (PD position control)
2. **Use action smoothing** — Low-pass filter actions to avoid jerky real-robot motion
3. **Add observation noise** — Train with Gaussian noise on joint readings (σ ≈ 0.01 rad)
4. **Limit action frequency** — Match real control rate (~50 Hz) in simulation during fine-tuning
5. **Conservative force limits** — Use forcerange slightly below real servo capability
6. **Test incrementally** — Verify each joint moves correctly before running full policy
