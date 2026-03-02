# Reinforcement Learning for Robotics

Provides expert guidance on building RL environments for MuJoCo robot manipulation — Gymnasium wrappers, observation/action spaces, reward shaping, training with Stable-Baselines3, and GPU-accelerated training with MJX.

---

## Activation

- **Auto-invoked** when the user asks about: reinforcement learning, RL training, Gymnasium environments, reward functions, PPO, SAC, policy training, observation space, action space, Stable-Baselines3, MJX, or learning to grasp/pick-and-place.
- **User-invocable** via `/robot-rl`

---

## Project Context

This project has an SO-101 arm with 3 colored blocks on a table. The current control is scripted (pre-computed grasp configs + interpolation). RL can learn adaptive grasping policies that generalize to new block positions.

### Key Details for Environment Design

- **Joints:** 6 (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`)
- **Actuators:** 6 position actuators (ctrl = target joint angle)
- **Joint ranges:** See `so101_new_calib.xml` (all have limits)
- **Objects:** `red_block`, `green_block`, `blue_block` — freejoint (7 qpos each: pos + quat)
- **Sites:** `gripperframe` (gripper center), `fixed_jaw_tip`, `moving_jaw_tip`
- **Scene:** `SO101/scene_blocks.xml`

---

## Gymnasium Environment Wrapper

### Basic Structure

```python
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class SO101PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, target_block="red_block"):
        super().__init__()
        self.render_mode = render_mode
        self.target_block = target_block

        # Load model
        self.model = mujoco.MjModel.from_xml_path("SO101/scene_blocks.xml")
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                            "wrist_flex", "wrist_roll", "gripper"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in self.joint_names]
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                 target_block)
        self.gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                                   "gripperframe")

        # Action space: delta joint angles (normalized to [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.max_delta = 0.05  # max radians per step

        # Observation space
        # [joint_pos(6), joint_vel(6), gripper_pos(3), block_pos(3), block_to_gripper(3)]
        obs_dim = 6 + 6 + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.max_steps = 500
        self.step_count = 0
        self.n_substeps = 10  # physics substeps per action

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)

        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                              for j in self.joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[j]]
                              for j in self.joint_ids])
        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
        block_pos = self.data.xpos[self.block_body_id].copy()
        relative = block_pos - gripper_pos

        return np.concatenate([joint_pos, joint_vel, gripper_pos, block_pos, relative]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize block position on table
        block_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"{self.target_block}_joint")
        adr = self.model.jnt_qposadr[block_jnt_id]
        self.data.qpos[adr + 0] = self.np_random.uniform(0.05, 0.35)   # x on table
        self.data.qpos[adr + 1] = self.np_random.uniform(-0.15, 0.15)  # y on table
        self.data.qpos[adr + 2] = 0.23                                  # z on table

        # Set arm to home position
        home = {"shoulder_pan": 0, "shoulder_lift": -1.75, "elbow_flex": 1.69,
                "wrist_flex": 1.08, "wrist_roll": 0, "gripper": 0}
        for name, val in home.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[jid]] = val
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.data.ctrl[aid] = val

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Scale action and apply as delta to current ctrl
        delta = action * self.max_delta
        for i, aid in enumerate(self.actuator_ids):
            jid = self.joint_ids[i]
            new_ctrl = self.data.ctrl[aid] + delta[i]
            new_ctrl = np.clip(new_ctrl,
                               self.model.jnt_range[jid, 0],
                               self.model.jnt_range[jid, 1])
            self.data.ctrl[aid] = new_ctrl

        # Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self):
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        block_pos = self.data.xpos[self.block_body_id]

        # Distance reward (negative distance)
        dist = np.linalg.norm(gripper_pos - block_pos)
        reach_reward = -dist

        # Height reward (block lifted above table)
        table_height = 0.23
        lift_reward = max(0, block_pos[2] - table_height) * 10

        # Grasp bonus (block near gripper AND lifted)
        grasp_bonus = 0
        if dist < 0.03 and block_pos[2] > table_height + 0.02:
            grasp_bonus = 5.0

        return reach_reward + lift_reward + grasp_bonus

    def _check_success(self):
        block_pos = self.data.xpos[self.block_body_id]
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        dist = np.linalg.norm(gripper_pos - block_pos)
        return dist < 0.03 and block_pos[2] > 0.30  # lifted 7cm above table
```

---

## Observation Space Design

### Minimal (Joint-Only)

```python
# 12 dims: joint positions + velocities
obs = np.concatenate([joint_pos, joint_vel])
```

### Standard (Joint + Object)

```python
# 21 dims: joints + gripper Cartesian + block Cartesian + relative vector
obs = np.concatenate([joint_pos, joint_vel, gripper_pos, block_pos, relative])
```

### Rich (With Contacts and Gripper State)

```python
# Add gripper opening width, contact forces, block orientation
gripper_width = data.qpos[model.jnt_qposadr[gripper_jnt_id]]
jaw_dist = np.linalg.norm(data.site_xpos[fixed_jaw_id] - data.site_xpos[moving_jaw_id])
```

### Image-Based

See the `/robot-vision` skill for pixel observation setup.

---

## Action Space Design

### Delta Joint Angles (Recommended for Stability)

```python
action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
# Scale by max_delta (e.g., 0.05 rad/step) and add to current ctrl
```

### Absolute Joint Angles

```python
# Action directly sets target joint angles
action_space = spaces.Box(
    low=np.array([model.jnt_range[j, 0] for j in joint_ids]),
    high=np.array([model.jnt_range[j, 1] for j in joint_ids]),
    dtype=np.float32
)
```

### Cartesian (Task-Space) Actions

```python
# 4D: [dx, dy, dz, gripper_open]
action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
# Requires IK to convert to joint commands
```

---

## Reward Shaping for Pick-and-Place

### Phased Reward (Reach → Grasp → Lift → Place)

```python
def compute_phased_reward(self):
    gripper = self.data.site_xpos[self.gripper_site_id]
    block = self.data.xpos[self.block_body_id]
    target_place = np.array([0.3, 0.1, 0.25])  # target placement position

    dist_to_block = np.linalg.norm(gripper - block)
    block_height = block[2]
    dist_to_target = np.linalg.norm(block - target_place)

    reward = 0.0

    # Phase 1: Reach — reward for approaching block
    reward += -1.0 * dist_to_block

    # Phase 2: Grasp — bonus when close to block
    if dist_to_block < 0.03:
        reward += 2.0

    # Phase 3: Lift — reward for lifting block
    if dist_to_block < 0.03:
        reward += 5.0 * max(0, block_height - 0.23)

    # Phase 4: Place — reward for moving block to target
    if block_height > 0.28:
        reward += -2.0 * dist_to_target
        if dist_to_target < 0.03:
            reward += 50.0  # success bonus

    # Penalties
    reward -= 0.01  # time penalty
    reward -= 0.1 * np.sum(np.abs(self.data.ctrl))  # energy penalty

    return reward
```

---

## Training with Stable-Baselines3

### PPO (Reliable Default)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

def make_env(rank):
    def _init():
        env = SO101PickPlaceEnv(target_block="red_block")
        env.reset(seed=rank)
        return env
    return _init

# Parallel environments
n_envs = 8
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tb_logs/",
)

eval_env = SO101PickPlaceEnv(target_block="red_block")
eval_callback = EvalCallback(eval_env, eval_freq=10000, n_eval_episodes=10)

model.learn(total_timesteps=2_000_000, callback=eval_callback)
model.save("ppo_so101_pick")
```

### SAC (Better for Continuous Control)

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    tensorboard_log="./tb_logs/",
)

model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save("sac_so101_pick")
```

### Hyperparameter Tips for Manipulation

| Parameter | PPO | SAC | Notes |
|-----------|-----|-----|-------|
| learning_rate | 3e-4 | 3e-4 | Reduce if unstable |
| gamma | 0.99 | 0.99 | Increase for long-horizon tasks |
| batch_size | 256 | 256 | Larger = more stable |
| n_envs | 8-16 | 1 | PPO benefits from parallel; SAC is off-policy |
| network | [256, 256] | [256, 256] | 2-layer MLP sufficient for joint-space |

---

## GPU-Accelerated Training with MJX

MJX is MuJoCo's JAX backend for massively parallel simulation on GPU:

```python
import jax
from mujoco import mjx

# Load model and create MJX model
mj_model = mujoco.MjModel.from_xml_path("SO101/scene_blocks.xml")
mjx_model = mjx.put_model(mj_model)

# Batch reset
rng = jax.random.PRNGKey(0)
batch_size = 4096

@jax.jit
@jax.vmap
def batch_step(mjx_data, action):
    mjx_data = mjx_data.replace(ctrl=action)
    return mjx.step(mjx_model, mjx_data)

# Initialize batch of data
mjx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))
mjx_data = jax.tree.map(lambda x: jnp.stack([x] * batch_size), mjx_data)
```

> **Note:** MJX requires JAX and a GPU. Not all MuJoCo features are supported (check mesh collision support). Best for simple scenes with primitive geoms.

---

## Evaluation and Deployment

### Evaluate Trained Policy

```python
model = PPO.load("ppo_so101_pick")
env = SO101PickPlaceEnv(render_mode="human")

for episode in range(10):
    obs, _ = env.reset()
    total_reward = 0
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode {episode}: reward={total_reward:.2f}, steps={step+1}")
```

### Convert Policy to Joint Trajectory

```python
# Run policy and record joint commands for replay in sim.py
obs, _ = env.reset()
trajectory = []
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, _ = env.step(action)
    joint_dict = {name: float(env.data.ctrl[aid])
                  for name, aid in zip(env.joint_names, env.actuator_ids)}
    trajectory.append(joint_dict)
    if done or trunc:
        break
# trajectory can be fed to sim.py's trajectory execution loop
```

---

## Environment Registration

```python
gym.register(
    id="SO101PickPlace-v0",
    entry_point="envs.so101_pick_place:SO101PickPlaceEnv",
    max_episode_steps=500,
)

env = gym.make("SO101PickPlace-v0", target_block="red_block")
```
