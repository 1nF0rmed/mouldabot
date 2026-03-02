# Vision for Robotics

Provides expert guidance on camera setup in MuJoCo, rendering, image-based observations for RL, object detection from simulation cameras, visual servoing, and video recording.

---

## Activation

- **Auto-invoked** when the user asks about: cameras, rendering, images, pixels, vision, visual observations, video recording, offscreen rendering, object detection, visual servoing, or depth images in MuJoCo.
- **User-invocable** via `/robot-vision`

---

## Project Context

The project has one camera defined in `scene_blocks.xml`:

```xml
<camera name="top_down" pos="0.2 0 0.8" xyaxes="1 0 0 0 1 0" fovy="60"/>
```

This is a fixed top-down camera centered above the table, useful for observing block positions.

---

## Camera Setup in MJCF

### Fixed Camera (World-Mounted)

```xml
<!-- Top-down view (existing in this project) -->
<camera name="top_down" pos="0.2 0 0.8" xyaxes="1 0 0 0 1 0" fovy="60"/>

<!-- Angled side view -->
<camera name="side_view" pos="0.6 -0.3 0.5" xyaxes="0.5 0.866 0 -0.289 0.167 0.943" fovy="45"/>

<!-- Front view of workspace -->
<camera name="front" pos="0.2 -0.5 0.4" lookat="0.2 0 0.22" fovy="50"/>
```

### Body-Mounted Camera (Eye-in-Hand)

Mount a camera on the robot's gripper body for eye-in-hand visual servoing:

```xml
<!-- Inside the gripper body in so101_new_calib.xml -->
<body name="gripper">
  <!-- existing geoms and joints -->
  <camera name="wrist_cam" pos="0 -0.02 -0.05" xyaxes="1 0 0 0 0 -1"
          fovy="60" resolution="640 480"/>
</body>
```

To add this programmatically via MjSpec:

```python
spec = mujoco.MjSpec.from_file("SO101/scene_blocks.xml")
model = spec.compile()

# Find gripper body and add camera
gripper_body = spec.find_body("gripper")
cam = gripper_body.add_camera(
    name="wrist_cam",
    pos=[0, -0.02, -0.05],
    xyaxes=[1, 0, 0, 0, 0, -1],
    fovy=60,
)
model = spec.compile()
```

### Camera Parameters

| Attribute | Description | Default |
|-----------|-------------|---------|
| `pos` | Position in parent frame | [0,0,0] |
| `quat` | Orientation quaternion | [1,0,0,0] |
| `xyaxes` | Camera x and y axes (alternative to quat) | — |
| `fovy` | Vertical field of view (degrees) | 45 |
| `lookat` | Point camera looks at (convenience) | — |
| `resolution` | Image width × height | — |
| `ipd` | Interpupillary distance (stereo) | 0.068 |

---

## Offscreen Rendering (Headless)

### Basic RGB Rendering

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("SO101/scene_blocks.xml")
data = mujoco.MjData(model)

# Create renderer
width, height = 640, 480
renderer = mujoco.Renderer(model, height=height, width=width)

# Step simulation, then render
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="top_down")
image = renderer.render()  # numpy array (H, W, 3), uint8

# Save image
from PIL import Image
Image.fromarray(image).save("frame.png")
```

### Depth Rendering

```python
renderer.enable_depth_rendering(True)
renderer.update_scene(data, camera="top_down")
depth = renderer.render()  # numpy array (H, W), float32

# Convert to metric depth
extent = model.stat.extent
near = model.vis.map.znear * extent
far = model.vis.map.zfar * extent
metric_depth = near / (1 - depth * (1 - near / far))

renderer.enable_depth_rendering(False)  # switch back to RGB
```

### Segmentation Rendering

```python
renderer.enable_segmentation_rendering(True)
renderer.update_scene(data, camera="top_down")
seg = renderer.render()  # (H, W, 2) — [geom_id, obj_type]

# Find pixels belonging to red_block
red_block_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_block")
red_mask = seg[:, :, 0] == red_block_geom_id

renderer.enable_segmentation_rendering(False)
```

---

## Image-Based Observations for RL

### Pixel Observation Environment

```python
class SO101VisionEnv(gym.Env):
    def __init__(self, image_size=84, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("SO101/scene_blocks.xml")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self.image_size = image_size

        # Image observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "joints": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })

        self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="top_down")
        image = self.renderer.render()

        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                              for j in self.joint_ids], dtype=np.float32)

        return {"image": image, "joints": joint_pos}
```

### Frame Stacking for Temporal Info

```python
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

env = SO101VisionEnv(image_size=84)
env = GrayScaleObservation(env, keep_dim=True)
env = FrameStack(env, num_stack=4)  # stack 4 frames → (4, 84, 84, 1)
```

### CNN Policy with Stable-Baselines3

```python
from stable_baselines3 import PPO

model = PPO(
    "CnnPolicy",  # uses CNN feature extractor
    env,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=4,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
```

---

## Object Detection from Simulation

### Using Segmentation for Ground-Truth Detection

```python
def detect_blocks(model, data, renderer, camera="top_down"):
    """Get block positions in image using segmentation rendering."""
    renderer.enable_segmentation_rendering(True)
    renderer.update_scene(data, camera=camera)
    seg = renderer.render()
    renderer.enable_segmentation_rendering(False)

    detections = {}
    for block_name in ["red_block", "green_block", "blue_block"]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
        if geom_id < 0:
            continue
        mask = seg[:, :, 0] == geom_id
        if mask.any():
            ys, xs = np.where(mask)
            detections[block_name] = {
                "centroid": (np.mean(xs), np.mean(ys)),
                "bbox": (xs.min(), ys.min(), xs.max(), ys.max()),
                "area": mask.sum(),
            }
    return detections
```

### World-to-Pixel Projection

```python
def world_to_pixel(model, data, world_pos, camera_name, width=640, height=480):
    """Project a 3D world point to 2D pixel coordinates."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # Camera position and rotation from data
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

    # Point in camera frame
    p_cam = cam_rot.T @ (world_pos - cam_pos)

    # Perspective projection
    fovy = model.cam_fovy[cam_id]
    f = height / (2 * np.tan(np.radians(fovy) / 2))

    u = f * p_cam[0] / (-p_cam[2]) + width / 2
    v = f * (-p_cam[1]) / (-p_cam[2]) + height / 2

    return int(u), int(v)
```

---

## Visual Servoing

### Image-Based Visual Servoing (IBVS)

```python
def ibvs_step(model, data, target_pixel, current_pixel, camera_name,
              gain=0.5, site_name="gripperframe"):
    """Compute joint velocity from pixel error using image Jacobian."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_forward(model, data)

    # Pixel error
    error = np.array(target_pixel) - np.array(current_pixel)

    # Approximate: use world-space Jacobian projected through camera
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    # Simplified: move end-effector toward target in camera plane
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

    # Convert pixel error to world-space direction (approximate)
    fovy = model.cam_fovy[cam_id]
    depth = -cam_rot.T @ (data.site_xpos[site_id] - data.cam_xpos[cam_id])
    z = depth[2]
    f = 480 / (2 * np.tan(np.radians(fovy) / 2))

    dx_world = cam_rot @ np.array([error[0] * z / f, -error[1] * z / f, 0])

    # Damped least squares IK for velocity
    JJT = jacp @ jacp.T + 1e-4 * np.eye(3)
    dq = jacp.T @ np.linalg.solve(JJT, gain * dx_world)

    return dq  # joint velocity command
```

---

## Video Recording

### Record from Simulation

```python
import mediapy  # pip install mediapy

def record_trajectory(model, data, trajectory, camera="top_down",
                      width=640, height=480, fps=50):
    """Record video of trajectory execution."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []

    for joint_dict in trajectory:
        # Apply joints
        for name, val in joint_dict.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = val
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            data.ctrl[aid] = val

        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

    mediapy.write_video("trajectory.mp4", frames, fps=fps)
    print(f"Saved {len(frames)} frames to trajectory.mp4")
    return frames
```

### Record with Multiple Camera Views

```python
def record_multi_camera(model, data, trajectory, cameras=["top_down", "side_view"],
                         width=320, height=240, fps=50):
    """Record side-by-side multi-camera video."""
    renderers = {cam: mujoco.Renderer(model, height=height, width=width) for cam in cameras}
    frames = []

    for joint_dict in trajectory:
        for name, val in joint_dict.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = val
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            data.ctrl[aid] = val

        mujoco.mj_step(model, data)

        row = []
        for cam in cameras:
            renderers[cam].update_scene(data, camera=cam)
            row.append(renderers[cam].render())

        frames.append(np.concatenate(row, axis=1))

    mediapy.write_video("multi_cam.mp4", frames, fps=fps)
```

### Using OpenCV for Video

```python
import cv2

def record_opencv(model, data, trajectory, camera="top_down", width=640, height=480):
    renderer = mujoco.Renderer(model, height=height, width=width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, 50, (width, height))

    for joint_dict in trajectory:
        # ... apply joints, step ...
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
```

---

## Tips

- **Camera naming:** Always use `name=` so you can reference by string in `renderer.update_scene(data, camera="name")`
- **Performance:** Create `Renderer` once and reuse; creating/destroying is expensive
- **Multiple renders per step:** You can call `update_scene` + `render` multiple times (RGB, depth, segmentation) per physics step
- **Resolution vs speed:** For RL training, use small images (64×64 or 84×84). For videos, use 640×480+
- **Lighting matters:** Add multiple lights with ambient for realistic images. The project uses a single directional light
