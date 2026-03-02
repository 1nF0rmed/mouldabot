#!/usr/bin/env python3
"""SO-101 MuJoCo simulation: arm + table + 3 colored blocks."""

import threading
import urllib.request
from pathlib import Path

import mujoco
import mujoco.viewer

REPO_RAW = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"
LOCAL_DIR = Path(__file__).resolve().parent / "SO101"

XML_FILES = [
    "scene.xml",
    "so101_new_calib.xml",
    "joints_properties.xml",
]

STL_FILES = [
    "base_so101_v2.stl",
    "base_motor_holder_so101_v1.stl",
    "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl",
    "sts3215_03a_v1.stl",
    "sts3215_03a_no_horn_v1.stl",
    "upper_arm_so101_v1.stl",
    "under_arm_so101_v1.stl",
    "rotation_pitch_so101_v1.stl",
    "wrist_roll_pitch_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl",
    "moving_jaw_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl",
]


def download_assets():
    """Download SO-101 MJCF + mesh files if not already present."""
    assets_dir = LOCAL_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    for fname in XML_FILES:
        dest = LOCAL_DIR / fname
        if not dest.exists():
            url = f"{REPO_RAW}/{fname}"
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)

    for fname in STL_FILES:
        dest = assets_dir / fname
        if not dest.exists():
            url = f"{REPO_RAW}/assets/{fname}"
            print(f"Downloading assets/{fname} ...")
            urllib.request.urlretrieve(url, dest)


SCENE_BLOCKS_XML = """\
<mujoco model="scene_blocks">
  <include file="so101_new_calib.xml" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="160" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2" />
    <material name="table_mat" rgba="0.45 0.30 0.18 1" />
    <material name="red_block"   rgba="0.9 0.15 0.15 1" />
    <material name="green_block" rgba="0.15 0.8 0.15 1" />
    <material name="blue_block"  rgba="0.15 0.15 0.9 1" />
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
    <camera name="top_down" pos="0.2 0 0.8" xyaxes="1 0 0 0 1 0" fovy="60" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />

    <!-- Table: 60x40cm surface, 20cm tall -->
    <body name="table" pos="0.2 0 0.1">
      <geom name="tabletop" type="box" size="0.3 0.2 0.01"
            pos="0 0 0.1" material="table_mat" />
      <geom name="leg1" type="cylinder" size="0.015 0.1"
            pos=" 0.27  0.17 0" material="table_mat" />
      <geom name="leg2" type="cylinder" size="0.015 0.1"
            pos="-0.27  0.17 0" material="table_mat" />
      <geom name="leg3" type="cylinder" size="0.015 0.1"
            pos=" 0.27 -0.17 0" material="table_mat" />
      <geom name="leg4" type="cylinder" size="0.015 0.1"
            pos="-0.27 -0.17 0" material="table_mat" />
    </body>

    <!-- Block 1: red -->
    <body name="red_block" pos="0.1 0.0 0.23">
      <freejoint />
      <geom name="red_block" type="box" size="0.02 0.02 0.02"
            material="red_block" mass="0.05" />
    </body>

    <!-- Block 2: green -->
    <body name="green_block" pos="0.2 0.05 0.23">
      <freejoint />
      <geom name="green_block" type="box" size="0.02 0.02 0.02"
            material="green_block" mass="0.05" />
    </body>

    <!-- Block 3: blue -->
    <body name="blue_block" pos="0.3 0.0 0.23">
      <freejoint />
      <geom name="blue_block" type="box" size="0.02 0.02 0.02"
            material="blue_block" mass="0.05" />
    </body>
  </worldbody>
</mujoco>
"""


ARM_POS = [0.2, -0.18, 0.21]
ARM_QUAT = [0.707107, 0, 0, 0.707107]

# Home position: arm folded upright, ready to reach the table
HOME_JOINTS = {
    "shoulder_pan": 0,
    "shoulder_lift": -1.75,
    "elbow_flex": 1.69,
    "wrist_flex": 1.08,
    "wrist_roll": 0,
    "gripper": 0,
}

# Raised transit configuration: arm above table height so pan rotation clears blocks.
# shoulder_pan and gripper are overridden dynamically per-move.
RAISED_JOINTS = {
    "shoulder_pan": 0,
    "shoulder_lift": -1.0,
    "elbow_flex": 1.2,
    "wrist_flex": 0.8,
    "wrist_roll": 0,
    "gripper": 0,
}


# Grasp positions for each block — fill in after manual tuning
GRASP_CONFIGS = {
    "red_block": {
        "shoulder_pan": -0.4992,
        "shoulder_lift": -0.0173,
        "elbow_flex": 0.4226,
        "wrist_flex": 1.3264,
        "wrist_roll": 1.1802,
        "gripper": 0.5262,
    },
    "green_block": {
        "shoulder_pan": -0.0,
        "shoulder_lift": 0.1921,
        "elbow_flex": 0.1522,
        "wrist_flex": 1.5088,
        "wrist_roll": 0.0,
        "gripper": 0.4782,
    },
    "blue_block": {
        "shoulder_pan": 0.4992,
        "shoulder_lift": -0.0173,
        "elbow_flex": 0.4226,
        "wrist_flex": 1.3264,
        "wrist_roll": -1.1802,
        "gripper": 0.5262,
    },
}

JOINT_NAMES = list(HOME_JOINTS.keys())


def get_current_joints(model, data):
    """Read current joint positions from data.qpos for all joints."""
    result = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        result[name] = data.qpos[model.jnt_qposadr[jid]]
    return result


def make_motion_steps(current, target, steps=200):
    """Linear interpolation from current to target joint dicts."""
    trajectory = []
    for i in range(steps):
        t = (i + 1) / steps
        step = {}
        for name in JOINT_NAMES:
            step[name] = current[name] + t * (target[name] - current[name])
        trajectory.append(step)
    return trajectory


def make_pre_grasp(grasp):
    """Derive an approach-from-above config from a grasp config.

    Pulls shoulder_lift back ~0.3 rad and tucks elbow_flex +0.15 rad so the
    final descent to the block is roughly vertical.
    """
    pre = dict(grasp)
    pre["shoulder_lift"] = grasp["shoulder_lift"] - 0.3
    pre["elbow_flex"] = grasp["elbow_flex"] + 0.15
    return pre


def is_arm_raised(joints):
    """Return True if the arm is already above table height (e.g. at HOME)."""
    return joints["shoulder_lift"] < -0.6


def make_safe_trajectory(current, target, target_name="home", steps_per_seg=80):
    """Build a multi-segment trajectory that lifts above the table before panning.

    Phases:
      1. Lift  — current -> raised (preserve current shoulder_pan & gripper)
      2. Pan   — raised (current pan) -> raised (target pan & wrist_roll)
      3. Lower — raised -> pre-grasp (block targets) or -> target (home)
      4. Descend — pre-grasp -> target (block targets only)
    """
    gripper_val = current["gripper"]
    trajectory = []

    # --- Phase 1: Lift (skip if arm is already raised) ---
    if not is_arm_raised(current):
        raised_start = dict(RAISED_JOINTS)
        raised_start["shoulder_pan"] = current["shoulder_pan"]
        raised_start["wrist_roll"] = current["wrist_roll"]
        raised_start["gripper"] = gripper_val
        trajectory.extend(make_motion_steps(current, raised_start, steps=steps_per_seg))
        phase2_start = raised_start
    else:
        phase2_start = dict(current)

    # --- Phase 2: Pan to target shoulder_pan while raised ---
    raised_end = dict(RAISED_JOINTS)
    raised_end["shoulder_pan"] = target["shoulder_pan"]
    raised_end["wrist_roll"] = target["wrist_roll"]
    raised_end["gripper"] = gripper_val
    trajectory.extend(make_motion_steps(phase2_start, raised_end, steps=steps_per_seg))

    # --- Phase 3 & 4: Lower to target ---
    is_block_target = target_name and target_name != "home"
    if is_block_target:
        pre_grasp = make_pre_grasp(target)
        pre_grasp["gripper"] = gripper_val
        trajectory.extend(make_motion_steps(raised_end, pre_grasp, steps=steps_per_seg))
        # Phase 4: final descent
        trajectory.extend(make_motion_steps(pre_grasp, target, steps=steps_per_seg))
    else:
        trajectory.extend(make_motion_steps(raised_end, target, steps=steps_per_seg))

    return trajectory


def generate_scene():
    """Write scene_blocks.xml into the SO101 directory."""
    scene_path = LOCAL_DIR / "scene_blocks.xml"
    scene_path.write_text(SCENE_BLOCKS_XML)
    return scene_path


def load_model(scene_path):
    """Load scene, position the arm on the table, and set home keyframe."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    base = spec.body("base")
    base.pos = ARM_POS
    base.quat = ARM_QUAT

    # Add a "home" keyframe so the sim starts in the rest pose
    model = spec.compile()
    key = spec.add_key()
    key.name = "home"
    data = mujoco.MjData(model)
    for name, val in HOME_JOINTS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = val
    key.qpos = data.qpos
    key.ctrl = [HOME_JOINTS[model.actuator(i).name] for i in range(model.nu)]

    return spec.compile()


def apply_joints(model, data, joint_dict):
    """Set qpos and ctrl for the given joint dictionary."""
    for name, val in joint_dict.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = val
    for i in range(model.nu):
        aname = model.actuator(i).name
        if aname in joint_dict:
            data.ctrl[i] = joint_dict[aname]


def command_loop(pending_cmd):
    """Read stdin commands and queue them for the sim loop."""
    print("Commands:  pos  |  final <block_name>  |  home  |  quit")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "pos":
            pending_cmd.append(("pos", None, None))
        elif cmd == "final" and len(parts) == 2:
            block = parts[1]
            # Allow shorthand: "red" -> "red_block"
            if not block.endswith("_block"):
                block = f"{block}_block"
            cfg = GRASP_CONFIGS.get(block)
            if cfg is None:
                print(f"No grasp config for '{block}' (not tuned yet)")
            else:
                pending_cmd.append(("final", block, cfg))
        elif cmd == "home":
            pending_cmd.append(("home", None, HOME_JOINTS))
        elif cmd == "quit":
            pending_cmd.append(("quit", None, None))
            break
        else:
            print("Unknown command. Use: pos | final <block_name> | home | quit")


def main():
    download_assets()
    scene_path = generate_scene()
    print(f"Loading {scene_path}")
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    pending_cmd = []
    motion_queue = []
    cmd_thread = threading.Thread(
        target=command_loop, args=(pending_cmd,), daemon=True
    )
    cmd_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Process pending commands
            while pending_cmd:
                action, block, joints = pending_cmd.pop(0)
                if action == "quit":
                    viewer.close()
                    return
                if action == "pos":
                    angles = {k: round(v, 4) for k, v in get_current_joints(model, data).items()}
                    print(f"joints: {angles}")
                else:
                    # Interrupt any in-progress motion and start fresh
                    current = get_current_joints(model, data)
                    motion_queue.clear()
                    label = block if block else "home"
                    motion_queue.extend(
                        make_safe_trajectory(current, joints, target_name=label)
                    )
                    print(f"Moving to '{label}' ({len(motion_queue)} steps)")

            # Apply next motion step if queued
            if motion_queue:
                step = motion_queue.pop(0)
                for i in range(model.nu):
                    aname = model.actuator(i).name
                    if aname in step:
                        data.ctrl[i] = step[aname]

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
