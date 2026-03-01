#!/usr/bin/env python3
"""SO-101 MuJoCo simulation: arm + table + 3 colored blocks."""

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


def generate_scene():
    """Write scene_blocks.xml into the SO101 directory."""
    scene_path = LOCAL_DIR / "scene_blocks.xml"
    scene_path.write_text(SCENE_BLOCKS_XML)
    return scene_path


def load_model(scene_path):
    """Load scene and position the arm on the table via MjSpec."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    base = spec.body("base")
    base.pos = ARM_POS
    base.quat = ARM_QUAT
    return spec.compile()


def main():
    download_assets()
    scene_path = generate_scene()
    print(f"Loading {scene_path}")
    model = load_model(scene_path)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
