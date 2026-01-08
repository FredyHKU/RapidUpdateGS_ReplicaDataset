#!/usr/bin/env python3
"""Debug script to check sensor configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from src.viewer_wrapper import create_default_sim_settings
from habitat_sim.utils.settings import make_cfg

# Load config
with open("config/default.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get settings
scene_config = config.get("scene", {})
trajectory_settings = config.get("trajectory", {})
rendering_settings = config.get("rendering", {})

dataset_path = scene_config["dataset"]
scene_path = scene_config["name"]
window_size = trajectory_settings.get("interactive_window_size", [680, 1200])
width = window_size[1]
height = window_size[0]
enable_hbao = rendering_settings.get("enable_hbao", False)

# Create sim_settings
sim_settings = create_default_sim_settings(
    scene_path=scene_path,
    dataset_path=dataset_path,
    width=width,
    height=height,
    enable_hbao=enable_hbao
)

print("=" * 70)
print("SIM SETTINGS:")
print("=" * 70)
for key in sorted(sim_settings.keys()):
    if sim_settings[key] is not None:
        print(f"{key}: {sim_settings[key]}")

# Create cfg
cfg = make_cfg(sim_settings)

print("\n" + "=" * 70)
print("AGENT[0] SENSOR SPECIFICATIONS:")
print("=" * 70)

if cfg.agents:
    for i, spec in enumerate(cfg.agents[0].sensor_specifications):
        print(f"\nSensor {i}:")
        print(f"  UUID: {spec.uuid}")
        print(f"  Sensor Type: {spec.sensor_type}")
        print(f"  Sensor Subtype: {spec.sensor_subtype}")
        print(f"  Resolution: {spec.resolution}")
        print(f"  Position: {spec.position}")
        print(f"  Orientation: {spec.orientation}")

        # Check if it's a camera sensor
        if hasattr(spec, "hfov"):
            print(f"  HFoV: {spec.hfov}")
        if hasattr(spec, "near"):
            print(f"  Near: {spec.near}")
        if hasattr(spec, "far"):
            print(f"  Far: {spec.far}")
else:
    print("No agents configured!")
