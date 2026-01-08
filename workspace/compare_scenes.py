#!/usr/bin/env python3
"""Compare scene loading between official viewer and our recorder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from src.viewer_wrapper import create_default_sim_settings

print("=" * 70)
print("OFFICIAL VIEWER SETTINGS")
print("=" * 70)

# Official viewer settings (like command line)
official_settings = default_sim_settings.copy()
official_settings["scene"] = "Baked_sc0_staging_01"
official_settings["scene_dataset_config_file"] = "/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"
official_settings["window_width"] = 640
official_settings["window_height"] = 480
official_settings["default_agent"] = 0
official_settings["enable_physics"] = False
official_settings["default_agent_navmesh"] = False

print(f"Scene: {official_settings['scene']}")
print(f"Dataset: {official_settings['scene_dataset_config_file']}")
print(f"Window: {official_settings['window_width']}x{official_settings['window_height']}")
print(f"Width/Height (render): {official_settings.get('width')}/{official_settings.get('height')}")

cfg_official = make_cfg(official_settings)
print(f"\nAgent[0] Scene: {cfg_official.sim_cfg.scene_id}")
if cfg_official.agents:
    sensor = cfg_official.agents[0].sensor_specifications[0]
    print(f"Sensor Resolution: {sensor.resolution}")

print("\n" + "=" * 70)
print("OUR RECORDER SETTINGS")
print("=" * 70)

# Our recorder settings
our_settings = create_default_sim_settings(
    scene_path="Baked_sc0_staging_01",
    dataset_path="/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json",
    width=640,
    height=480,
    enable_hbao=True
)

print(f"Scene: {our_settings['scene']}")
print(f"Dataset: {our_settings['scene_dataset_config_file']}")
print(f"Window: {our_settings['window_width']}x{our_settings['window_height']}")
print(f"Width/Height (render): {our_settings.get('width')}/{our_settings.get('height')}")

cfg_ours = make_cfg(our_settings)
print(f"\nAgent[0] Scene: {cfg_ours.sim_cfg.scene_id}")
if cfg_ours.agents:
    sensor = cfg_ours.agents[0].sensor_specifications[0]
    print(f"Sensor Resolution: {sensor.resolution}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"Scene ID Match: {cfg_official.sim_cfg.scene_id == cfg_ours.sim_cfg.scene_id}")
print(f"Official Scene ID: {cfg_official.sim_cfg.scene_id}")
print(f"Our Scene ID: {cfg_ours.sim_cfg.scene_id}")
print(f"\nScene ID Match: {cfg_official.sim_cfg.scene_id == cfg_ours.sim_cfg.scene_id}")

# Additional config comparison
print("\n" + "=" * 70)
print("DETAILED CONFIG COMPARISON")
print("=" * 70)

print(f"\nOfficial:")
print(f"  scene_id: {cfg_official.sim_cfg.scene_id}")
print(f"  enable_physics: {cfg_official.sim_cfg.enable_physics}")
print(f"  requires_textures: {cfg_official.sim_cfg.requires_textures}")
print(f"  scene_dataset_config_file: {cfg_official.sim_cfg.scene_dataset_config_file}")

print(f"\nOurs:")
print(f"  scene_id: {cfg_ours.sim_cfg.scene_id}")
print(f"  enable_physics: {cfg_ours.sim_cfg.enable_physics}")
print(f"  requires_textures: {cfg_ours.sim_cfg.requires_textures}")
print(f"  scene_dataset_config_file: {cfg_ours.sim_cfg.scene_dataset_config_file}")

# Compare agent configurations
print("\n" + "=" * 70)
print("AGENT CONFIGURATION")
print("=" * 70)

if cfg_official.agents and cfg_ours.agents:
    print(f"\nOfficial Agent State Sensors: {len(cfg_official.agents[0].sensor_specifications)}")
    for i, sensor in enumerate(cfg_official.agents[0].sensor_specifications):
        print(f"  Sensor {i}: {sensor.uuid} - {sensor.sensor_type} - Resolution: {sensor.resolution}")

    print(f"\nOur Agent State Sensors: {len(cfg_ours.agents[0].sensor_specifications)}")
    for i, sensor in enumerate(cfg_ours.agents[0].sensor_specifications):
        print(f"  Sensor {i}: {sensor.uuid} - {sensor.sensor_type} - Resolution: {sensor.resolution}")
