#!/usr/bin/env python3
"""Compare ALL simulation settings between official viewer and our recorder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from habitat_sim.utils.settings import default_sim_settings
from src.viewer_wrapper import create_default_sim_settings

print("=" * 70)
print("OFFICIAL VIEWER SETTINGS (all keys)")
print("=" * 70)

# Official viewer settings (like command line)
official_settings = default_sim_settings.copy()
official_settings["scene"] = "Baked_sc0_staging_01"
official_settings["scene_dataset_config_file"] = "/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"
official_settings["window_width"] = 640
official_settings["window_height"] = 480
official_settings["width"] = 640
official_settings["height"] = 480
official_settings["default_agent"] = 0
official_settings["enable_physics"] = False
official_settings["default_agent_navmesh"] = False

for key in sorted(official_settings.keys()):
    print(f"{key:40} = {official_settings[key]}")

print("\n" + "=" * 70)
print("OUR RECORDER SETTINGS (all keys)")
print("=" * 70)

# Our recorder settings
our_settings = create_default_sim_settings(
    scene_path="Baked_sc0_staging_01",
    dataset_path="/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json",
    width=640,
    height=480,
    enable_hbao=True
)

for key in sorted(our_settings.keys()):
    print(f"{key:40} = {our_settings[key]}")

print("\n" + "=" * 70)
print("DIFFERENCES")
print("=" * 70)

all_keys = set(official_settings.keys()) | set(our_settings.keys())
differences = []

for key in sorted(all_keys):
    official_val = official_settings.get(key)
    our_val = our_settings.get(key)

    if official_val != our_val:
        differences.append((key, official_val, our_val))
        print(f"\n{key}:")
        print(f"  Official: {official_val}")
        print(f"  Ours:     {our_val}")

if not differences:
    print("\nNo differences found!")
else:
    print(f"\n\nTotal differences: {len(differences)}")
