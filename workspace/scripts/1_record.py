#!/usr/bin/env python3
"""
Interactive keyframe recording script.

This script serves as the main entry point for recording keyframes interactively
using an X11 window. It loads configuration from a YAML file, initializes the
recording viewer, and starts the interactive session.

Usage:
    python scripts/1_record.py config/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.viewer_wrapper import KeyframeRecorder, create_default_sim_settings


def load_config(config_path: str) -> dict:
    """Load YAML configuration from file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Main entry point for interactive recording."""
    parser = argparse.ArgumentParser(
        description="Interactive keyframe recording with X11 window",
        epilog="Example: python scripts/1_record.py config/default.yaml"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to scene dataset config file (e.g., /replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene name (e.g., Baked_sc0_staging_01) when using --dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output keyframes JSON file"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1440,
        help="Window width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Window height"
    )
    parser.add_argument(
        "--hbao",
        action="store_true",
        help="Enable horizon-based ambient occlusion"
    )

    args = parser.parse_args()

    try:
        # Determine if using config file or command line args
        if args.config:
            # Config file mode
            config = load_config(args.config)

            # Extract required settings
            scene_config = config.get("scene", {})
            output_root = Path(config.get("output_root", "./output"))
            trajectory_settings = config.get("trajectory", {})
            rendering_settings = config.get("rendering", {})

            # Check if using dataset mode or standalone scene
            if "dataset" in scene_config and "name" in scene_config:
                # Dataset mode (Replica CAD)
                dataset_path = scene_config["dataset"]
                scene_path = scene_config["name"]
                scene_name = scene_path  # Use scene name directly for output dir
            elif "path" in scene_config:
                # Standalone scene mode (legacy Replica v1)
                scene_path = scene_config["path"]
                dataset_path = "default"
                scene_name = Path(scene_path).parent.name
            else:
                raise ValueError("Config must specify either 'scene.dataset + scene.name' OR 'scene.path'")

            output_dir = output_root / scene_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Path for keyframes JSON
            keyframes_path = output_dir / "keyframes.json"

            # Get window size from trajectory settings
            window_size = trajectory_settings.get("interactive_window_size", [1080, 1440])
            width = window_size[1]
            height = window_size[0]

            # Get rendering settings
            enable_hbao = rendering_settings.get("enable_hbao", False)

            # Get physics settings
            physics_settings = config.get("physics", {})
            enable_physics = physics_settings.get("enable_physics", True)

            # Get agent configuration for camera positioning and collision
            agent_config = trajectory_settings.get("agent", {})
            agent_height = agent_config.get("height", 1.5)
            agent_radius = agent_config.get("radius", 0.1)
            sensor_height = agent_config.get("sensor_height", 1.5)

        else:
            # Command line mode (like official viewer)
            if not args.dataset or not args.scene:
                parser.error("Either provide config file OR use --dataset and --scene together")

            scene_path = args.scene
            dataset_path = args.dataset
            width = args.width
            height = args.height
            enable_hbao = args.hbao
            enable_physics = True  # Always enable physics for articulated objects
            agent_height = 1.5  # Default agent collision height
            agent_radius = 0.1  # Default agent collision radius
            sensor_height = 1.5  # Default camera height offset

            # Output path
            if args.output:
                keyframes_path = Path(args.output)
            else:
                output_dir = Path("./output") / args.scene
                output_dir.mkdir(parents=True, exist_ok=True)
                keyframes_path = output_dir / "keyframes.json"

        print("="*70)
        print("INTERACTIVE KEYFRAME RECORDER")
        print("="*70)
        print(f"Scene: {scene_path}")
        if 'dataset_path' in locals() and dataset_path != "default":
            print(f"Dataset: {dataset_path}")
        print(f"Output: {keyframes_path}")
        print(f"Window size: {width}x{height}")
        print(f"Camera height: {sensor_height}m")
        print(f"HBAO: {enable_hbao}")
        print(f"Physics: {enable_physics}")
        print("="*70)

        # Create simulation settings
        sim_settings = create_default_sim_settings(
            scene_path=scene_path,
            dataset_path=dataset_path if 'dataset_path' in locals() else "default",
            width=width,
            height=height,
            enable_hbao=enable_hbao,
            enable_physics=enable_physics,
            agent_height=agent_height,
            agent_radius=agent_radius,
            sensor_height=sensor_height
        )

        # Start the recorder
        print("\nStarting interactive viewer...")
        print("Use WASD/Arrow keys to navigate, SPACE to capture, ESC to save and exit")

        recorder = KeyframeRecorder(sim_settings, output_path=str(keyframes_path))
        recorder.exec()

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfig error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
