#!/usr/bin/env python3
"""
Dataset generation script from recorded keyframes.

This script takes the recorded keyframes, interpolates a smooth trajectory,
and renders the RGB-D sequence. It supports two modes:
1. Preview mode: Visualize trajectory and render sparse frames
2. Full render mode: Generate complete RGB-D dataset

Usage:
    python scripts/2_generate.py config/default.yaml [--preview-only]

The script will:
- Load keyframes from output/<scene_name>/keyframes.json
- Interpolate smooth trajectory using TrajectoryGenerator
- In preview mode: Generate plots and sparse preview frames
- In full mode: Render all frames and save trajectory data
"""

import os
# CRITICAL: Set rendering backend BEFORE importing any habitat_sim modules
# Force CPU rendering with swiftshader for Docker environments without GPU
os.environ['MAGNUM_DEVICE'] = 'swiftshader'
os.environ['MAGNUM_LOG'] = 'quiet'

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from tqdm import tqdm

from src.generator import TrajectoryGenerator
from src.renderer import DatasetRenderer


def load_config(config_path: str) -> dict:
    """Load YAML configuration from file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required top-level fields
    if 'scene' not in config:
        raise ValueError("Missing required section in config: 'scene'")

    scene_cfg = config.get('scene', {})
    # Check for either standalone scene or Replica CAD dataset
    has_standalone = 'path' in scene_cfg
    has_replica_cad = 'dataset' in scene_cfg and 'name' in scene_cfg

    if not (has_standalone or has_replica_cad):
        raise ValueError("Config must contain either 'scene.path' (standalone) or 'scene.dataset' + 'scene.name' (Replica CAD)")

    if 'trajectory' not in config:
        raise ValueError("Missing required field in config: 'trajectory'")

    return config


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate RGB-D dataset from recorded keyframes",
        epilog="Example: python scripts/2_generate.py config/default.yaml --preview-only"
    )
    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--preview-only", "--preview",
        action="store_true",
        help="Generate trajectory plots and preview frames only (no full render)"
    )

    args = parser.parse_args()

    try:
        print("=" * 70)
        print("DATASET GENERATOR - TRAJECTORY INTERPOLATION & RENDERING")
        print("=" * 70)

        # Load configuration
        config = load_config(args.config)

        # Extract scene settings (support both Replica CAD dataset and standalone scenes)
        scene_cfg = config.get("scene", {})

        # For Replica CAD dataset: use dataset config + scene name
        if "dataset" in scene_cfg and "name" in scene_cfg:
            dataset_path = scene_cfg["dataset"]
            scene_name = scene_cfg["name"]
            scene_path = scene_name  # Replica CAD uses scene name as identifier
            print(f"\nScene Type: Replica CAD Dataset")
            print(f"Dataset: {dataset_path}")
            print(f"Scene: {scene_name}")
        # For standalone scenes: use direct path
        elif "path" in scene_cfg:
            scene_path = scene_cfg["path"]
            scene_name = Path(scene_path).parent.name
            dataset_path = None
            print(f"\nScene Type: Standalone")
            print(f"Scene path: {scene_path}")
            print(f"Scene name: {scene_name}")
        else:
            raise ValueError("Config must contain either 'scene.path' (standalone) or 'scene.dataset' + 'scene.name' (Replica CAD)")

        output_root = Path(config.get("output_root", "./output"))
        trajectory_settings = config.get("trajectory", {})

        print(f"Output root: {output_root}")

        # Check for keyframes file
        keyframes_path = output_root / scene_name / "keyframes.json"
        if not keyframes_path.exists():
            print(f"\nError: Keyframes file not found: {keyframes_path}")
            print("Please run 'python scripts/1_record.py' first to record keyframes.")
            sys.exit(1)

        # Load keyframes
        print(f"\nLoading keyframes from: {keyframes_path}")
        with open(keyframes_path, 'r') as f:
            data = json.load(f)
            keyframes = data.get('keyframes', data)  # Handle both formats

        print(f"Loaded {len(keyframes)} keyframes")

        # Step 1: Interpolate trajectory
        print("\n" + "=" * 70)
        print("STEP 1: TRAJECTORY INTERPOLATION")
        print("=" * 70)

        generator = TrajectoryGenerator(
            scene_path=scene_path,
            seed=config.get('seed', 42)
        )

        num_frames = trajectory_settings.get('num_frames', 2000)
        sway_cfg = trajectory_settings.get('head_sway', {})

        print(f"Interpolating to {num_frames} frames...")
        positions, rotations = generator.interpolate_keyframes(
            keyframes=keyframes,
            num_frames=num_frames,
            sway_cfg=sway_cfg
        )

        print(f"Generated trajectory with {len(positions)} frames")

        # Step 2: Rendering
        print("\n" + "=" * 70)
        if args.preview_only:
            print("STEP 2: PREVIEW MODE")
        else:
            print("STEP 2: FULL RGB-D DATASET RENDERING")
        print("=" * 70)

        scene_output_dir = output_root / scene_name

        # Common Step A: Generate trajectory plots (always done)
        print("\n[2A] Generating trajectory visualization plots...")
        keyframe_positions = np.array([kf['pos'] for kf in keyframes])
        generator.generate_trajectory_plots(
            positions=positions,
            output_dir=str(scene_output_dir),
            keyframe_positions=keyframe_positions
        )
        print(f"Saved trajectory plots to: {scene_output_dir}/trajectory_preview_*.png")

        # Initialize renderer (pass dataset_path for Replica CAD)
        renderer = DatasetRenderer(
            scene_path=scene_path,
            config=config,
            dataset_path=dataset_path if 'dataset_path' in locals() else None
        )

        # Common Step B: Render original keyframes (always done for verification)
        print("\n[2B] Rendering original keyframes (for verification)...")
        keyframe_rotations = np.array([kf['rot'] for kf in keyframes])
        keyframes_dir = scene_output_dir / "keyframes_render"
        renderer.render_sequence(
            positions=keyframe_positions,
            rotations=keyframe_rotations,
            output_dir=keyframes_dir,
            indices=None  # Render all keyframes
        )

        # Mode-specific rendering
        if args.preview_only:
            # Preview mode: Render sparse preview frames
            print("\n[2C] Rendering interpolated preview frames...")
            sample_indices = np.linspace(0, len(positions) - 1, 6, dtype=int).tolist()
            preview_frames_dir = scene_output_dir / "preview_frames"
            renderer.render_sequence(
                positions=positions,
                rotations=rotations,
                output_dir=preview_frames_dir,
                indices=sample_indices
            )

            renderer.close()

            # Summary
            print("\n" + "=" * 70)
            print("PREVIEW COMPLETE!")
            print("=" * 70)
            print(f"\nTrajectory Plots:")
            print(f"  {scene_output_dir / 'trajectory_preview_*.png'}")
            print(f"\nOriginal Keyframes Render (for verification):")
            print(f"  {keyframes_dir}/ ({len(keyframes)} frames)")
            print(f"\nInterpolated Preview Frames:")
            print(f"  {preview_frames_dir}/ ({len(sample_indices)} frames)")
            print(f"\nNote: Compare keyframes_render with what you saw during recording")
            print(f"      to verify that position/rotation data was saved correctly.")

        else:
            # Full render mode: Render all frames and save trajectory data
            print("\n[2C] Rendering full interpolated sequence...")
            results_dir = scene_output_dir / "results"
            renderer.render_sequence(
                positions=positions,
                rotations=rotations,
                output_dir=results_dir
            )

            print("\n[2D] Saving trajectory data...")
            traj_path = scene_output_dir / "traj.txt"
            fps = trajectory_settings.get('fps', 30)
            generator.save_traj_txt(
                positions=positions,
                rotations=rotations,
                output_path=str(traj_path),
                fps=fps
            )

            print("\n[2E] Saving camera parameters...")
            cam_params_path = scene_output_dir / "cam_params.json"
            renderer.save_camera_params(str(cam_params_path))

            renderer.close()

            # Summary
            print("\n" + "=" * 70)
            print("DATASET GENERATION COMPLETE!")
            print("=" * 70)
            print(f"\nOutput directory: {scene_output_dir}")
            print(f"  - traj.txt                    [Camera poses]")
            print(f"  - cam_params.json             [Camera intrinsics]")
            print(f"  - results/frame*.jpg          [RGB images ({len(positions)} frames)]")
            print(f"  - results/depth*.png          [Depth maps ({len(positions)} frames)]")
            print(f"  - keyframes_render/           [Original keyframes ({len(keyframes)} frames)]")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nConfig error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"\nMissing configuration key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
