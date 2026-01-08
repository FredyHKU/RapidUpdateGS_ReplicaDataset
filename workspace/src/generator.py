#!/usr/bin/env python3
"""
Trajectory Generator Module - Production-Ready Implementation

This module provides trajectory generation and interpolation capabilities for camera motion,
including position interpolation, rotation interpolation (Slerp), head sway simulation,
and trajectory visualization.

Features:
- Deterministic trajectory generation via seed control
- Cubic spline interpolation for smooth position trajectories
- Spherical linear interpolation (Slerp) for rotation quaternions
- Configurable head sway for natural camera motion
- Multi-view trajectory visualization (XY, XZ, YZ planes)
- Standard trajectory file export format
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings


class TrajectoryGenerator:
    """
    Generate and process camera trajectories with position/rotation interpolation.

    This class provides methods for:
    - Keyframe-based trajectory interpolation
    - Smooth position transitions using cubic splines
    - Smooth rotation transitions using spherical linear interpolation
    - Natural head sway simulation
    - Trajectory visualization and export

    Attributes:
        scene_path (str): Path to the scene file (for reference/context)
        seed (int): Random seed for deterministic behavior
    """

    def __init__(self, scene_path: str, seed: int = 42):
        """
        Initialize trajectory generator.

        Args:
            scene_path: Path to scene mesh/file (stored for reference)
            seed: Random seed for reproducible trajectory generation
        """
        self.scene_path = scene_path
        self.seed = seed

        # Set random seed for deterministic behavior
        np.random.seed(seed)

        print(f"[TrajectoryGenerator] Initialized")
        print(f"  Scene: {scene_path}")
        print(f"  Seed: {seed}")

    def interpolate_keyframes(self,
                            keyframes: List[Dict],
                            num_frames: int,
                            sway_cfg: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate positions and rotations between keyframes.

        This method performs smooth interpolation between user-defined keyframes:
        - Positions: Cubic spline interpolation for smooth spatial motion
        - Rotations: Spherical linear interpolation (Slerp) for smooth orientation
        - Optional head sway: Sinusoidal perturbations for natural motion

        Args:
            keyframes: List of keyframe dictionaries with structure:
                       [{"pos": [x, y, z], "rot": [w, x, y, z]}, ...]
                       where pos is position in meters and rot is quaternion
            num_frames: Number of interpolated frames to generate
            sway_cfg: Optional head sway configuration dict with keys:
                     - 'enabled' (bool): Enable head sway
                     - 'amplitude' (float): Sway amplitude in degrees (default: 2.0)
                     - 'frequency' (float): Sway frequency in Hz (default: 1.0)

        Returns:
            positions: (num_frames, 3) array of interpolated positions [x, y, z]
            rotations: (num_frames, 4) array of interpolated quaternions [w, x, y, z]

        Raises:
            ValueError: If keyframes list is invalid or contains malformed data
        """
        # Validate input
        if not keyframes or len(keyframes) < 2:
            raise ValueError(f"At least 2 keyframes required, got {len(keyframes) if keyframes else 0}")

        print(f"\n[Keyframe Interpolation] Processing {len(keyframes)} keyframes -> {num_frames} frames")

        # Extract positions and rotations from keyframes
        try:
            key_positions = np.array([kf['pos'] for kf in keyframes])  # (N, 3)
            key_rotations = np.array([kf['rot'] for kf in keyframes])  # (N, 4) [w, x, y, z]
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid keyframe format. Expected dict with 'pos' and 'rot' keys: {e}")

        # Validate shapes
        if key_positions.shape[1] != 3:
            raise ValueError(f"Positions must be 3D, got shape {key_positions.shape}")
        if key_rotations.shape[1] != 4:
            raise ValueError(f"Rotations must be quaternions (4D), got shape {key_rotations.shape}")

        # 1. POSITION INTERPOLATION - Cubic Spline
        print("  [1/3] Interpolating positions (Cubic Spline)...")
        positions = self._interpolate_positions(key_positions, num_frames)

        # 2. ROTATION INTERPOLATION - Slerp
        print("  [2/3] Interpolating rotations (Slerp)...")
        rotations = self._interpolate_rotations(key_rotations, num_frames)

        # 3. HEAD SWAY (Optional)
        if sway_cfg and sway_cfg.get('enabled', False):
            print("  [3/3] Applying head sway...")
            rotations = self._apply_head_sway(
                rotations,
                num_frames,
                amplitude=sway_cfg.get('amplitude', 2.0),
                frequency=sway_cfg.get('frequency', 1.0)
            )
        else:
            print("  [3/3] Head sway disabled")

        print(f"  Interpolation complete: {positions.shape[0]} frames")

        return positions, rotations

    def _interpolate_positions(self, key_positions: np.ndarray, num_frames: int) -> np.ndarray:
        """
        Interpolate positions using cubic spline.

        Args:
            key_positions: (N, 3) array of keyframe positions
            num_frames: Number of output frames

        Returns:
            (num_frames, 3) array of interpolated positions
        """
        num_keyframes = len(key_positions)

        # Parameter values for keyframes (normalized to [0, 1])
        key_times = np.linspace(0, 1, num_keyframes)
        target_times = np.linspace(0, 1, num_frames)

        # Handle edge case: only 2 keyframes (linear interpolation)
        if num_keyframes == 2:
            warnings.warn("Only 2 keyframes provided, using linear interpolation instead of cubic spline")
            positions = np.zeros((num_frames, 3))
            for dim in range(3):
                positions[:, dim] = np.interp(target_times, key_times, key_positions[:, dim])
            return positions

        # Cubic spline for each dimension
        positions = np.zeros((num_frames, 3))
        for dim in range(3):
            # Create cubic spline (bc_type='natural' for natural boundary conditions)
            cs = CubicSpline(key_times, key_positions[:, dim], bc_type='natural')
            positions[:, dim] = cs(target_times)

        return positions

    def _interpolate_rotations(self, key_rotations: np.ndarray, num_frames: int) -> np.ndarray:
        """
        Interpolate rotations using Spherical Linear Interpolation (Slerp).

        Args:
            key_rotations: (N, 4) array of keyframe quaternions [w, x, y, z]
            num_frames: Number of output frames

        Returns:
            (num_frames, 4) array of interpolated quaternions [w, x, y, z]
        """
        num_keyframes = len(key_rotations)

        # Convert from [w, x, y, z] to scipy format [x, y, z, w]
        key_rot_scipy = key_rotations[:, [1, 2, 3, 0]]

        # Normalize quaternions to ensure unit length
        norms = np.linalg.norm(key_rot_scipy, axis=1, keepdims=True)
        key_rot_scipy = key_rot_scipy / norms

        # Create Rotation objects
        key_rotations_obj = R.from_quat(key_rot_scipy)

        # Parameter values for keyframes
        key_times = np.linspace(0, 1, num_keyframes)
        target_times = np.linspace(0, 1, num_frames)

        # Spherical linear interpolation
        slerp = Slerp(key_times, key_rotations_obj)
        interp_rotations = slerp(target_times)

        # Convert back to [w, x, y, z] format
        quat_xyzw = interp_rotations.as_quat()  # (num_frames, 4) [x, y, z, w]
        rotations = quat_xyzw[:, [3, 0, 1, 2]]  # Convert to [w, x, y, z]

        return rotations

    def _apply_head_sway(self,
                        rotations: np.ndarray,
                        num_frames: int,
                        amplitude: float = 2.0,
                        frequency: float = 1.0,
                        fps: float = 30.0) -> np.ndarray:
        """
        Apply sinusoidal head sway to rotations for natural camera motion.

        This simulates natural head movement patterns that are beneficial for
        SLAM algorithms and create more realistic camera trajectories.

        Args:
            rotations: (num_frames, 4) array of quaternions [w, x, y, z]
            num_frames: Number of frames
            amplitude: Sway amplitude in degrees (default: 2.0)
            frequency: Sway frequency in Hz (default: 1.0)
            fps: Frames per second for timing (default: 30.0)

        Returns:
            (num_frames, 4) array of quaternions with sway applied
        """
        # Convert amplitude to radians
        amplitude_rad = np.radians(amplitude)

        # Generate time array
        time_array = np.arange(num_frames) / fps

        # Generate sinusoidal yaw noise
        yaw_noise = amplitude_rad * np.sin(2 * np.pi * frequency * time_array)

        # Apply noise to each rotation
        swayed_rotations = []
        for i, quat_wxyz in enumerate(rotations):
            # Convert to scipy format [x, y, z, w]
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]

            # Create base rotation
            base_rot = R.from_quat(quat_xyzw)

            # Create noise rotation (around Y-axis/yaw)
            noise_rot = R.from_euler('y', yaw_noise[i])

            # Compose rotations: base rotation followed by sway
            final_rot = base_rot * noise_rot

            # Convert back to [w, x, y, z] format
            final_quat_xyzw = final_rot.as_quat()
            final_quat_wxyz = final_quat_xyzw[[3, 0, 1, 2]]

            swayed_rotations.append(final_quat_wxyz)

        return np.array(swayed_rotations)

    def generate_trajectory_plots(self,
                                 positions: np.ndarray,
                                 output_dir: str,
                                 keyframe_positions: Optional[np.ndarray] = None) -> None:
        """
        Generate trajectory visualization plots for three orthogonal views.

        Creates three matplotlib plots showing trajectory projections:
        - XY plane (front view): horizontal vs vertical motion
        - XZ plane (top-down view): bird's eye view
        - YZ plane (side view): depth vs height

        Args:
            positions: (N, 3) array of camera positions [x, y, z]
            output_dir: Directory path to save visualization images
            keyframe_positions: Optional (K, 3) array of keyframe positions to mark on plots

        Saves:
            - trajectory_preview_XY.png
            - trajectory_preview_XZ.png
            - trajectory_preview_YZ.png
        """
        print(f"\n[Trajectory Visualization] Generating plots...")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract coordinates
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Get keyframe positions if provided
        if keyframe_positions is not None and len(keyframe_positions) > 2:
            intermediate_kf = keyframe_positions[1:-1]
        else:
            intermediate_kf = None

        # Common plot settings
        figsize = (10, 8)
        linewidth = 2
        dpi = 150

        # Plot 1: XY Plane (Front View)
        print(f"  Creating XY plane plot...")
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, 'b-', linewidth=linewidth, label='Trajectory', alpha=0.7)
        ax.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(x[-1], y[-1], 'r^', markersize=12, label='End', zorder=5)

        # Mark keyframes if provided
        if intermediate_kf is not None:
            kf_x = intermediate_kf[:, 0]
            kf_y = intermediate_kf[:, 1]
            ax.plot(kf_x, kf_y, 'o', markersize=8, markerfacecolor='none',
                   markeredgecolor='orange', markeredgewidth=2, label='Keyframes', zorder=4)

        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title('Trajectory - XY Plane (Front View)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.axis('equal')

        xy_path = output_path / 'trajectory_preview_XY.png'
        plt.savefig(xy_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {xy_path}")

        # Plot 2: XZ Plane (Top-Down View)
        print(f"  Creating XZ plane plot...")
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, z, 'g-', linewidth=linewidth, label='Trajectory', alpha=0.7)
        ax.plot(x[0], z[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(x[-1], z[-1], 'r^', markersize=12, label='End', zorder=5)

        # Mark keyframes if provided
        if intermediate_kf is not None:
            kf_x = intermediate_kf[:, 0]
            kf_z = intermediate_kf[:, 2]
            ax.plot(kf_x, kf_z, 'o', markersize=8, markerfacecolor='none',
                   markeredgecolor='orange', markeredgewidth=2, label='Keyframes', zorder=4)

        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Z Position (meters)', fontsize=12)
        ax.set_title('Trajectory - XZ Plane (Top-Down View)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.axis('equal')

        xz_path = output_path / 'trajectory_preview_XZ.png'
        plt.savefig(xz_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {xz_path}")

        # Plot 3: YZ Plane (Side View)
        print(f"  Creating YZ plane plot...")
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(z, y, 'm-', linewidth=linewidth, label='Trajectory', alpha=0.7)
        ax.plot(z[0], y[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(z[-1], y[-1], 'r^', markersize=12, label='End', zorder=5)

        # Mark keyframes if provided
        if intermediate_kf is not None:
            kf_z = intermediate_kf[:, 2]
            kf_y = intermediate_kf[:, 1]
            ax.plot(kf_z, kf_y, 'o', markersize=8, markerfacecolor='none',
                   markeredgecolor='orange', markeredgewidth=2, label='Keyframes', zorder=4)

        ax.set_xlabel('Z Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        ax.set_title('Trajectory - YZ Plane (Side View)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        yz_path = output_path / 'trajectory_preview_YZ.png'
        plt.savefig(yz_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {yz_path}")

        print(f"  Visualization complete: 3 plots saved to {output_dir}")

    def save_traj_txt(self,
                     positions: np.ndarray,
                     rotations: np.ndarray,
                     output_path: str,
                     fps: float = 30.0) -> None:
        """
        Save trajectory to text file in Replica dataset format.

        The Replica format stores 4x4 transformation matrices (one per line).
        Each line contains 16 space-separated values in row-major order:
        R00 R01 R02 tx  R10 R11 R12 ty  R20 R21 R22 tz  0 0 0 1

        Args:
            positions: (N, 3) array of positions [x, y, z]
            rotations: (N, 4) array of quaternions [w, x, y, z]
            output_path: Output file path (will be created/overwritten)
            fps: Frames per second (used for statistics only)

        Raises:
            ValueError: If positions and rotations have different lengths
        """
        if len(positions) != len(rotations):
            raise ValueError(f"Position and rotation arrays must have same length: "
                           f"{len(positions)} vs {len(rotations)}")

        print(f"\n[Trajectory Export] Saving to {output_path}")
        print(f"  Format: Replica (4x4 transformation matrix per line)")
        print(f"  Frames: {len(positions)}")

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        num_frames = len(positions)

        # Open file and write trajectory (NO comments for Replica format)
        with open(output_path, 'w') as f:
            for i in range(num_frames):
                pos = positions[i]  # [x, y, z]
                rot = rotations[i]  # [w, x, y, z]

                # Convert quaternion [w, x, y, z] to scipy format [x, y, z, w]
                quat_xyzw = rot[[1, 2, 3, 0]]

                # Convert to rotation matrix
                rot_obj = R.from_quat(quat_xyzw)
                rot_matrix = rot_obj.as_matrix()  # 3x3 rotation matrix

                # Build 4x4 transformation matrix
                # [R00 R01 R02 tx ]
                # [R10 R11 R12 ty ]
                # [R20 R21 R22 tz ]
                # [0   0   0   1  ]

                # Write in row-major order (16 values per line)
                # Row 0: R00 R01 R02 tx
                f.write(f"{rot_matrix[0,0]:.18e} {rot_matrix[0,1]:.18e} {rot_matrix[0,2]:.18e} {pos[0]:.18e} ")
                # Row 1: R10 R11 R12 ty
                f.write(f"{rot_matrix[1,0]:.18e} {rot_matrix[1,1]:.18e} {rot_matrix[1,2]:.18e} {pos[1]:.18e} ")
                # Row 2: R20 R21 R22 tz
                f.write(f"{rot_matrix[2,0]:.18e} {rot_matrix[2,1]:.18e} {rot_matrix[2,2]:.18e} {pos[2]:.18e} ")
                # Row 3: 0 0 0 1
                f.write(f"0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00\n")

        print(f"  Trajectory saved successfully")

        # Print statistics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        duration = (num_frames - 1) / fps
        print(f"  Statistics:")
        print(f"    Duration: {duration:.2f} seconds")
        print(f"    Total path length: {total_distance:.2f} meters")
        if duration > 0:
            print(f"    Average speed: {total_distance/duration:.2f} m/s")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("TrajectoryGenerator - Example Usage")
    print("=" * 70)

    # Example 1: Simple keyframe interpolation
    print("\n[Example 1] Basic keyframe interpolation")

    # Create generator
    generator = TrajectoryGenerator(
        scene_path="/path/to/scene.glb",
        seed=42
    )

    # Define keyframes (simple circular path)
    keyframes = [
        {"pos": [0.0, 1.5, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
        {"pos": [2.0, 1.5, 0.0], "rot": [0.924, 0.0, 0.383, 0.0]},
        {"pos": [2.0, 1.5, 2.0], "rot": [0.707, 0.0, 0.707, 0.0]},
        {"pos": [0.0, 1.5, 2.0], "rot": [0.383, 0.0, 0.924, 0.0]},
        {"pos": [0.0, 1.5, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
    ]

    # Interpolate
    positions, rotations = generator.interpolate_keyframes(
        keyframes=keyframes,
        num_frames=300,
        sway_cfg={"enabled": True, "amplitude": 2.0, "frequency": 1.0}
    )

    # Visualize
    generator.generate_trajectory_plots(
        positions=positions,
        output_dir="./output"
    )

    # Save trajectory
    generator.save_traj_txt(
        positions=positions,
        rotations=rotations,
        output_path="./output/trajectory.txt",
        fps=30.0
    )

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
