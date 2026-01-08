#!/usr/bin/env python3
"""
Dataset Renderer - RGB-D Sequence Generator

This module handles rendering RGB-D sequences from 3D scenes using Habitat-sim.
Key features:
- High-quality RGB and Depth rendering
- Configurable camera parameters
- Selective frame rendering (full or indexed subsets)
- Camera intrinsics export
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List
from PIL import Image

try:
    import habitat_sim
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False


class DatasetRenderer:
    """
    Renders RGB-D datasets from trajectory data using Habitat-sim.
    """

    def __init__(self, scene_path: str, config: dict, dataset_path: Optional[str] = None):
        """
        Initialize renderer.

        Args:
            scene_path: Path to scene mesh file or scene name (for Replica CAD)
            config: Configuration dictionary containing 'camera' and 'rendering' settings
            dataset_path: Optional path to scene dataset config (for Replica CAD)
        """
        if not HABITAT_AVAILABLE:
            raise RuntimeError("habitat_sim is required for DatasetRenderer")

        self.scene_path = scene_path
        self.dataset_path = dataset_path
        self.config = config
        self.cam_cfg = config['camera']

        # Initialize simulator with RGB and Depth sensors
        self.sim = self._init_simulator()

    def _init_simulator(self) -> habitat_sim.Simulator:
        """Initialize Habitat simulator with RGB and Depth sensors."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path

        # CRITICAL: Enable physics to load articulated objects (furniture)
        # Replica CAD scenes contain articulated objects like refrigerators, cabinets, doors
        # Without physics, these objects will NOT be loaded and scenes appear empty
        physics_cfg = self.config.get('physics', {})
        sim_cfg.enable_physics = physics_cfg.get('enable_physics', True)

        # If using Replica CAD dataset, set dataset config file
        if self.dataset_path:
            sim_cfg.scene_dataset_config_file = self.dataset_path

        # Rendering settings from config
        render_cfg = self.config.get('rendering', {})
        sim_cfg.requires_textures = render_cfg.get('requires_textures', True)
        sim_cfg.enable_hbao = render_cfg.get('enable_hbao', False)  # Match official viewer default

        # GPU device ID (if specified)
        # If not specified or commented out, will use CPU rendering (software rasterization)
        gpu_device_id = render_cfg.get('gpu_device_id')
        if gpu_device_id is not None and gpu_device_id >= 0:
            sim_cfg.gpu_device_id = gpu_device_id
            print(f"[Renderer] Using GPU device {gpu_device_id}")
        else:
            sim_cfg.gpu_device_id = -1  # Use CPU
            print("[Renderer] GPU device ID not specified. Use CPU.")

        # RGB Sensor - position at [0,0,0] since keyframes store actual camera position
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.cam_cfg['height'], self.cam_cfg['width']]
        rgb_spec.position = [0.0, 0.0, 0.0]  # No offset - keyframes have camera position
        rgb_spec.orientation = [0.0, 0.0, 0.0]

        # Depth Sensor - position at [0,0,0] since keyframes store actual camera position
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.cam_cfg['height'], self.cam_cfg['width']]
        depth_spec.position = [0.0, 0.0, 0.0]  # No offset - keyframes have camera position
        depth_spec.orientation = [0.0, 0.0, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    def render_sequence(self,
                       positions: np.ndarray,
                       rotations: np.ndarray,
                       output_dir: str,
                       indices: Optional[List[int]] = None):
        """
        Render RGB-D sequence for the given trajectory.

        Args:
            positions: Nx3 array of camera positions
            rotations: Nx4 array of quaternions [w,x,y,z]
            output_dir: Directory to save results
            indices: Optional list of frame indices to render. If None, render all frames.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        agent = self.sim.get_agent(0)
        num_frames = len(positions)

        # Determine which frames to render
        if indices is None:
            frames_to_render = range(num_frames)
        else:
            frames_to_render = indices

        # Get rendering parameters
        render_cfg = self.config.get('rendering', {})
        jpeg_quality = render_cfg.get('jpeg_quality', 95)
        depth_scale = self.cam_cfg.get('depth_scale', 1000.0)

        # Render frames with progress bar
        for i in tqdm(frames_to_render, desc="Rendering frames"):
            # Set agent state
            state = habitat_sim.AgentState()
            state.position = positions[i]

            # Convert quaternion [w,x,y,z] to Habitat format
            q_wxyz = rotations[i]
            state.rotation = np.quaternion(q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3])

            agent.set_state(state)

            # Capture observations
            obs = self.sim.get_sensor_observations()
            rgb = obs['color_sensor'][:, :, :3]  # Remove alpha channel
            depth = obs['depth_sensor']

            # Save RGB as JPEG with specified quality
            rgb_img = Image.fromarray(rgb, mode='RGB')
            rgb_path = output_path / f"frame{i:04d}.jpg"
            rgb_img.save(rgb_path, 'JPEG', quality=jpeg_quality)

            # Save Depth as 16-bit PNG
            depth_mm = (depth * depth_scale).astype(np.uint16)
            depth_img = Image.fromarray(depth_mm, mode='I;16')
            depth_path = output_path / f"depth{i:04d}.png"
            depth_img.save(depth_path)

    def save_camera_params(self, output_path: str):
        """
        Save camera intrinsics as JSON.

        Args:
            output_path: Path to save JSON file
        """
        # Get camera intrinsics directly from configuration
        # Use config values if provided, otherwise calculate from HFOV
        width = self.cam_cfg['width']
        height = self.cam_cfg['height']

        # Use fx, fy from config if available, otherwise calculate from HFOV
        if 'fx' in self.cam_cfg and 'fy' in self.cam_cfg:
            fx = self.cam_cfg['fx']
            fy = self.cam_cfg['fy']
        else:
            # Calculate focal length from HFOV
            hfov = self.cam_cfg.get('hfov', 90.0)
            hfov_rad = np.deg2rad(hfov)
            fx = width / (2.0 * np.tan(hfov_rad / 2.0))
            fy = fx

        # Use cx, cy from config if available, otherwise use image center
        cx = self.cam_cfg.get('cx', width / 2.0 - 0.5)
        cy = self.cam_cfg.get('cy', height / 2.0 - 0.5)

        # Get depth scale from config (Replica standard parameter)
        depth_scale = self.cam_cfg.get('depth_scale', 6553.5)

        # Camera parameters dictionary (Replica standard format)
        camera_params = {
            "camera": {
                'w': width,
                'h': height,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'scale': depth_scale
            }
        }

        # Save as JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(camera_params, f, indent=2)

    def close(self):
        """Clean up simulator resources."""
        if hasattr(self, 'sim'):
            self.sim.close()

    def __del__(self):
        """Ensure simulator is closed on deletion."""
        self.close()
