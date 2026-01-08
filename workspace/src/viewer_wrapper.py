#!/usr/bin/env python3

"""
Simplified Interactive Keyframe Recorder for Habitat-sim

This module provides a standalone keyframe recording system that:
- Allows interactive navigation with WASD/arrow keys
- Captures camera keyframes (position + rotation) on SPACE
- Saves keyframes to JSON on ESC
"""

import ctypes
import json
import os
import sys
from enum import Enum
from typing import Any, Dict, List

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import text, shaders
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg


class MouseMode(Enum):
    """Mouse interaction modes"""
    LOOK = 0
    GRAB = 1  # Not implemented for keyframe recorder, but keep for consistency


class KeyframeRecorder(Application):
    """
    Interactive viewer that records camera keyframes for trajectory playback.

    Controls:
        - WASD: Move forward/backward and strafe left/right
        - Arrow keys: Turn left/right and look up/down
        - Z/X: Move up/down
        - SPACE: Capture current camera keyframe
        - ESC: Save keyframes to JSON and exit
    """

    MAX_DISPLAY_TEXT_CHARS = 512
    TEXT_DELTA_FROM_CENTER = 0.49
    # Font size will be calculated based on window height
    BASE_FONT_SIZE = 16.0
    BASE_WINDOW_HEIGHT = 680  # Reference height for font scaling

    def __init__(self, sim_settings: Dict[str, Any], output_path: str = "keyframes.json") -> None:
        """
        Initialize the keyframe recorder.

        Args:
            sim_settings: Habitat-sim configuration settings
            output_path: Path to save keyframes JSON file
        """
        self.sim_settings = sim_settings
        self.output_path = output_path
        self.keyframes = []

        # Setup window
        configuration = self.Configuration()
        configuration.title = "Habitat-sim Keyframe Recorder"
        configuration.size = (
            self.sim_settings["window_width"],
            self.sim_settings["window_height"],
        )
        Application.__init__(self, configuration)
        self.fps = 60.0

        # Setup movement key bindings
        key = Application.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.X: False,
            key.Z: False,
        }

        self.key_to_action = {
            key.UP: "look_up",
            key.DOWN: "look_down",
            key.LEFT: "turn_left",
            key.RIGHT: "turn_right",
            key.A: "move_left",
            key.D: "move_right",
            key.S: "move_backward",
            key.W: "move_forward",
            key.X: "move_down",
            key.Z: "move_up",
        }

        # Mouse interaction state (copied from official viewer line 168-170)
        self.mouse_interaction = MouseMode.LOOK
        self.previous_mouse_point = None

        # Setup display text rendering (optional, disable if font not found)
        self.text_rendering_enabled = False

        try:
            self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
            relative_path_to_font = "../support/fonts/proggy_clean/ProggyClean.ttf"
            font_path = os.path.join(os.path.dirname(__file__), relative_path_to_font)

            # Try to load font
            self.display_font.open_file(font_path, 13)

            # Setup glyph cache for text rendering
            self.glyph_cache = text.GlyphCacheGL(
                mn.PixelFormat.R8_UNORM, mn.Vector2i(256), mn.Vector2i(1)
            )
            self.display_font.fill_glyph_cache(
                self.glyph_cache,
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-_+,.! []",
            )

            # Calculate font size based on window height (scale proportionally)
            window_h = self.sim_settings.get("window_height", self.BASE_WINDOW_HEIGHT)
            self.display_font_size = self.BASE_FONT_SIZE * (window_h / self.BASE_WINDOW_HEIGHT)

            # Create text renderer
            self.window_text = text.Renderer2D(
                self.display_font,
                self.glyph_cache,
                self.display_font_size,
                text.Alignment.TOP_LEFT,
            )
            self.window_text.reserve(self.MAX_DISPLAY_TEXT_CHARS)

            # Setup text transform
            self.window_text_transform = mn.Matrix3.projection(
                self.framebuffer_size
            ) @ mn.Matrix3.translation(
                mn.Vector2(self.framebuffer_size)
                * mn.Vector2(
                    -self.TEXT_DELTA_FROM_CENTER,
                    self.TEXT_DELTA_FROM_CENTER,
                )
            )
            self.shader = shaders.VectorGL2D()

            # Setup blend function for text
            mn.gl.Renderer.set_blend_equation(
                mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
            )

            self.text_rendering_enabled = True
            print("Text rendering enabled")

        except Exception as e:
            print(f"Warning: Text rendering disabled - could not load font: {e}")
            print("The viewer will work without on-screen text display")

        # Configure and create simulator
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id = self.sim_settings.get("default_agent", 0)
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        self.sim = habitat_sim.Simulator(self.cfg)
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.render_camera = self.default_agent.scene_node.node_sensor_suite.get(
            "color_sensor"
        )

        print("\n" + "="*60)
        print("Keyframe Recorder - Controls:")
        print("="*60)
        print("WASD: Move forward/backward and strafe left/right")
        print("Arrow keys: Turn left/right and look up/down")
        print("Z/X: Move up/down")
        print("SPACE: Capture keyframe")
        print("ESC: Save keyframes and exit")
        print("="*60 + "\n")

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """Configure agent with movement actions."""
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space = {}
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        # Get agent configuration from sim_settings (populated by config YAML)
        agent_height = self.sim_settings.get("agent_height", 1.5)
        agent_radius = self.sim_settings.get("agent_radius", 0.1)
        sensor_height = self.sim_settings.get("sensor_height", 1.5)

        # Create RGB sensor with height offset
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [
            self.sim_settings["height"],
            self.sim_settings["width"]
        ]
        rgb_sensor.position = [0.0, sensor_height, 0.0]  # Y offset for camera height
        rgb_sensor.orientation = [0.0, 0.0, 0.0]

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=agent_height,
            radius=agent_radius,
            sensor_specifications=[rgb_sensor],
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def capture_keyframe(self) -> None:
        """Capture current camera transformation as a keyframe."""
        # Get actual camera sensor transformation (not agent body)
        # This ensures keyframes contain real camera position for standard SLAM datasets
        sensor = self.render_camera
        transform = sensor.node.absolute_transformation()

        # Extract position (camera optical center, not agent body)
        position = transform.translation
        pos = [float(position.x), float(position.y), float(position.z)]

        # Extract rotation as quaternion
        rotation = mn.Quaternion.from_matrix(transform.rotation())
        rot = [
            float(rotation.scalar),  # w component first
            float(rotation.vector.x),
            float(rotation.vector.y),
            float(rotation.vector.z),
        ]

        # Create keyframe dictionary
        keyframe = {
            "index": len(self.keyframes),
            "pos": pos,
            "rot": rot,
        }

        self.keyframes.append(keyframe)
        print(f"Keyframe {len(self.keyframes)} captured at position {pos}")

    def save_keyframes(self) -> None:
        """Save all captured keyframes to JSON file."""
        output_data = {
            "keyframes": self.keyframes,
            "total_keyframes": len(self.keyframes),
        }

        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved {len(self.keyframes)} keyframes to {self.output_path}")

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """Handle key press events."""
        key = event.key
        pressed = Application.Key

        if key == pressed.ESC:
            # Save keyframes and exit
            self.save_keyframes()
            event.accepted = True
            self.exit_event(Application.ExitEvent)
            return

        elif key == pressed.SPACE:
            # Capture keyframe
            self.capture_keyframe()
            event.accepted = True
            self.redraw()
            return

        # Update movement keys
        if key in self.pressed:
            self.pressed[key] = True

        event.accepted = True
        self.redraw()

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """Handle key release events."""
        key = event.key

        if key in self.pressed:
            self.pressed[key] = False

        event.accepted = True
        self.redraw()

    def move_and_look(self, repetitions: int) -> None:
        """
        Process movement based on currently pressed keys.
        Copied from official viewer lines 454-471.
        """
        if repetitions == 0:
            return

        agent = self.sim.agents[self.agent_id]
        press = self.pressed
        act = self.key_to_action

        action_queue = [act[k] for k, v in press.items() if v]

        for _ in range(int(repetitions)):
            [agent.act(x) for x in action_queue]

    def get_mouse_position(self, mouse_event_position: mn.Vector2i) -> mn.Vector2i:
        """
        Get screen-space mouse position scaled for framebuffer vs window size.
        Copied from official viewer lines 891-899.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size)
        return mouse_event_position * scaling

    def pointer_move_event(self, event: Application.PointerMoveEvent) -> None:
        """
        Handle mouse movement for camera look control.
        Copied from official viewer lines 680-711 (simplified to LOOK mode only).
        """
        # Only LOOK mode for keyframe recorder
        if (
            event.pointers & Application.Pointer.MOUSE_LEFT
            and self.mouse_interaction == MouseMode.LOOK
        ):
            agent = self.sim.agents[self.agent_id]
            delta = self.get_mouse_position(event.relative_position) / 2
            action = habitat_sim.agent.ObjectControls()
            act_spec = habitat_sim.agent.ActuationSpec

            # left/right on agent scene node
            action(agent.scene_node, "turn_right", act_spec(delta.x))

            # up/down on cameras' scene nodes
            action = habitat_sim.agent.ObjectControls()
            sensors = list(self.default_agent.scene_node.subtree_sensors.values())
            [action(s.object, "look_down", act_spec(delta.y), False) for s in sensors]

        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def pointer_press_event(self, event: Application.PointerEvent) -> None:
        """
        Handle mouse button press.
        Simplified from official viewer lines 713-809 (removed GRAB mode physics).
        """
        self.previous_mouse_point = self.get_mouse_position(event.position)
        self.redraw()
        event.accepted = True

    def scroll_event(self, event: Application.ScrollEvent) -> None:
        """
        Handle mouse wheel scrolling for camera zoom.
        Copied from official viewer lines 811-862 (LOOK mode only).
        """
        scroll_mod_val = (
            event.offset.y
            if abs(event.offset.y) > abs(event.offset.x)
            else event.offset.x
        )
        if not scroll_mod_val:
            return

        # use shift to scale action response (line 826-828)
        shift_pressed = bool(event.modifiers & Application.Modifier.SHIFT)

        # LOOK mode zoom (lines 831-837)
        if self.mouse_interaction == MouseMode.LOOK:
            # use shift for fine-grained zooming
            mod_val = 1.01 if shift_pressed else 1.1
            mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
            cam = self.render_camera
            cam.zoom(mod)
            self.redraw()

        event.accepted = True

    def pointer_release_event(self, event: Application.PointerEvent) -> None:
        """
        Handle mouse button release.
        Copied from official viewer lines 864-870 (simplified).
        """
        event.accepted = True

    def draw_event(self) -> None:
        """Main render loop."""
        import time

        # Clear framebuffer
        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )

        # Process movement
        self.move_and_look(1)

        # Render camera view
        self.sim._Simulator__sensors[self.agent_id]["color_sensor"].draw_observation()
        self.render_camera.render_target.blit_rgba_to_default()

        # Draw status text
        mn.gl.default_framebuffer.bind()
        self.draw_text()

        # Swap buffers and redraw
        self.swap_buffers()
        time.sleep(1.0 / self.fps)
        self.redraw()

    def draw_text(self) -> None:
        """Draw status text on screen."""
        # Skip if text rendering is not enabled
        if not self.text_rendering_enabled:
            return

        # Enable blending for text
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )

        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        # Get current position and rotation for display
        transform = self.default_agent.scene_node.transformation
        pos = transform.translation

        # Get rotation as euler angles (approximate)
        rot = mn.Quaternion.from_matrix(transform.rotation())

        # Get scene and dataset info
        scene_name = self.sim_settings.get("scene", "Unknown")
        dataset_name = self.sim_settings.get("scene_dataset_config_file", "default")
        if dataset_name != "default":
            dataset_name = dataset_name.split("/")[-1]  # Get filename only

        # Get window/framebuffer size
        win_w, win_h = self.window_size
        fb_w, fb_h = self.framebuffer_size

        status_text = f"""Keyframe Recorder
Scene: {scene_name}
Dataset: {dataset_name}
Resolution: {win_w}x{win_h} [FB: {fb_w}x{fb_h}]
FPS: {self.fps:.0f}
Position: [{pos.x:.2f} {pos.y:.2f} {pos.z:.2f}]
Rotation: [{rot.scalar:.2f} {rot.vector.x:.2f} {rot.vector.y:.2f} {rot.vector.z:.2f}]
Keyframes: {len(self.keyframes)}

Controls: WASD+Arrows+ZX=Move  Mouse=Look  Space=Capture  Esc=Save"""

        self.window_text.render(status_text)
        self.shader.draw(self.window_text.mesh)

        # Disable blending
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)

    def exit_event(self, event: Application.ExitEvent) -> None:
        """Clean up on exit."""
        self.sim.close(destroy=True)
        event.accepted = True
        exit(0)


def create_default_sim_settings(
    scene_path: str = "./data/test_assets/scenes/simple_room.glb",
    dataset_path: str = "default",
    width: int = 800,
    height: int = 600,
    enable_hbao: bool = False,
    enable_physics: bool = True,
    agent_height: float = 1.5,
    agent_radius: float = 0.1,
    sensor_height: float = 1.5,
) -> Dict[str, Any]:
    """
    Create default simulation settings for the keyframe recorder.

    Args:
        scene_path: Path to scene file or scene name (when using dataset)
        dataset_path: Path to scene dataset config file (or "default" for standalone scenes)
        width: Window width
        height: Window height
        enable_hbao: Enable horizon-based ambient occlusion
        enable_physics: Enable physics engine to load articulated objects (furniture with movable parts)
                       Replica CAD scenes require this to load refrigerators, cabinets, doors, etc.
                       Requires Bullet physics engine (installed via "habitat-sim withbullet")
        agent_height: Agent collision cylinder height (default: 1.5m)
        agent_radius: Agent collision cylinder radius (default: 0.1m)
        sensor_height: Camera Y offset from agent position for recording (default: 1.5m)

    Returns:
        Dictionary of simulation settings
    """
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = scene_path
    sim_settings["scene_dataset_config_file"] = dataset_path
    sim_settings["window_width"] = width
    sim_settings["window_height"] = height
    sim_settings["width"] = width
    sim_settings["height"] = height
    sim_settings["default_agent"] = 0
    sim_settings["enable_physics"] = enable_physics
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_hbao"] = enable_hbao
    sim_settings["agent_height"] = agent_height
    sim_settings["agent_radius"] = agent_radius
    sim_settings["sensor_height"] = sensor_height

    return sim_settings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive keyframe recorder for Habitat-sim"
    )
    parser.add_argument(
        "--scene",
        default="./data/test_assets/scenes/simple_room.glb",
        type=str,
        help="Path to scene file to load",
    )
    parser.add_argument(
        "--output",
        default="keyframes.json",
        type=str,
        help="Path to output JSON file for keyframes",
    )
    parser.add_argument(
        "--width",
        default=800,
        type=int,
        help="Window width",
    )
    parser.add_argument(
        "--height",
        default=600,
        type=int,
        help="Window height",
    )

    args = parser.parse_args()

    # Create simulation settings
    sim_settings = create_default_sim_settings(
        scene_path=args.scene,
        width=args.width,
        height=args.height,
    )

    # Start the recorder
    recorder = KeyframeRecorder(sim_settings, output_path=args.output)
    recorder.exec()
