# Habitat-Sim Interactive Keyframe Recorder

Docker-based tool for recording camera trajectories and generating RGB-D datasets from Replica CAD scenes with GUI support on Windows.

## Features

- 🎮 Interactive navigation with WASD + Mouse controls
- 🎯 Keyframe capture with spatial information
- 📹 Trajectory interpolation and smooth camera paths
- 📊 RGB-D dataset rendering with depth maps
- 🏠 Full Replica CAD scene support (furniture, doors, cabinets)
- 🐳 Reproducible Docker environment with X11 GUI

## Quick Start

### Prerequisites

1. **Docker Desktop for Windows** - https://www.docker.com/products/docker-desktop/
2. **VcXsrv (X11 Server)** - https://sourceforge.net/projects/vcxsrv/
3. **Replica CAD Baked Lighting Dataset** - Place in `./replica_cad_baked_lighting/`

### Setup

**1. Start X11 Server (VcXsrv)**

Launch XLaunch with these settings:
- Display: Multiple windows
- Client startup: Start no client
- ✅ **Disable access control** (CRITICAL!)
- ✅ Clipboard
- ❌ Native opengl

**2. Build and Start Container**

```bash
# Build Docker image (first time only, ~15 minutes)
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker exec -it habitat-sim-gui bash
```

## Workflow

The tool has a 2-step workflow:

### Step 1: Record Keyframes (Interactive)

Navigate the scene and capture keyframes at desired viewpoints.

```bash
cd /workspace
source activate habitat

# Edit config/default.yaml to select scene, then:
python scripts/1_record.py config/default.yaml
```

**Controls:**
- `WASD` - Move forward/back, strafe left/right
- `Mouse` - Look around (left-click + drag)
- `Z/X` - Move up/down
- `Space` - Capture keyframe
- `ESC` - Save and exit

**Output:** `output/<scene_name>/keyframes.json`

### Step 2: Generate Trajectory & Render Dataset

This step handles both trajectory generation and rendering.

#### 2A. Preview Mode (Recommended First)

Generate trajectory plots and preview frames to verify the path before full rendering:

```bash
python scripts/2_generate.py config/default.yaml --preview-only
```

**Output:**
```
output/<scene_name>/
├── trajectory_preview_XY.png   # Top-down view
├── trajectory_preview_XZ.png   # Side view
├── trajectory_preview_YZ.png   # Front view
├── keyframes_render/           # Renders of recorded keyframes (verification)
└── preview_frames/             # 6 sparse frames from interpolated path
```

**Why Preview?**
- Visualize trajectory shape before committing to full render
- Verify recorded keyframes render correctly
- Check smoothness and coverage of interpolated path
- Identify issues early (e.g., clipping, unwanted rotations)

#### 2B. Full Rendering

Once trajectory looks good, render the complete RGB-D dataset:

```bash
python scripts/2_generate.py config/default.yaml
```

**Output:**
```
output/<scene_name>/
├── traj.txt                    # Camera poses (timestamp tx ty tz qx qy qz qw)
├── cam_params.json             # Camera intrinsics
├── keyframes_render/           # Original keyframe renders (verification)
└── results/
    ├── frame0000.jpg           # RGB images
    ├── depth0000.png           # Depth maps (16-bit)
    ├── frame0001.jpg
    ├── depth0001.png
    └── ...
```

**What Happens:**
1. Loads keyframes from `keyframes.json`
2. Interpolates smooth trajectory using cubic splines (position) and Slerp (rotation)
3. Optionally applies head sway for realistic handheld motion
4. Renders RGB-D frames for every pose in trajectory
5. Saves camera intrinsics and trajectory file

## Configuration

Edit `workspace/config/default.yaml`:

```yaml
# Scene selection
scene:
  dataset: "/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json"
  name: "Baked_sc0_staging_01"

# Output directory
output_root: "./output"
seed: 1024  # For reproducible camera shake

# Trajectory parameters
trajectory:
  interactive_window_size: [480, 640]  # [height, width] - X11 window size
  num_frames: 2000                     # Total frames to generate
  fps: 60                              # Playback framerate
  smoothness: 0.5                      # Spline smoothness (0=exact, >0=smoother)
  interpolation: "cubic"               # Position interpolation method

  # Agent configuration - CRITICAL: Must be IDENTICAL for recording AND rendering!
  agent:
    height: 1.5         # Agent body collision cylinder height
    radius: 0.1         # Agent collision radius
    sensor_height: 1.5  # Camera Y-offset from agent position (1.5m = human eye level)

  head_sway:
    enabled: true                      # Add realistic head wobble
    amplitude: 2.0                     # Degrees of sway
    frequency: 1.0                     # Hz (breathing rate)

# Camera parameters (Replica standard)
camera:
  width: 1200
  height: 680
  fx: 600.0
  fy: 600.0
  cx: 599.5
  cy: 339.5
  depth_scale: 6553.5                  # 1m = 6553.5 units

# Rendering quality
rendering:
  enable_hbao: false                   # Ambient occlusion (soft shadows)
  jpeg_quality: 100                    # RGB compression quality (0-100)
  # gpu_device_id: 0                   # Uncomment to use GPU (0=first GPU, 1=second GPU, etc.)
                                       # If not set or -1: uses CPU rendering (SwiftShader)

# Physics settings
physics:
  enable_physics: true                 # Required to load furniture (refrigerators, cabinets, doors)
```

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── workspace/
│   ├── config/
│   │   └── default.yaml              # Main configuration
│   ├── scripts/
│   │   ├── 1_record.py               # Interactive keyframe recording
│   │   └── 2_generate.py             # Trajectory generation & rendering
│   ├── src/
│   │   ├── viewer_wrapper.py         # Interactive recorder (based on habitat-sim viewer)
│   │   ├── generator.py              # Trajectory interpolation and plots
│   │   └── renderer.py               # RGB-D rendering engine
│   └── output/                       # Generated datasets
│       └── <scene_name>/
│           ├── keyframes.json        # Recorded waypoints
│           ├── traj.txt              # Full trajectory
│           ├── cam_params.json       # Camera intrinsics
│           ├── trajectory_preview_*.png  # Debug plots
│           ├── keyframes_render/    # Original keyframe verification
│           ├── preview_frames/       # Sparse interpolated preview
│           └── results/              # Final RGB-D dataset
└── replica_cad_baked_lighting/       # Dataset (volume)
```

## Common Tasks

### Change Scene

Edit `config/default.yaml`:
```yaml
scene:
  name: "Baked_sc1_staging_05"  # Pick any scene from the dataset
```

### List Available Scenes

```bash
docker exec habitat-sim-gui bash -c "
source activate habitat && python -c \"
import json
with open('/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json') as f:
    scenes = list(json.load(f)['scene_instances'].keys())
    print(f'Available scenes ({len(scenes)}):')
    for s in sorted(scenes)[:20]: print(f'  {s}')
\""
```

### Adjust Recording Resolution

**Important:** Window size should match camera resolution for crisp rendering!

```yaml
camera:
  width: 640           # Final rendered image width
  height: 480          # Final rendered image height
trajectory:
  interactive_window_size: [480, 640]  # [height, width] - must match camera resolution!
```

**Note:** The interactive window size only affects the recording viewer (X11 window), not the final rendered output. Final renders always use `camera.width` × `camera.height`.

### Change Camera Height

```yaml
trajectory:
  agent:
    height: 1.1         # Agent collision cylinder height
    radius: 0.1         # Agent collision radius (rarely needs changing)
    sensor_height: 1.1  # Camera Y-offset (should match height for alignment)
```

Standard heights: `1.5m` standing, `1.1m` seated, `0.9m` wheelchair, `0.3m` child/ground view.

**Note:** `sensor_height` sets camera position. Best practice: match `height` to `sensor_height` for proper collision alignment. Re-record keyframes after changing.

### Change Number of Frames

```yaml
trajectory:
  num_frames: 2000  # More frames = longer video
  fps: 60           # Playback speed
```

## Troubleshooting

### Empty / Wrong Scene (No Furniture)

**Cause:** Physics disabled - articulated objects not loading

**Fix:** Check `config/default.yaml`:
```yaml
physics:
  enable_physics: true  # Must be true for Replica CAD!
```

### Rendering Looks Distorted

**Cause:** Window size doesn't match camera resolution

**Fix:** Ensure matching sizes in `config/default.yaml`:
```yaml
camera:
  width: 640
  height: 480
trajectory:
  interactive_window_size: [480, 640]  # [height, width]
```

### GUI Not Showing

**Checklist:**
1. ✅ VcXsrv running (check system tray)
2. ✅ "Disable access control" enabled in XLaunch
3. ✅ `echo $DISPLAY` shows `host.docker.internal:0.0` inside container
4. ✅ Windows Firewall allows VcXsrv

## Docker Commands Reference

```bash
# Start/stop
docker-compose up -d
docker-compose down
docker-compose restart

# Access container
docker exec -it habitat-sim-gui bash

# Run script with conda
docker exec habitat-sim-gui bash -c "source activate habitat && python script.py"

# View logs
docker-compose logs -f
```

## Output Format

### Keyframes JSON
```json
[
  {
    "timestamp": 1234567890.123,
    "position": [x, y, z],
    "rotation": [w, x, y, z],  // quaternion
    "index": 0
  }
]
```

### Trajectory File (traj.txt)
Replica dataset format: 4x4 transformation matrices (one per line, no comments).
Each line contains 16 space-separated values in row-major order:
```
R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz 0 0 0 1
```

Example:
```
9.062491181555123454e-01 -2.954311239679592860e-01 3.023788796086531172e-01 -3.569159214564542326e-01 -4.227440547687880690e-01 -6.333245673155291078e-01 6.482186796076164770e-01 -6.602722315763628336e-01 8.759610522377010340e-17 -7.152764804085384176e-01 -6.988415818870352680e-01 8.192365926179191460e-01 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
9.117974242635417115e-01 -2.821317573836910064e-01 2.983741419459151611e-01 -3.392391072404580266e-01 -4.106403013665013702e-01 -6.264533920059638383e-01 6.625184454321639826e-01 -6.529105261083280043e-01 8.898370098269448719e-17 -7.266070596407748772e-01 -6.870532591292960456e-01 8.089290835703050186e-01 0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
...
```

### Camera Parameters (cam_params.json)
```json
{
  "camera": {
    "w": 1200,
    "h": 680,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,
    "scale": 6553.5
  }
}
```

### Depth Maps
- 16-bit PNG format
- Depth in millimeters
- Scale factor: 6553.5 (1m = 6553.5 units)
- To convert to meters: `depth_meters = pixel_value / 6553.5`

## System Requirements

**Minimum:**
- Windows 10/11
- 8GB RAM
- 20GB disk space

**Recommended:**
- 16GB+ RAM
- SSD storage
- Multi-core CPU (4+)

## Key Technical Details

### Camera Control System

The interactive recorder is based on `habitat-sim/examples/viewer.py`, ensuring:
- Smooth mouse look and keyboard navigation
- Record keyframe with `space`
- Reliable transformation matrix capture

### Trajectory Interpolation

- **Position**: Cubic spline interpolation for smooth paths
- **Rotation**: Slerp (Spherical Linear Interpolation) for quaternions
- **Head Sway**: Optional sinusoidal rotation applied after interpolation

### Physics Integration

Replica CAD scenes contain articulated objects (furniture with movable parts):
- Refrigerators with opening doors
- Kitchen cabinets and drawers
- Swinging doors
- Chests of drawers

**Critical:** `enable_physics: true` required to load these objects. Without physics, scenes appear empty. Requires Bullet physics engine (included in Docker image via `habitat-sim withbullet`).

## Resources

- **Habitat-Sim**: https://github.com/facebookresearch/habitat-sim
- **Replica CAD**: https://aihabitat.org/datasets/replica_cad/
- **Documentation**: https://aihabitat.org/docs/habitat-sim/

## License

Uses Habitat-Sim (MIT License) and Replica CAD dataset. Please cite original papers if used in research.

## Quick Reference

| Task | Command |
|------|---------|
| Start container | `docker-compose up -d` |
| Enter container | `docker exec -it habitat-sim-gui bash` |
| Activate conda | `source activate habitat` |
| Record keyframes | `python scripts/1_record.py config/default.yaml` |
| Preview trajectory | `python scripts/2_generate.py config/default.yaml --preview-only` |
| Full rendering | `python scripts/2_generate.py config/default.yaml` |
| Check physics support | `python -c "import habitat_sim; print(habitat_sim.built_with_bullet)"` |
