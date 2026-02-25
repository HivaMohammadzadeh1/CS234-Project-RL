# Video Generation using Wan2.2 Diffusion Model

This folder contains scripts and configuration files for generating physics simulation videos using the **Wan2.2 T2V-A14B** text-to-video diffusion model via the fal.ai API.

## Overview

The video generation system uses a state-of-the-art diffusion model to create realistic physics simulations from text prompts. This approach was used to generate all videos in our dataset for evaluating video generation models on physics understanding.

### Model Details

- **Model**: Wan2.2 T2V-A14B (Text-to-Video, 14 Billion parameters)
- **Platform**: fal.ai API (cloud-based generation, no local GPU required)
- **Resolution**: 480p, 580p, or 720p
- **Frame Count**: 17-161 frames (default: 81 frames)
- **Aspect Ratios**: 16:9 or 9:16

## Setup

### 1. Install Dependencies

```bash
pip install fal-client requests
```

### 2. Get API Key

1. Sign up at [fal.ai](https://fal.ai)
2. Get your API key from [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)
3. Set the environment variable:

```bash
export FAL_KEY="your-api-key-here"
```

## Usage

### Basic Generation

Generate videos using default example prompts:

```bash
python batch_generate_api.py
```

### Generate from Text File

Use a text file with one prompt per line:

```bash
python batch_generate_api.py --prompts_file my_prompts.txt
```

### Generate from JSON Configuration

Use a JSON file with per-prompt configurations (including seeds, resolution, etc.):

```bash
python batch_generate_api.py --prompts_json prompts_8x.json
```

### Advanced Options

```bash
python batch_generate_api.py \
  --prompts_file my_prompts.txt \
  --output_dir ./generated_videos \
  --resolution 720p \
  --num_frames 81 \
  --num_inference_steps 27 \
  --guidance_scale 3.5 \
  --aspect_ratio 16:9
```

#### Parameters

- `--prompts_file`: Text file with one prompt per line
- `--prompts_json`: JSON file with prompt configurations
- `--output_dir`: Directory to save generated videos (default: ./generated_videos)
- `--resolution`: Video resolution (480p, 580p, 720p)
- `--num_frames`: Number of frames (17-161, default: 81)
- `--num_inference_steps`: Sampling steps (higher = better quality but slower, default: 27)
- `--guidance_scale`: Classifier-free guidance scale (default: 3.5)
- `--aspect_ratio`: Aspect ratio (16:9 or 9:16)
- `--seed`: Random seed for reproducibility

## Files in This Directory

### Scripts

- **`batch_generate_api.py`**: Main script for batch video generation via fal.ai API
  - Handles prompt loading from text or JSON files
  - Manages API calls with proper error handling
  - Downloads generated videos automatically
  - Creates manifest.json with generation metadata

### Configuration Files

- **`config.json`**: Example configuration file with model parameters
- **`my_prompts.txt`**: 60 example prompts for physics scenarios:
  - 20 bouncing ball scenarios (various objects, heights, surfaces)
  - 20 pendulum motion scenarios (different lengths, materials, angles)
  - 20 inclined plane scenarios (boxes sliding down ramps at various angles)

- **`prompts_8x.json`**: JSON configuration with 8 variations per prompt
  - Includes specific random seeds for reproducibility
  - Used to generate multiple videos per scenario for evaluation

## Prompt Engineering Tips

For best results generating physics videos:

1. **Be Specific**: Include exact measurements (e.g., "2 meters", "30 degrees")
2. **Specify Camera**: Always mention "static camera", "side view", "fixed shot"
3. **Describe Motion**: Clearly state the expected physics behavior
4. **Lighting**: Mention lighting conditions for consistency
5. **Background**: Specify clean or simple backgrounds

### Example Prompt Structure

```
[Object] [action] from [height/angle] onto [surface], [expected behavior].
[Camera specification], [lighting], [duration].
```

**Good Example:**
```
A tennis ball falling from 2 meters onto a hard concrete floor and bouncing
several times. Static camera, side view, well lit, 5-second video.
```

**Poor Example:**
```
Ball bouncing
```

## Physics Scenarios Covered

Our prompt collection covers three main physics concepts:

### 1. Bouncing/Free Fall (Prompts 1-20)
- Various objects: tennis balls, basketballs, rubber balls, golf balls
- Different heights: 1-3 meters
- Multiple surface types: concrete, wood, marble, grass
- Tests: gravity, elastic collisions, energy dissipation

### 2. Pendulum Motion (Prompts 21-40)
- Different pendulum lengths: 0.5-2 meters
- Various bob materials: metal, wood, brass
- Different release angles: 30-60 degrees
- Tests: periodic motion, conservation of energy, damping

### 3. Inclined Planes (Prompts 41-60)
- Various angles: 5-60 degrees
- Different objects: wooden boxes, metal blocks, cardboard
- Multiple materials: smooth, friction surfaces
- Tests: acceleration on inclines, friction effects

## Output Format

Generated videos are saved with descriptive filenames:

```
{index:03d}_{prompt_snippet}_{timestamp}.mp4
```

A `manifest.json` file is created with metadata:

```json
{
  "index": 1,
  "prompt": "Full prompt text...",
  "resolution": "720p",
  "seed": 123456789,
  "num_frames": 81,
  "num_inference_steps": 27,
  "file": "./generated_videos/001_A_tennis_ball_falling_20260219_131536.mp4",
  "video_url": "https://...",
  "generation_time_s": 45.2
}
```

## Performance Notes

- **Generation Time**: ~30-60 seconds per video (cloud-based)
- **Cost**: Varies based on resolution and frame count (see fal.ai pricing)
- **No Local GPU Required**: All processing happens in the cloud
- **Batch Processing**: Script handles multiple prompts sequentially

## Reproducibility

To reproduce exact videos:
1. Use the same prompt text
2. Use the same seed value (see `prompts_8x.json` for seeds used in our dataset)
3. Use the same model parameters (resolution, frames, inference steps)

## Troubleshooting

### Common Issues

1. **"FAL_KEY environment variable not set"**
   - Solution: Export your API key: `export FAL_KEY="your-key"`

2. **"fal-client not installed"**
   - Solution: `pip install fal-client`

3. **Rate Limiting**
   - Solution: Add delays between generations or upgrade fal.ai plan

4. **Out of Memory Errors**
   - Solution: Reduce resolution or number of frames

## Related Files

- **Full Dataset**: https://huggingface.co/datasets/hivamoh/wan22-physics-videos
- **Ranking Tool**: See `../video_rater_server.py` and `../video_rater.html`
- **Sample Videos**: See `../sample_videos/` directory

## Citation

If you use this generation pipeline or prompts in your research, please cite:

```
CS234 Project: Physics Video Generation and Ranking
Using Wan2.2 T2V-A14B via fal.ai
https://github.com/HivaMohammadzadeh1/CS234-Project
```

## License

This code is provided for research and educational purposes.
