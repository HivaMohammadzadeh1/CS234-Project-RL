# Video Ranking Tool

A simple web app to view and rank 8 videos side by side.

## Dataset

This repository includes a few sample videos in the `sample_videos/` directory for demonstration purposes.

**Full Dataset:** The complete video dataset is available on Hugging Face. You can download all generated videos from:
- https://huggingface.co/datasets/hivamoh/wan22-physics-videos

To use the full dataset, download the videos and place them in a `generated_videos/` directory, or update the paths in `video_rater_server.py` accordingly.

## Features

- Displays 8 videos in a grid (4x2 layout)
- Videos loop automatically
- Rank videos from 1 (best) to 8 (worst)
- Navigate between different groups of 8 videos
- Rankings are saved to `video_rankings.json`
- Keyboard shortcuts supported

## Usage

1. Start the server:
   ```bash
   python video_rater_server.py
   ```

2. Open your browser to: http://localhost:5001

3. Rank the videos:
   - Click on a video to select it (optional)
   - Click rank buttons (1-8) below each video, or press keys 1-8 on keyboard
   - 1 = Best, 8 = Worst
   - Each rank can only be assigned to one video

4. Navigation:
   - Use "Previous Group" / "Next Group" buttons
   - Or use Left/Right arrow keys

5. Save your rankings:
   - Click "Save Rankings" to save current group
   - Click "Save & Next Group" to save and move to next
   - Rankings are saved to `video_rankings.json`

## Keyboard Shortcuts

- `1-8`: Assign rank to selected video
- `←/→`: Navigate between groups
- Click videos to select them

## Output

Rankings are saved to `video_rankings.json` with the following structure:
```json
{
  "group_1": {
    "group": 1,
    "prompt": "The prompt text...",
    "rankings": {
      "001_video_name.mp4": 1,
      "002_video_name.mp4": 3,
      ...
    },
    "timestamp": "2024-01-01T12:00:00"
  }
}
```
