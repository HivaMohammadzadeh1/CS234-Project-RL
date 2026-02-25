#!/usr/bin/env python3
"""
Wan2.2 T2V-A14B batch video generation via fal.ai API.

No GPU required -- runs on your Mac, generates videos in the cloud.

Setup:
    pip install fal-client requests
    export FAL_KEY="your-api-key-from-https://fal.ai/dashboard/keys"

Usage:
    # With default example prompts:
    python batch_generate_api.py

    # With a text file (one prompt per line):
    python batch_generate_api.py --prompts_file my_prompts.txt

    # With a JSON file (per-prompt config):
    python batch_generate_api.py --prompts_json prompts.json

    # Lower resolution to save cost:
    python batch_generate_api.py --prompts_file my_prompts.txt --resolution 480p
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def load_prompts(prompts_file=None, prompts_json=None):
    """Load prompts from a text file (one per line), a JSON file, or return defaults."""
    if prompts_json:
        with open(prompts_json, "r") as f:
            data = json.load(f)
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append({"prompt": item})
            else:
                prompts.append(item)
        return prompts

    if prompts_file:
        with open(prompts_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return [{"prompt": line} for line in lines]

    return [
        {"prompt": "A tennis ball falling from 2 meters onto a hard concrete floor and bouncing several times. Static camera, side view, well lit, 5-second video."},
        {"prompt": "A red apple rolling down a wooden ramp and falling off the edge onto a table. Side view, natural lighting."},
        {"prompt": "A pendulum swinging back and forth in a physics lab. Close-up, smooth motion, 5-second video."},
    ]


def generate_video_fal(prompt, resolution="720p", num_frames=81, seed=None,
                       num_inference_steps=27, guidance_scale=3.5,
                       aspect_ratio="16:9", fal_key=None):
    """Call fal.ai Wan2.2 T2V-A14B endpoint and return the result."""
    import fal_client

    arguments = {
        "prompt": prompt,
        "resolution": resolution,
        "num_frames": num_frames,
        "aspect_ratio": aspect_ratio,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "enable_safety_checker": False,
        "enable_output_safety_checker": False,
    }
    if seed is not None:
        arguments["seed"] = seed

    def on_queue_update(update):
        status = getattr(update, "status", None) or type(update).__name__
        if hasattr(update, "logs") and update.logs:
            for log in update.logs:
                msg = log.get("message", str(log)) if isinstance(log, dict) else str(log)
                logging.info(f"  [fal] {msg}")

    result = fal_client.subscribe(
        "fal-ai/wan/v2.2-a14b/text-to-video",
        arguments=arguments,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result


def download_video(url, save_path):
    """Download video from URL to local file."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate videos with Wan2.2 T2V-A14B via fal.ai API"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Text file with one prompt per line.",
    )
    parser.add_argument(
        "--prompts_json",
        type=str,
        default=None,
        help="JSON file with prompt configs. List of objects with 'prompt', optional 'seed', 'resolution'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_videos",
        help="Directory to save generated videos (default: ./generated_videos).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["480p", "580p", "720p"],
        help="Default video resolution (default: 720p).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames (17-161, default: 81).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=27,
        help="Sampling steps, higher = better quality but slower (default: 27).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale (default: 3.5).",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        choices=["16:9", "9:16"],
        help="Aspect ratio (default: 16:9).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed (overridden by per-prompt seeds in JSON).",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        logging.error(
            "FAL_KEY environment variable not set.\n"
            "  1. Sign up at https://fal.ai\n"
            "  2. Get your key at https://fal.ai/dashboard/keys\n"
            "  3. Run: export FAL_KEY='your-key-here'"
        )
        sys.exit(1)

    try:
        import fal_client  # noqa: F401
    except ImportError:
        logging.error("fal-client not installed. Run: pip install fal-client")
        sys.exit(1)

    prompts = load_prompts(args.prompts_file, args.prompts_json)
    logging.info(f"Loaded {len(prompts)} prompt(s) for generation.")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    total_start = time.time()

    for i, prompt_cfg in enumerate(prompts):
        prompt = prompt_cfg["prompt"]
        resolution = prompt_cfg.get("resolution", args.resolution)
        seed = prompt_cfg.get("seed", args.seed)
        num_frames = prompt_cfg.get("num_frames", args.num_frames)
        num_inference_steps = prompt_cfg.get("num_inference_steps", args.num_inference_steps)
        guidance_scale = prompt_cfg.get("guidance_scale", args.guidance_scale)

        logging.info(f"\n{'='*60}")
        logging.info(f"[{i+1}/{len(prompts)}] Generating video...")
        logging.info(f"  Prompt     : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        logging.info(f"  Resolution : {resolution}")
        logging.info(f"  Frames     : {num_frames}")
        logging.info(f"  Steps      : {num_inference_steps}")
        logging.info(f"  Seed       : {seed or 'random'}")
        logging.info(f"{'='*60}")

        t_start = time.time()

        try:
            result = generate_video_fal(
                prompt=prompt,
                resolution=resolution,
                num_frames=num_frames,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                aspect_ratio=args.aspect_ratio,
            )
        except Exception as e:
            logging.error(f"  FAILED: {e}")
            results.append({
                "index": i + 1,
                "prompt": prompt,
                "error": str(e),
            })
            continue

        t_gen = time.time() - t_start
        video_url = result["video"]["url"]
        actual_seed = result.get("seed", "unknown")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        filename = f"{i+1:03d}_{safe_prompt}_{timestamp}.mp4"
        save_path = os.path.join(args.output_dir, filename)

        logging.info(f"  Downloading video...")
        download_video(video_url, save_path)

        logging.info(f"  Saved  : {save_path}")
        logging.info(f"  Seed   : {actual_seed}")
        logging.info(f"  Time   : {t_gen:.1f}s")

        results.append({
            "index": i + 1,
            "prompt": prompt,
            "resolution": resolution,
            "seed": actual_seed,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "file": save_path,
            "video_url": video_url,
            "generation_time_s": round(t_gen, 1),
        })

    total_time = time.time() - total_start
    successful = [r for r in results if "error" not in r]

    logging.info(f"\n{'='*60}")
    logging.info("BATCH GENERATION COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"  Videos generated : {len(successful)}/{len(prompts)}")
    if successful:
        total_gen_time = sum(r["generation_time_s"] for r in successful)
        logging.info(f"  Total gen time   : {total_gen_time:.1f}s")
    logging.info(f"  Wall clock time  : {total_time:.1f}s")
    logging.info(f"  Output directory : {os.path.abspath(args.output_dir)}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"  Manifest saved   : {manifest_path}")


if __name__ == "__main__":
    main()
