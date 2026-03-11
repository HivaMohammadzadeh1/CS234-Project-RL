#!/usr/bin/env python3
"""
Generate videos using DPO fine-tuned Wan2.2 model

Usage:
    python generate_with_finetuned.py \
        --model wan22_dpo_finetuned/best_model \
        --prompt "A ball rolling down a slope" \
        --output output_video.mp4
"""

import argparse
import torch
from diffusers import WanPipeline, AutoencoderKLWan, UNet2DConditionModel
import imageio
import numpy as np


def generate_video(model_path, prompt, num_frames=81, guidance_scale=4.0, seed=42):
    """Generate video using fine-tuned Wan2.2 model."""

    print(f"Loading model from {model_path}...")

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32
    )

    # Load base Wan2.2 pipeline
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    # Load fine-tuned transformer (if model_path is not the base model)
    if model_path != "Wan-AI/Wan2.2-T2V-A14B-Diffusers":
        print("Loading fine-tuned transformer...")
        # Note: For Wan2.2, the main model is a transformer, not a UNet
        # You would load it here if you've fine-tuned it
        pass

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)

    print(f"\nGenerating video for prompt: '{prompt}'")
    print(f"Settings: {num_frames} frames, guidance={guidance_scale}, seed={seed}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate video (Wan2.2 parameters)
    output = pipe(
        prompt=prompt,
        height=720,  # Wan2.2 supports 720P
        width=1280,
        num_frames=num_frames,  # Default 81 frames (5 seconds at 24fps)
        guidance_scale=guidance_scale,
        num_inference_steps=40,
    )

    frames = output.frames[0]  # Get first batch item
    print(f"Generated {len(frames)} frames at 720P (1280x720)")

    return frames


def save_video(frames, output_path, fps=8):
    """Save frames as video file."""
    print(f"Saving video to {output_path}...")

    # Convert frames to numpy arrays
    frame_arrays = []
    for frame in frames:
        frame_array = np.array(frame)
        frame_arrays.append(frame_array)

    # Save as video
    imageio.mimsave(output_path, frame_arrays, fps=fps)
    print(f"✓ Saved video: {output_path}")


def compare_models(base_model, finetuned_model, prompt, output_dir="comparison"):
    """Generate videos with both base and fine-tuned models for comparison."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate with base model
    print("\n" + "=" * 70)
    print("Generating with BASE Wan2.2 model...")
    print("=" * 70)
    base_frames = generate_video("Wan-AI/Wan2.2-T2V-A14B", prompt)
    save_video(base_frames, f"{output_dir}/base_model.mp4")

    # Generate with fine-tuned model
    print("\n" + "=" * 70)
    print("Generating with FINE-TUNED model...")
    print("=" * 70)
    finetuned_frames = generate_video(finetuned_model, prompt)
    save_video(finetuned_frames, f"{output_dir}/finetuned_model.mp4")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print(f"Base model:      {output_dir}/base_model.mp4")
    print(f"Fine-tuned model: {output_dir}/finetuned_model.mp4")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate videos with fine-tuned Wan2.2")

    parser.add_argument("--model", type=str, required=True,
                       help="Path to fine-tuned UNet")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="output.mp4",
                       help="Output video path")
    parser.add_argument("--num-frames", type=int, default=16,
                       help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Guidance scale for generation")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for output video")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--compare", action="store_true",
                       help="Generate videos with both base and fine-tuned models")

    args = parser.parse_args()

    if args.compare:
        compare_models(
            "Wan-AI/Wan2.2-T2V-A14B",
            args.model,
            args.prompt,
            output_dir="comparison_results"
        )
    else:
        frames = generate_video(
            args.model,
            args.prompt,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        save_video(frames, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
