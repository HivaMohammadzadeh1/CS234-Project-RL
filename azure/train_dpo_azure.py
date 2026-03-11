#!/usr/bin/env python3
"""
Azure ML Training Script for Wan2.2 DPO Fine-tuning

This script runs on Azure ML compute and handles:
- Data downloading from Azure Blob Storage or HuggingFace
- Model training with DPO
- Checkpoint saving to Azure ML outputs
- Logging metrics to Azure ML
"""

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import WanPipeline, AutoencoderKLWan, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Weights & Biases import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not installed. Install with: pip install wandb")

# Azure ML imports
try:
    from azureml.core import Run
    run = Run.get_context()
    AZURE_ML = True
except Exception:
    run = None
    AZURE_ML = False
    print("Not running in Azure ML context, logging locally")


class TextEmbeddingProjector(nn.Module):
    """Projects Qwen3-VL text embeddings to match Wan2.2 transformer's expected dimension."""

    def __init__(self, qwen_dim=896, wan_dim=4096):  # Qwen3-VL-2B has 896 dim, Wan2.2 typically uses 4096
        super().__init__()
        self.projection = nn.Linear(qwen_dim, wan_dim)

    def forward(self, x):
        return self.projection(x)


def log_metric(key, value, step=None):
    """Log metrics to Azure ML, W&B, or print locally."""
    # Log to Azure ML
    if AZURE_ML and run:
        run.log(key, value)
        if step is not None:
            run.log(f"{key}_step", step)
    else:
        print(f"Metric: {key} = {value}" + (f" (step {step})" if step else ""))

    # Log to W&B
    if WANDB_AVAILABLE and wandb.run is not None:
        if step is not None:
            wandb.log({key: value, "step": step})
        else:
            wandb.log({key: value})


def log_image(name, path):
    """Log image to Azure ML."""
    if AZURE_ML and run:
        run.log_image(name, path=path)


class PreferencePairDataset(Dataset):
    """Dataset for DPO training on video preference pairs."""

    def __init__(self, pref_data_path, videos_dir, filter_indices=None):
        print(f"\n{'='*70}")
        print("  Loading Preference Dataset")
        print(f"{'='*70}")
        print(f"Preference file (raw): {pref_data_path}")
        print(f"Preference file type: {type(pref_data_path)}")
        print(f"Videos directory (raw): {videos_dir}")
        print(f"Videos directory type: {type(videos_dir)}")

        # Convert to string if it's not already
        pref_data_path = str(pref_data_path)
        videos_dir = str(videos_dir)

        print(f"Preference file (converted): {pref_data_path}")
        print(f"Videos directory (converted): {videos_dir}")

        # Handle Azure ML datastore URIs
        # Azure ML v2 SDK auto-resolves azureml:// URIs, but v1 SDK doesn't
        if pref_data_path.startswith('azureml://'):
            print(f"⚠ Detected Azure ML datastore URI - this won't work with direct file access")
            print(f"  ERROR: Please use actual file paths or mount the datastore properly")
            print(f"  The azureml:// URI format is for Azure ML v2 SDK")
            print(f"  For v1 SDK, data should be mounted/downloaded to actual paths first")

            raise ValueError(
                f"Cannot use azureml:// datastore URI directly: {pref_data_path}\n"
                f"This training script expects actual file paths.\n"
                f"Please update submit_job.py to properly mount/download data using Dataset API."
            )

        if videos_dir.startswith('azureml://'):
            print(f"⚠ Detected Azure ML datastore URI for videos - this won't work")
            raise ValueError(
                f"Cannot use azureml:// datastore URI directly: {videos_dir}\n"
                f"This training script expects actual file paths.\n"
                f"Please update submit_job.py to properly mount/download data using Dataset API."
            )

        # Handle Azure ML named input paths
        # When using .as_download() or .as_mount(), Azure ML may create special directory structures
        # The path might be: /tmp/azureml_runs/<run_id>/pref_data/<actual_file>
        # Or for mount: /tmp/azureml_runs/<run_id>/videos/<actual_videos>

        # Handle preference file path
        if os.path.isdir(pref_data_path):
            # It's a directory - find the JSON file
            print(f"pref_data_path is a directory, searching for JSON files...")

            # First check if it's a nested Azure ML structure
            # Look for the file recursively (in case of nested directories)
            json_files = []
            for root, dirs, files in os.walk(pref_data_path):
                json_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])

            if json_files:
                pref_data_path = json_files[0]  # Use first JSON file found
                print(f"Found JSON file: {pref_data_path}")
            else:
                raise FileNotFoundError(f"No JSON file found in directory: {pref_data_path}")

        # Debug: List current directory contents
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"Contents of current directory:")
        try:
            for item in os.listdir('.'):
                item_path = os.path.join('.', item)
                if os.path.isdir(item_path):
                    print(f"  [DIR]  {item}")
                else:
                    print(f"  [FILE] {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")

        # Handle videos directory path
        print(f"\nChecking videos directory: {videos_dir}")
        print(f"videos_dir exists: {os.path.exists(videos_dir)}")
        print(f"videos_dir is directory: {os.path.isdir(videos_dir)}")

        if not os.path.exists(videos_dir):
            # Path doesn't exist - might be a mount that hasn't completed or wrong path
            # Try to find videos in common locations
            print(f"⚠ videos_dir path doesn't exist, searching for videos...")
            search_paths = [
                '.',
                'videos',
                'wan22_dpo_data/videos',
                'videos_data',
                'videos_data/videos',
                'videos_data/wan22_dpo_data/videos',
            ]
            found_videos = False
            for search_path in search_paths:
                if os.path.isdir(search_path):
                    test_files = [f for f in os.listdir(search_path) if f.endswith(('.mp4', '.avi', '.mov'))]
                    if test_files:
                        videos_dir = search_path
                        print(f"✓ Found videos in: {videos_dir}")
                        found_videos = True
                        break

            if not found_videos:
                raise FileNotFoundError(
                    f"Videos directory not found. Searched in: {search_paths}\n"
                    f"Current directory contents listed above."
                )

        elif os.path.isdir(videos_dir):
            # Path exists - check if it has videos or needs to search subdirectories
            video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            if not video_files:
                # Might be a parent directory, look for a videos subdirectory
                print(f"No videos in {videos_dir}, searching subdirectories...")
                possible_dirs = [
                    os.path.join(videos_dir, 'videos'),
                    os.path.join(videos_dir, 'wan22_dpo_data'),
                    os.path.join(videos_dir, 'wan22_dpo_data', 'videos'),
                ]
                found = False
                for possible_dir in possible_dirs:
                    if os.path.isdir(possible_dir):
                        test_files = [f for f in os.listdir(possible_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                        if test_files:
                            videos_dir = possible_dir
                            print(f"✓ Found videos directory: {videos_dir}")
                            found = True
                            break

                if not found:
                    raise FileNotFoundError(
                        f"No videos found in {videos_dir} or its subdirectories.\n"
                        f"Searched: {possible_dirs}"
                    )

        # Final validation
        if not os.path.exists(pref_data_path):
            raise FileNotFoundError(f"Preference file not found: {pref_data_path}")

        if not os.path.exists(videos_dir):
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

        with open(pref_data_path) as f:
            raw_data = json.load(f)

        print(f"Found {len(raw_data)} prompt groups in preference file")

        # Count available videos
        video_files = []
        if os.path.isdir(videos_dir):
            video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Found {len(video_files)} video files in {videos_dir}")
        if len(video_files) > 0:
            print(f"Example video files: {video_files[:3]}")

        self.pairs = []
        missing_videos = []
        total_pairs_checked = 0

        for group_data in raw_data.values():
            prompt = group_data["prompt"]
            for pair in group_data["pairwise_comparisons"]:
                total_pairs_checked += 1

                if pair.get("tie"):
                    continue

                preferred_path = os.path.join(videos_dir, pair["preferred"])
                rejected_path = os.path.join(videos_dir, pair["rejected"])

                if os.path.exists(preferred_path) and os.path.exists(rejected_path):
                    self.pairs.append({
                        "prompt": prompt,
                        "preferred_video": preferred_path,
                        "rejected_video": rejected_path,
                    })
                else:
                    # Track missing videos
                    if not os.path.exists(preferred_path):
                        missing_videos.append(pair["preferred"])
                    if not os.path.exists(rejected_path):
                        missing_videos.append(pair["rejected"])

        # Filter by indices if specified
        if filter_indices is not None:
            print(f"\nFiltering dataset by {len(filter_indices)} indices...")
            filter_set = set(int(idx) for idx in filter_indices)
            filtered_pairs = []

            for idx, pair in enumerate(self.pairs):
                # Extract video index from filename (e.g., "001_..." -> 1)
                preferred_name = os.path.basename(pair["preferred_video"])
                video_idx = int(preferred_name.split("_")[0])

                if video_idx in filter_set:
                    filtered_pairs.append(pair)

            print(f"  Pairs before filtering: {len(self.pairs)}")
            print(f"  Pairs after filtering: {len(filtered_pairs)}")
            self.pairs = filtered_pairs

        print(f"\nDataset Statistics:")
        print(f"  Total pairs in JSON: {total_pairs_checked}")
        print(f"  Valid pairs (both videos exist): {len(self.pairs)}")
        print(f"  Missing video files: {len(set(missing_videos))}")

        if len(self.pairs) == 0 and len(missing_videos) > 0:
            print(f"\n⚠ WARNING: No valid pairs found!")
            print(f"Example missing videos: {list(set(missing_videos))[:5]}")
            print(f"\nThis usually means:")
            print(f"  1. The videos_dir path is incorrect")
            print(f"  2. The video files haven't been uploaded/downloaded")
            print(f"  3. The video filenames in the JSON don't match the actual files")

        print(f"{'='*70}\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "prompts": [item["prompt"] for item in batch],
        "preferred_videos": [item["preferred_video"] for item in batch],
        "rejected_videos": [item["rejected_video"] for item in batch],
    }


def load_video_to_latent(video_path, vae, n_frames=16, device="cuda", reference_image=None):
    """
    Load video and encode to VAE latent space.

    For TI2V models, reference_image can be provided to prepend to the video sequence.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # For TI2V models with reference image, we need n_frames total including the reference
    # So sample (n_frames - 1) frames from the video
    frames_to_sample = n_frames - 1 if reference_image is not None else n_frames

    # Sample frames uniformly
    if total_frames < frames_to_sample:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)

    frames = []

    # If TI2V with reference image, add it as the first frame
    if reference_image is not None:
        # reference_image should be a numpy array [H, W, C]
        ref_frame = cv2.resize(reference_image, (256, 256))
        frames.append(ref_frame)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame)
    cap.release()

    # Pad if needed
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((256, 256, 3), dtype=np.uint8))

    # Convert to tensor and normalize
    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()
    video_tensor = (video_tensor / 127.5) - 1.0  # [-1, 1]

    # Encode to latent space
    # video_tensor shape: [n_frames, C, H, W]
    # VAE expects: [batch, C, n_frames, H, W]
    video_tensor = video_tensor.to(device)

    # Rearrange to VAE's expected format: [1, C, n_frames, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, n_frames, H, W]

    with torch.no_grad():
        latent_dist = vae.encode(video_tensor).latent_dist
        latents = latent_dist.sample()  # [1, latent_C, n_frames, latent_H, latent_W]

    # Remove batch dimension: [latent_C, n_frames, latent_H, latent_W]
    # Then rearrange to: [n_frames, latent_C, latent_H, latent_W]
    latents = latents.squeeze(0).permute(1, 0, 2, 3)

    return latents


def compute_dpo_loss(
    policy_transformer,
    reference_transformer,
    vae,
    text_encoder,
    tokenizer,
    text_projector,
    noise_scheduler,
    batch,
    args,
    device,
):
    """Compute DPO loss for Wan2.2 diffusion transformer."""

    # Encode text prompts using Qwen3-VL
    if isinstance(text_encoder, Qwen3VLForConditionalGeneration):
        # Use Qwen3-VL processor for text encoding
        text_inputs = tokenizer(
            text=batch["prompts"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # Get text embeddings from Qwen3-VL
            # Qwen3-VL returns hidden states, we use the last hidden state
            outputs = text_encoder(**text_inputs, output_hidden_states=True)
            # Use the last hidden state as text embeddings
            qwen_embeddings = outputs.hidden_states[-1]  # [batch_size, seq_len, qwen_dim]

        # Project to Wan2.2's expected dimension
        # text_projector: Linear(qwen_dim, wan_dim)
        # Reshape for projection: [batch_size * seq_len, qwen_dim]
        batch_size, seq_len, qwen_dim = qwen_embeddings.shape
        qwen_embeddings_flat = qwen_embeddings.reshape(batch_size * seq_len, qwen_dim)

        # Project and reshape back: [batch_size, seq_len, wan_dim]
        text_embeddings = text_projector(qwen_embeddings_flat)
        text_embeddings = text_embeddings.reshape(batch_size, seq_len, -1)
    else:
        # Use original text encoder (CLIP/T5)
        text_inputs = tokenizer(
            batch["prompts"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    # Load and encode videos to latent space
    # For TI2V models, extract first frame and prepend to video sequence
    preferred_latents = []
    rejected_latents = []

    # Check if model is TI2V (needs reference images)
    is_ti2v_model = "TI2V" in args.model or "ti2v" in args.model.lower()

    if is_ti2v_model:
        print("TI2V model detected - will use first frame as reference image")

    for pref_path, rej_path in zip(batch["preferred_videos"], batch["rejected_videos"]):
        if is_ti2v_model:
            # For TI2V: Extract first frame from each video to use as reference
            # Open video to get first frame
            cap = cv2.VideoCapture(pref_path)
            ret, pref_ref_frame = cap.read()
            cap.release()
            if ret:
                pref_ref_frame = cv2.cvtColor(pref_ref_frame, cv2.COLOR_BGR2RGB)
            else:
                pref_ref_frame = np.zeros((256, 256, 3), dtype=np.uint8)

            cap = cv2.VideoCapture(rej_path)
            ret, rej_ref_frame = cap.read()
            cap.release()
            if ret:
                rej_ref_frame = cv2.cvtColor(rej_ref_frame, cv2.COLOR_BGR2RGB)
            else:
                rej_ref_frame = np.zeros((256, 256, 3), dtype=np.uint8)

            # Load video with reference frame prepended
            pref_latent = load_video_to_latent(pref_path, vae, args.n_frames, device, reference_image=pref_ref_frame)
            rej_latent = load_video_to_latent(rej_path, vae, args.n_frames, device, reference_image=rej_ref_frame)
        else:
            # T2V model - no reference images needed
            pref_latent = load_video_to_latent(pref_path, vae, args.n_frames, device)
            rej_latent = load_video_to_latent(rej_path, vae, args.n_frames, device)

        preferred_latents.append(pref_latent)
        rejected_latents.append(rej_latent)

    preferred_latents = torch.stack(preferred_latents)  # (B, T, C, H, W)
    rejected_latents = torch.stack(rejected_latents)

    # Sample random timestep for each video
    bsz = preferred_latents.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()

    # Add noise to latents (diffusion forward process)
    noise_w = torch.randn_like(preferred_latents)
    noise_l = torch.randn_like(rejected_latents)

    noisy_latents_w = noise_scheduler.add_noise(preferred_latents, noise_w, timesteps)
    noisy_latents_l = noise_scheduler.add_noise(rejected_latents, noise_l, timesteps)

    # Predict noise with policy transformer
    # noisy_latents shape: [batch, n_frames, C, H, W]
    # Transformer expects: [batch, C, n_frames, H, W] in bfloat16

    # Rearrange for transformer: [batch, C, n_frames, H, W]
    noisy_latents_w_transformed = noisy_latents_w.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
    noisy_latents_l_transformed = noisy_latents_l.permute(0, 2, 1, 3, 4).to(torch.bfloat16)

    # Pass entire video through transformer at once
    policy_pred_w_output = policy_transformer(
        noisy_latents_w_transformed,
        timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    policy_pred_l_output = policy_transformer(
        noisy_latents_l_transformed,
        timesteps,
        encoder_hidden_states=text_embeddings,
    ).sample

    # Rearrange back to [batch, n_frames, C, H, W]
    policy_pred_w = policy_pred_w_output.permute(0, 2, 1, 3, 4)
    policy_pred_l = policy_pred_l_output.permute(0, 2, 1, 3, 4)

    # Predict noise with reference transformer (frozen)
    with torch.no_grad():
        # Check if reference model is on CPU (memory saving mode)
        ref_device = next(reference_transformer.parameters()).device

        # Rearrange for transformer: [batch, C, n_frames, H, W] and convert to bfloat16
        noisy_latents_w_ref = noisy_latents_w.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
        noisy_latents_l_ref = noisy_latents_l.permute(0, 2, 1, 3, 4).to(torch.bfloat16)

        if ref_device.type == 'cpu':
            # Move data to CPU for reference model (dtype already bfloat16)
            noisy_latents_w_cpu = noisy_latents_w_ref.cpu()
            noisy_latents_l_cpu = noisy_latents_l_ref.cpu()
            timesteps_cpu = timesteps.cpu()
            text_emb_cpu = text_embeddings.cpu()

            ref_pred_w_output = reference_transformer(
                noisy_latents_w_cpu, timesteps_cpu,
                encoder_hidden_states=text_emb_cpu,
            ).sample.to(device)

            ref_pred_l_output = reference_transformer(
                noisy_latents_l_cpu, timesteps_cpu,
                encoder_hidden_states=text_emb_cpu,
            ).sample.to(device)
        else:
            ref_pred_w_output = reference_transformer(
                noisy_latents_w_ref, timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            ref_pred_l_output = reference_transformer(
                noisy_latents_l_ref, timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

        # Rearrange back to [batch, n_frames, C, H, W]
        ref_pred_w = ref_pred_w_output.permute(0, 2, 1, 3, 4)
        ref_pred_l = ref_pred_l_output.permute(0, 2, 1, 3, 4)

    # Compute implicit rewards (negative MSE)
    # Convert noise to bfloat16 to match prediction dtype
    noise_w_bf16 = noise_w.to(torch.bfloat16)
    noise_l_bf16 = noise_l.to(torch.bfloat16)

    policy_reward_w = -F.mse_loss(policy_pred_w, noise_w_bf16, reduction="none").mean(dim=[1, 2, 3, 4])
    policy_reward_l = -F.mse_loss(policy_pred_l, noise_l_bf16, reduction="none").mean(dim=[1, 2, 3, 4])
    ref_reward_w = -F.mse_loss(ref_pred_w, noise_w_bf16, reduction="none").mean(dim=[1, 2, 3, 4])
    ref_reward_l = -F.mse_loss(ref_pred_l, noise_l_bf16, reduction="none").mean(dim=[1, 2, 3, 4])

    # DPO loss (same structure as CS234)
    logits = args.beta * ((policy_reward_w - ref_reward_w) - (policy_reward_l - ref_reward_l))
    loss = -F.logsigmoid(logits).mean()

    # Compute metrics
    with torch.no_grad():
        accuracy = (logits > 0).float().mean().item()
        reward_margin = logits.mean().item()

        # Log individual reward components for debugging
        policy_reward_w_mean = policy_reward_w.mean().item()
        policy_reward_l_mean = policy_reward_l.mean().item()
        ref_reward_w_mean = ref_reward_w.mean().item()
        ref_reward_l_mean = ref_reward_l.mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "reward_margin": reward_margin,
        "policy_reward_w": policy_reward_w_mean,
        "policy_reward_l": policy_reward_l_mean,
        "ref_reward_w": ref_reward_w_mean,
        "ref_reward_l": ref_reward_l_mean,
    }

    return loss, metrics


def train_dpo(args):
    """Main DPO training loop."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # CRITICAL: Set HuggingFace cache to /tmp to avoid filling root disk
    # This must be done BEFORE any model imports/loads
    print("\n" + "=" * 70)
    print("  Configuring HuggingFace Cache Location")
    print("=" * 70)

    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/huggingface_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
    os.environ["HF_HUB_CACHE"] = "/tmp/huggingface_cache"
    os.environ["TORCH_HOME"] = "/tmp/torch_cache"

    print(f"✓ HuggingFace cache: {os.environ['HF_HOME']}")
    print(f"✓ Torch cache: {os.environ['TORCH_HOME']}")

    # Show disk space
    import subprocess
    print("\nDisk space before model loading:")
    try:
        df_output = subprocess.check_output(["df", "-h", "/", "/tmp"], text=True)
        print(df_output)
    except:
        pass

    # Check if model is already cached locally in /tmp
    print("\nChecking for cached models...")
    print("NOTE: DPO training requires Diffusers format (T2V or TI2V)")
    print("  (has vae/, text_encoder/, transformer/ subfolders)")
    print("  Cannot use the original non-Diffusers formats")

    # Map model names to cache paths
    model_cache_map = {
        "Wan-AI/Wan2.2-T2V-A14B": "/tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B/snapshots",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "/tmp/huggingface_cache/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "/tmp/huggingface_cache/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots",
    }

    local_model_path = None

    # First check for exact match
    if args.model in model_cache_map:
        base_path = model_cache_map[args.model]
        if os.path.exists(base_path):
            snapshots = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if snapshots:
                local_model_path = os.path.join(base_path, snapshots[0])
                print(f"✓ Found exact cached model at: {local_model_path}")

    # If not found, check all cache paths
    if not local_model_path:
        for model_name, base_path in model_cache_map.items():
            if os.path.exists(base_path):
                snapshots = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if snapshots:
                    local_model_path = os.path.join(base_path, snapshots[0])
                    print(f"✓ Found alternative cached model at: {local_model_path}")
                    print(f"  (Requested {args.model}, using cached {model_name})")
                    break

    if not local_model_path:
        print(f"⚠ No cached model found. Will download from HuggingFace to /tmp")
        print(f"  Model: {args.model}")

    # Use local path if available, otherwise download from HuggingFace
    model_path = local_model_path if local_model_path else args.model
    print(f"\nFinal model path: {model_path}")
    print("=" * 70)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("  Loading Wan2.2 T2V Model")
    print("=" * 70)

    # Try to load pipeline - it will load VAE, text encoder, and transformer
    print("Loading pipeline...")
    try:
        # First try loading as diffusers format (with subfolders)
        vae = AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir="/tmp/huggingface_cache"
        )
        print("✓ VAE loaded (diffusers format)")

        pipe = WanPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/huggingface_cache"
        )
    except (OSError, EnvironmentError) as e:
        print(f"Could not load as diffusers format: {e}")
        print("Trying to load complete pipeline...")

        # Load as complete pipeline (single model)
        pipe = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/huggingface_cache"
        )

    pipe.to(device)
    print("✓ Pipeline loaded")

    # Extract components
    policy_transformer = pipe.transformer
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Load text encoder (Qwen3-VL or default from pipeline)
    if args.text_encoder:
        print(f"\nLoading custom text encoder: {args.text_encoder}")
        if "qwen3" in args.text_encoder.lower() or "qwen" in args.text_encoder.lower():
            # Load Qwen3-VL for text encoding
            print("Loading Qwen3-VL model...")
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                args.text_encoder,
                torch_dtype=torch.bfloat16,
                cache_dir="/tmp/huggingface_cache"
            ).to(device)
            tokenizer = AutoProcessor.from_pretrained(
                args.text_encoder,
                cache_dir="/tmp/huggingface_cache"
            )
            print("✓ Qwen3-VL text encoder loaded")

            # Create projection layer to match Wan2.2's expected dimension
            # Debug: Print config attributes to understand structure
            print(f"Qwen3-VL config attributes: {dir(text_encoder.config)}")

            # For Qwen3-VL, we need the text/language model dimension, not vision
            # Check text_config first (for language model dimension)
            if hasattr(text_encoder.config, 'text_config') and hasattr(text_encoder.config.text_config, 'hidden_size'):
                qwen_dim = text_encoder.config.text_config.hidden_size
                print(f"Using config.text_config.hidden_size: {qwen_dim}")
            elif hasattr(text_encoder.config, 'hidden_size'):
                qwen_dim = text_encoder.config.hidden_size
                print(f"Using config.hidden_size: {qwen_dim}")
            elif hasattr(text_encoder.config, 'embed_dim'):
                qwen_dim = text_encoder.config.embed_dim
                print(f"Using config.embed_dim: {qwen_dim}")
            elif hasattr(text_encoder.config, 'd_model'):
                qwen_dim = text_encoder.config.d_model
                print(f"Using config.d_model: {qwen_dim}")
            else:
                # Qwen3-VL-2B default text dimension
                qwen_dim = 2048
                print(f"⚠ Could not find hidden dimension in config, using default: {qwen_dim}")

            wan_dim = policy_transformer.config.hidden_size if hasattr(policy_transformer.config, 'hidden_size') else 4096
            text_projector = TextEmbeddingProjector(qwen_dim, wan_dim).to(device=device, dtype=torch.bfloat16)
            print(f"✓ Text embedding projector created ({qwen_dim} -> {wan_dim}, dtype=bfloat16)")
        else:
            # Use custom text encoder from HuggingFace
            from transformers import AutoTokenizer, AutoModel
            text_encoder = AutoModel.from_pretrained(args.text_encoder, cache_dir="/tmp/huggingface_cache").to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, cache_dir="/tmp/huggingface_cache")
            text_projector = None  # May need to add projection layer based on dimensions
            print(f"✓ Custom text encoder loaded: {args.text_encoder}")
    else:
        # Use default text encoder from pipeline
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        text_projector = None
        print("✓ Using default text encoder from pipeline")

    # Create frozen reference copy
    # NOTE: To save GPU memory, we'll keep reference model on CPU and move to GPU only during forward pass
    print("\nCreating reference transformer (frozen)...")
    print("Loading reference model (will keep on CPU to save GPU memory)...")

    try:
        # Try loading reference transformer to CPU
        reference_pipe = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/huggingface_cache"
        )
        reference_transformer = reference_pipe.transformer
        reference_transformer.requires_grad_(False)
        reference_transformer.eval()

        # Keep on CPU to save GPU memory
        reference_transformer = reference_transformer.cpu()

        # Clean up temporary pipeline
        del reference_pipe
        torch.cuda.empty_cache()

        print("✓ Reference transformer loaded (on CPU)")
        reference_on_cpu = True

        # CRITICAL: Verify that policy and reference are actually different models
        print("\nVerifying policy and reference are separate models...")
        policy_param = next(policy_transformer.parameters())
        ref_param = next(reference_transformer.parameters())
        are_same = policy_param.data_ptr() == ref_param.data_ptr()

        if are_same:
            print("❌ ERROR: Policy and reference share the same parameters!")
            print("   This will cause loss to be stuck at log(2) = 0.6931")
            print("   DPO training cannot work with shared parameters!")
            raise RuntimeError("Policy and reference models must have separate parameters")
        else:
            print("✓ Policy and reference have separate parameters")
            # Print a few parameter values to verify they're identical initially
            print(f"  Policy param sample (GPU): {policy_param.flatten()[:5]}")
            print(f"  Reference param sample (CPU): {ref_param.flatten()[:5]}")
            # Move reference param to GPU temporarily for comparison
            ref_param_gpu = ref_param.to(policy_param.device)
            param_diff = (policy_param - ref_param_gpu).abs().max().item()
            print(f"  Max parameter difference: {param_diff:.10f}")
            if param_diff < 1e-8:
                print("  ✓ Parameters are identical (as expected at initialization)")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load separate reference model: {e}")
        print("   Falling back to shared parameters will NOT work for DPO!")
        print("   Loss will be stuck at log(2) because policy == reference")
        print("\n   Possible fixes:")
        print("   1. Check available memory (reference model needs ~10GB)")
        print("   2. Try reducing model size or using a smaller reference model")
        print("   3. Use gradient checkpointing and mixed precision")
        raise RuntimeError("Failed to load reference model - cannot continue DPO training")

    # Freeze VAE and text encoder (we only train the projector if using Qwen)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Train policy transformer and text projector (if using Qwen)
    policy_transformer.train()
    policy_transformer.requires_grad_(True)

    # Enable gradient checkpointing to reduce memory usage during backward pass
    if hasattr(policy_transformer, 'enable_gradient_checkpointing'):
        policy_transformer.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled for transformer")
    else:
        print("⚠ Gradient checkpointing not available for this model")

    trainable_params = sum(p.numel() for p in policy_transformer.parameters() if p.requires_grad)
    if text_projector is not None:
        text_projector.train()
        text_projector.requires_grad_(True)
        projector_params = sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
        trainable_params += projector_params
        print(f"Trainable parameters: {trainable_params:,} (transformer: {trainable_params - projector_params:,}, projector: {projector_params:,})")
    else:
        print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    print("\nSetting up optimizer...")
    # Collect trainable parameters
    trainable_params_list = list(policy_transformer.parameters())
    if text_projector is not None:
        trainable_params_list.extend(list(text_projector.parameters()))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params_list,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        print("✓ Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        print("✓ Using standard AdamW optimizer")

    # Load dataset
    print("\nLoading preference dataset...")
    filter_indices = getattr(args, 'filter_indices', None)
    dataset = PreferencePairDataset(args.data, args.videos_dir, filter_indices=filter_indices)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Setup learning rate scheduler
    print("\nSetting up learning rate scheduler...")
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    # Warmup scheduler (linear from 0 to lr)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1  # Minimum LR is 10% of initial LR
    )

    # Combine: warmup then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    print(f"✓ Cosine scheduler with linear warmup")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Initial LR: {args.lr}")
    print(f"  Min LR: {args.lr * 0.1}")

    # Initialize Weights & Biases
    if WANDB_AVAILABLE and args.use_wandb:
        print("\nInitializing Weights & Biases...")
        wandb_config = {
            "model": args.model,
            "text_encoder": args.text_encoder if args.text_encoder else "default",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "beta": args.beta,
            "learning_rate": args.lr,
            "n_frames": args.n_frames,
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "dataset_size": len(dataset),
        }

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
            resume="allow" if args.wandb_resume else False,
            tags=["dpo", "wan22", "video-generation"],
        )
        print(f"✓ W&B initialized: {wandb.run.name}")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run URL: {wandb.run.get_url()}")

    # Training loop
    print("\n" + "=" * 70)
    print("  Starting DPO Training")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"Beta: {args.beta}")
    print("=" * 70)

    global_step = 0
    best_accuracy = 0.0
    last_grad_norm = 0.0  # Track gradient norm for display

    for epoch in range(args.epochs):
        epoch_metrics = defaultdict(list)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            # Compute DPO loss
            loss, metrics = compute_dpo_loss(
                policy_transformer,
                reference_transformer,
                vae,
                text_encoder,
                tokenizer,
                text_projector,
                noise_scheduler,
                batch,
                args,
                device,
            )

            # Backward pass
            loss = loss / args.grad_accum

            # Clear cache before backward to reduce fragmentation
            torch.cuda.empty_cache()

            loss.backward()

            # Clear cache after backward
            torch.cuda.empty_cache()

            # Update weights
            if (step + 1) % args.grad_accum == 0:
                # Clip gradients for all trainable parameters
                params_to_clip = list(policy_transformer.parameters())
                if text_projector is not None:
                    params_to_clip.extend(list(text_projector.parameters()))

                # Compute gradient norm BEFORE clipping for debugging
                total_grad_norm = 0.0
                for p in params_to_clip:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                last_grad_norm = total_grad_norm  # Store for progress bar

                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                global_step += 1

                # Log to Azure ML and W&B
                log_metric("train_loss", metrics["loss"], step=global_step)
                log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
                log_metric("train_accuracy", metrics["accuracy"], step=global_step)
                log_metric("train_reward_margin", metrics["reward_margin"], step=global_step)
                log_metric("gradient_norm", last_grad_norm, step=global_step)

                # Log individual reward components for debugging
                log_metric("policy_reward_preferred", metrics["policy_reward_w"], step=global_step)
                log_metric("policy_reward_rejected", metrics["policy_reward_l"], step=global_step)
                log_metric("ref_reward_preferred", metrics["ref_reward_w"], step=global_step)
                log_metric("ref_reward_rejected", metrics["ref_reward_l"], step=global_step)

            # Collect metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.3f}",
                "margin": f"{metrics['reward_margin']:.3f}",
                "grad": f"{last_grad_norm:.2e}",
            })

            # Save checkpoint
            if global_step > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                policy_transformer.save_pretrained(save_path)
                if text_projector is not None:
                    torch.save(text_projector.state_dict(), os.path.join(save_path, "text_projector.pt"))
                print(f"\nSaved checkpoint to {save_path}")

        # Epoch summary
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_acc = np.mean(epoch_metrics["accuracy"])
        avg_margin = np.mean(epoch_metrics["reward_margin"])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.3f}")
        print(f"  Reward Margin: {avg_margin:.3f}")

        # Log epoch metrics
        log_metric("epoch_loss", avg_loss, step=epoch)
        log_metric("epoch_accuracy", avg_acc, step=epoch)
        log_metric("epoch_reward_margin", avg_margin, step=epoch)

        # Save best model
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            save_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            policy_transformer.save_pretrained(save_path)
            if text_projector is not None:
                torch.save(text_projector.state_dict(), os.path.join(save_path, "text_projector.pt"))
            print(f"  ✓ New best model saved (accuracy: {avg_acc:.3f})")

            log_metric("best_accuracy", best_accuracy)

    # Final save
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    policy_transformer.save_pretrained(final_path)
    if text_projector is not None:
        torch.save(text_projector.state_dict(), os.path.join(final_path, "text_projector.pt"))

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 70)

    # Log final metrics
    log_metric("final_best_accuracy", best_accuracy)


def main():
    parser = argparse.ArgumentParser(description="Azure ML DPO Training for Wan2.2")

    # Data
    parser.add_argument("--data", type=str, required=True,
                       help="Path to pairwise preference JSON")
    parser.add_argument("--videos-dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory (Azure ML will mount this)")

    # Model
    parser.add_argument("--model", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                       help="Wan2.2 model path")
    parser.add_argument("--text-encoder", type=str, default=None,
                       help="Custom text encoder model (e.g., Qwen/Qwen3-VL-2B-Instruct). If not specified, uses default from pipeline.")

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO temperature")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")

    # Training settings
    parser.add_argument("--n-frames", type=int, default=8,
                       help="Frames per video")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                       help="Number of diffusion steps (for evaluation/generation)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--save-every", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Train/test split
    parser.add_argument("--train-only", action="store_true",
                       help="Use only training video indices (exclude test set)")
    parser.add_argument("--test-indices", type=str, default="0,1,8,15,21,25,27,28,45,48,50,52",
                       help="Comma-separated list of test video indices")

    # Weights & Biases (wandb)
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="wan22-dpo",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name (auto-generated if not specified)")
    parser.add_argument("--wandb-resume", action="store_true",
                       help="Resume W&B run if it exists")

    args = parser.parse_args()

    # Parse test indices if train-only mode
    if args.train_only:
        test_indices = [int(idx.strip()) for idx in args.test_indices.split(",")]
        # Compute train indices (0-59 excluding test indices)
        all_indices = set(range(60))
        train_indices = sorted(list(all_indices - set(test_indices)))
        args.filter_indices = train_indices
        print(f"\nTrain/Test Split:")
        print(f"  Test indices: {test_indices}")
        print(f"  Train indices: {len(train_indices)} videos")
    else:
        args.filter_indices = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    train_dpo(args)


if __name__ == "__main__":
    main()
