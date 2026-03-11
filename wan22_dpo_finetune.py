#!/usr/bin/env python3
"""
Full DPO RL Loop for Wan2.2 T2V Model

This implements Direct Preference Optimization directly on the Wan2.2 video
generation model, following the CS234 DPO pattern but adapted for diffusion models.

Key differences from reward model training:
1. Optimizes the T2V UNet directly (not a separate reward model)
2. Uses denoising error as implicit reward signal
3. Trains model to prefer generating higher-quality videos

Reference: CS234 run_dpo.py lines 268-269
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
import torch.nn.functional as F
from diffusers import WanPipeline, AutoencoderKLWan, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

class Config:
    def __init__(self, args):
        # Model settings - FIXED: Use Diffusers-compatible version
        self.wan22_model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data paths
        self.pref_data_path = args.data
        self.videos_dir = args.videos_dir
        self.output_dir = args.output_dir

        # DPO hyperparameters
        self.beta = args.beta
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.epochs
        self.gradient_accumulation_steps = args.grad_accum

        # Diffusion settings
        self.n_frames = args.n_frames
        self.num_inference_steps = args.inference_steps
        self.guidance_scale = args.guidance_scale

        # Training settings
        self.max_grad_norm = 1.0
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.seed = args.seed

        # Misc
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# Dataset
# ============================================================================

class PreferencePairDataset(Dataset):
    """Dataset for DPO training on video preference pairs."""

    def __init__(self, pref_data_path, videos_dir):
        with open(pref_data_path) as f:
            raw_data = json.load(f)

        self.pairs = []
        self.prompts = []

        for group_data in raw_data.values():
            prompt = group_data["prompt"]
            for pair in group_data["pairwise_comparisons"]:
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
                    self.prompts.append(prompt)

        print(f"Loaded {len(self.pairs)} preference pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_video_to_latent(video_path, vae, n_frames=16, device="cuda"):
    """Load video and encode to VAE latent space."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames uniformly
    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
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
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        latents = []
        for frame in video_tensor:
            latent_dist = vae.encode(frame.unsqueeze(0)).latent_dist
            latent = latent_dist.sample()
            latents.append(latent)
        latents = torch.cat(latents, dim=0)

    return latents


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "prompts": [item["prompt"] for item in batch],
        "preferred_videos": [item["preferred_video"] for item in batch],
        "rejected_videos": [item["rejected_video"] for item in batch],
    }


# ============================================================================
# DPO Loss
# ============================================================================

def compute_dpo_loss(
    policy_transformer,
    reference_transformer,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    batch,
    cfg,
):
    """
    Compute DPO loss for Wan2.2 diffusion transformer.

    Following CS234 DPO (run_dpo.py:268-269):
        logits = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
        loss = -logsigmoid(logits)

    For diffusion models, we use negative denoising error as implicit log prob:
        reward = -mse(predicted_noise, true_noise)
    """
    device = cfg.device

    # Encode text prompts
    text_inputs = tokenizer(
        batch["prompts"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    # Load and encode videos to latent space
    preferred_latents = []
    rejected_latents = []

    for pref_path, rej_path in zip(batch["preferred_videos"], batch["rejected_videos"]):
        pref_latent = load_video_to_latent(pref_path, vae, cfg.n_frames, device)
        rej_latent = load_video_to_latent(rej_path, vae, cfg.n_frames, device)
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
    # For video: we process frame by frame
    policy_pred_w = []
    policy_pred_l = []

    for t in range(cfg.n_frames):
        # Preferred
        pred_w = policy_transformer(
            noisy_latents_w[:, t],
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample
        policy_pred_w.append(pred_w)

        # Rejected
        pred_l = policy_transformer(
            noisy_latents_l[:, t],
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample
        policy_pred_l.append(pred_l)

    policy_pred_w = torch.stack(policy_pred_w, dim=1)
    policy_pred_l = torch.stack(policy_pred_l, dim=1)

    # Predict noise with reference transformer (frozen)
    with torch.no_grad():
        ref_pred_w = []
        ref_pred_l = []

        for t in range(cfg.n_frames):
            # Preferred
            pred_w = reference_transformer(
                noisy_latents_w[:, t],
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample
            ref_pred_w.append(pred_w)

            # Rejected
            pred_l = reference_transformer(
                noisy_latents_l[:, t],
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample
            ref_pred_l.append(pred_l)

        ref_pred_w = torch.stack(ref_pred_w, dim=1)
        ref_pred_l = torch.stack(ref_pred_l, dim=1)

    # Compute implicit rewards (negative MSE)
    # Lower denoising error = higher reward
    policy_reward_w = -F.mse_loss(policy_pred_w, noise_w, reduction="none").mean(dim=[1, 2, 3, 4])
    policy_reward_l = -F.mse_loss(policy_pred_l, noise_l, reduction="none").mean(dim=[1, 2, 3, 4])
    ref_reward_w = -F.mse_loss(ref_pred_w, noise_w, reduction="none").mean(dim=[1, 2, 3, 4])
    ref_reward_l = -F.mse_loss(ref_pred_l, noise_l, reduction="none").mean(dim=[1, 2, 3, 4])

    # DPO loss (same structure as CS234)
    logits = cfg.beta * ((policy_reward_w - ref_reward_w) - (policy_reward_l - ref_reward_l))
    loss = -F.logsigmoid(logits).mean()

    # Compute accuracy (how often policy prefers preferred > rejected)
    with torch.no_grad():
        accuracy = (logits > 0).float().mean().item()
        reward_margin = logits.mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "reward_margin": reward_margin,
    }

    return loss, metrics


# ============================================================================
# Training Loop
# ============================================================================

def train_dpo(cfg):
    """Full DPO training loop for Wan2.2 T2V model."""

    print("=" * 70)
    print("  DPO Fine-tuning for Wan2.2 T2V Model")
    print("=" * 70)
    print(f"Device: {cfg.device}")
    print(f"Beta: {cfg.beta}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.num_epochs}")
    print()

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load Wan2.2 pipeline (using correct WanPipeline and AutoencoderKLWan)
    print("Loading Wan2.2 T2V model...")

    # Load VAE separately (float32 for stability)
    vae = AutoencoderKLWan.from_pretrained(
        cfg.wan22_model,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    # Load full pipeline
    pipe = WanPipeline.from_pretrained(
        cfg.wan22_model,
        vae=vae,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for MoE model
    )
    pipe.to(cfg.device)

    # Extract components (Wan2.2 uses transformer, not unet)
    policy_transformer = pipe.transformer
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Create frozen reference copy of transformer
    print("Creating reference transformer (frozen)...")
    reference_transformer = copy.deepcopy(policy_transformer)
    reference_transformer.requires_grad_(False)
    reference_transformer.eval()

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Only train policy transformer
    policy_transformer.train()
    policy_transformer.requires_grad_(True)

    # Optimizer (use AdamW8bit for memory efficiency if available)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            policy_transformer.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = torch.optim.AdamW(
            policy_transformer.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        print("Using standard AdamW optimizer")

    # Load dataset
    print("Loading preference data...")
    dataset = PreferencePairDataset(cfg.pref_data_path, cfg.videos_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Training loop
    print("\nStarting DPO training...")
    print("=" * 70)

    global_step = 0
    best_accuracy = 0.0

    for epoch in range(cfg.num_epochs):
        epoch_metrics = defaultdict(list)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Compute DPO loss
            loss, metrics = compute_dpo_loss(
                policy_transformer,
                reference_transformer,
                vae,
                text_encoder,
                tokenizer,
                noise_scheduler,
                batch,
                cfg,
            )

            # Backward pass
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # Update weights
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_transformer.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Log metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.3f}",
                "margin": f"{metrics['reward_margin']:.3f}",
            })

            # Save checkpoint
            if global_step > 0 and global_step % cfg.save_every == 0:
                save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                policy_transformer.save_pretrained(save_path)
                print(f"\nSaved checkpoint to {save_path}")

        # Epoch summary
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_acc = np.mean(epoch_metrics["accuracy"])
        avg_margin = np.mean(epoch_metrics["reward_margin"])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.3f}")
        print(f"  Reward Margin: {avg_margin:.3f}")

        # Save best model
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            save_path = os.path.join(cfg.output_dir, "best_model")
            policy_transformer.save_pretrained(save_path)
            print(f"  ✓ New best model saved (accuracy: {avg_acc:.3f})")

        print()

    # Final save
    final_path = os.path.join(cfg.output_dir, "final_model")
    policy_transformer.save_pretrained(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    print(f"Best accuracy: {best_accuracy:.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning for Wan2.2 T2V")

    # Data
    parser.add_argument("--data", type=str, required=True,
                       help="Path to pairwise preference JSON")
    parser.add_argument("--videos-dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="wan22_dpo_output",
                       help="Output directory for checkpoints")

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO temperature (controls strength of preference signal)")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (start small due to memory)")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")

    # Model settings
    parser.add_argument("--n-frames", type=int, default=8,
                       help="Number of frames per video (reduce for memory)")
    parser.add_argument("--inference-steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")

    # Training settings
    parser.add_argument("--save-every", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()
    cfg = Config(args)

    train_dpo(cfg)


if __name__ == "__main__":
    main()
