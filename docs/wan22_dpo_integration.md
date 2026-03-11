# Proper DPO Integration with Wan2.2 T2V Model

## DPO vs RLHF - Key Difference

❌ **RLHF** (what I mistakenly described before):
- Generate samples → Score with reward model → Update policy with RL

✅ **DPO** (what you actually want):
- Use **existing** preference pairs directly
- No reward model needed during fine-tuning
- No online sampling during training
- Direct optimization from preferences

---

## DPO for Diffusion Models

### Core Idea

DPO treats the diffusion model as an implicit preference model. The key insight from the Diffusion-DPO paper is:

```
For language models:
  reward(x, y) ∝ log π_θ(y|x) - log π_ref(y|x)

For diffusion models:
  reward(prompt, video) ∝ -ELBO_θ(video|prompt) + ELBO_ref(video|prompt)
                        ≈ -||ε_θ - ε||² + ||ε_ref - ε||²
```

Where `ε` is the true noise, and `ε_θ`, `ε_ref` are predicted by policy and reference models.

---

## Implementation

### Step 1: Prepare Preference Pairs

You already have this from your dataset!

```python
# Your existing preference data structure
preference_pairs = [
    {
        "prompt": "A ball rolling down a hill",
        "preferred_video": "video_A.mp4",  # Higher human ranking
        "rejected_video": "video_B.mp4",   # Lower human ranking
    },
    ...
]
```

### Step 2: Load Wan2.2 Model + Create Reference

```python
from diffusers import DiffusionPipeline
import torch
import copy

# Load Wan2.2 T2V model
policy_pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B",
    torch_dtype=torch.float16,
    use_safetensors=True
)
policy_pipe.to("cuda")

# Create frozen reference model
reference_pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B",
    torch_dtype=torch.float16,
    use_safetensors=True
)
reference_pipe.to("cuda")
reference_pipe.unet.eval()
for p in reference_pipe.unet.parameters():
    p.requires_grad = False

# Only train the policy UNet
optimizer = torch.optim.AdamW(policy_pipe.unet.parameters(), lr=1e-6)
```

### Step 3: DPO Training Loop (Core Implementation)

```python
def dpo_train_wan22(policy_pipe, reference_pipe, preference_pairs, beta=0.1, epochs=10):
    """
    True DPO for diffusion models.

    Args:
        policy_pipe: Trainable Wan2.2 model
        reference_pipe: Frozen reference Wan2.2 model
        preference_pairs: List of (prompt, preferred_video, rejected_video)
        beta: DPO temperature (default 0.1)
    """

    for epoch in range(epochs):
        for batch in preference_pairs:
            prompt = batch["prompt"]

            # Load actual video pairs from disk
            preferred_video = load_video(batch["preferred_video"])  # (T, C, H, W)
            rejected_video = load_video(batch["rejected_video"])    # (T, C, H, W)

            # Move to device
            preferred_video = preferred_video.to("cuda", dtype=torch.float16)
            rejected_video = rejected_video.to("cuda", dtype=torch.float16)

            # Encode prompt (same for both models)
            text_embeds = policy_pipe._encode_prompt(
                prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )

            # Encode videos to latent space
            with torch.no_grad():
                preferred_latent = policy_pipe.vae.encode(preferred_video).latent_dist.sample()
                rejected_latent = policy_pipe.vae.encode(rejected_video).latent_dist.sample()

            # Sample random timestep for denoising loss
            timestep = torch.randint(
                0,
                policy_pipe.scheduler.config.num_train_timesteps,
                (1,),
                device="cuda"
            )

            # Add noise to latents
            noise_preferred = torch.randn_like(preferred_latent)
            noise_rejected = torch.randn_like(rejected_latent)

            noisy_preferred = policy_pipe.scheduler.add_noise(
                preferred_latent, noise_preferred, timestep
            )
            noisy_rejected = policy_pipe.scheduler.add_noise(
                rejected_latent, noise_rejected, timestep
            )

            # ============================================================
            # DPO LOSS COMPUTATION
            # ============================================================

            # Policy model predictions (trainable)
            policy_pred_preferred = policy_pipe.unet(
                noisy_preferred, timestep, text_embeds
            ).sample
            policy_pred_rejected = policy_pipe.unet(
                noisy_rejected, timestep, text_embeds
            ).sample

            # Reference model predictions (frozen)
            with torch.no_grad():
                ref_pred_preferred = reference_pipe.unet(
                    noisy_preferred, timestep, text_embeds
                ).sample
                ref_pred_rejected = reference_pipe.unet(
                    noisy_rejected, timestep, text_embeds
                ).sample

            # Compute implicit rewards (negative denoising error)
            # Lower error = higher implicit probability = higher reward
            policy_reward_preferred = -F.mse_loss(
                policy_pred_preferred, noise_preferred, reduction='mean'
            )
            policy_reward_rejected = -F.mse_loss(
                policy_pred_rejected, noise_rejected, reduction='mean'
            )

            ref_reward_preferred = -F.mse_loss(
                ref_pred_preferred, noise_preferred, reduction='mean'
            )
            ref_reward_rejected = -F.mse_loss(
                ref_pred_rejected, noise_rejected, reduction='mean'
            )

            # DPO loss: maximize preference gap while staying close to reference
            logits = beta * (
                (policy_reward_preferred - ref_reward_preferred) -
                (policy_reward_rejected - ref_reward_rejected)
            )

            # Binary cross-entropy: we want preferred > rejected
            dpo_loss = -F.logsigmoid(logits)

            # ============================================================
            # OPTIONAL: Add regularization to prevent drift
            # ============================================================

            # KL penalty (keep policy close to reference)
            kl_loss = F.mse_loss(policy_pred_preferred, ref_pred_preferred) + \
                      F.mse_loss(policy_pred_rejected, ref_pred_rejected)

            # Total loss
            loss = dpo_loss + 0.01 * kl_loss

            # ============================================================
            # OPTIMIZATION STEP
            # ============================================================

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_pipe.unet.parameters(), 1.0)
            optimizer.step()

            # Logging
            with torch.no_grad():
                accuracy = (logits > 0).float().mean().item()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"DPO Acc: {accuracy:.3f}, Logits: {logits.item():.3f}")

        # Save checkpoint
        policy_pipe.save_pretrained(f"wan22_dpo_epoch{epoch}")
```

### Step 4: Video Loading Helper

```python
import cv2
import numpy as np
import torch
from torchvision import transforms

def load_video(video_path, n_frames=16, target_size=(256, 256)):
    """
    Load video file and convert to tensor.

    Returns:
        video_tensor: (T, C, H, W) tensor normalized to [-1, 1]
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample uniform frame indices
    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, target_size)
            frames.append(frame)

    cap.release()

    # Pad if needed
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

    # Convert to tensor (T, H, W, C) -> (T, C, H, W)
    video_array = np.stack(frames)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)

    # Normalize to [-1, 1] (standard for diffusion models)
    video_tensor = (video_tensor / 127.5) - 1.0

    return video_tensor
```

---

## Key Differences from RLHF

| Aspect | RLHF (Wrong) | DPO (Correct) |
|--------|-------------|---------------|
| Training data | Generate online | Use existing preference pairs |
| Reward model | Used during training | Only for initial BT/DPO training |
| Sampling | Need to sample from model | No sampling needed |
| Stability | Can be unstable | More stable |
| Efficiency | Slower (online generation) | Faster (offline) |
| Implementation | Complex (RL loop) | Simpler (supervised-like) |

---

## Memory-Efficient Version (LoRA)

```python
from peft import LoraConfig, get_peft_model

# Add LoRA adapters (only ~100M params instead of 14B)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none",
)

# Apply LoRA to policy UNet only
policy_pipe.unet = get_peft_model(policy_pipe.unet, lora_config)

# Reference remains full model (frozen)
# Now training is 100x faster and uses 10x less memory!
```

---

## Complete Training Script

```python
#!/usr/bin/env python3
"""
DPO Fine-tuning for Wan2.2 T2V Model
"""

import argparse
import json
import torch
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

class VideoPreferenceDataset(Dataset):
    def __init__(self, preference_json, videos_dir):
        with open(preference_json) as f:
            self.data = json.load(f)
        self.videos_dir = videos_dir

        # Flatten to preference pairs
        self.pairs = []
        for gid, group in self.data.items():
            prompt = group["prompt"]
            for comp in group["pairwise_comparisons"]:
                if not comp.get("tie"):
                    self.pairs.append({
                        "prompt": prompt,
                        "preferred": comp["preferred"],
                        "rejected": comp["rejected"],
                    })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "prompt": pair["prompt"],
            "preferred_path": f"{self.videos_dir}/{pair['preferred']}",
            "rejected_path": f"{self.videos_dir}/{pair['rejected']}",
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B")
    parser.add_argument("--data", default="video_rankings3_pairwise.json")
    parser.add_argument("--videos_dir", default="./wan22-dataset/videos")
    parser.add_argument("--output_dir", default="wan22_dpo_output")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl_penalty", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()

    print("Loading models...")
    policy_pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda")

    reference_pipe = DiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda")
    reference_pipe.unet.eval()
    for p in reference_pipe.unet.parameters():
        p.requires_grad = False

    if args.use_lora:
        print(f"Applying LoRA (r={args.lora_r})...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        policy_pipe.unet = get_peft_model(policy_pipe.unet, lora_config)
        policy_pipe.unet.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in policy_pipe.unet.parameters() if p.requires_grad],
        lr=args.lr
    )

    print("Loading dataset...")
    dataset = VideoPreferenceDataset(args.data, args.videos_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Starting DPO training for {args.epochs} epochs...")
    dpo_train_wan22(
        policy_pipe,
        reference_pipe,
        dataloader,
        optimizer,
        beta=args.beta,
        kl_penalty=args.kl_penalty,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
```

---

## Running the Training

```bash
# Install dependencies
pip install diffusers transformers peft accelerate

# Train with LoRA (recommended for 1-2 GPUs)
python dpo_finetune_wan22.py \
    --data video_rankings3_pairwise.json \
    --videos_dir ./wan22-dataset/videos \
    --use_lora \
    --lora_r 16 \
    --beta 0.1 \
    --epochs 10 \
    --lr 1e-6

# Train full model (requires 8x A100 80GB)
python dpo_finetune_wan22.py \
    --data video_rankings3_pairwise.json \
    --videos_dir ./wan22-dataset/videos \
    --beta 0.1 \
    --epochs 10 \
    --lr 1e-7
```

---

## What You DON'T Need

❌ **Reward model during fine-tuning** - DPO is reward-free!
❌ **Online sampling** - Use existing preference pairs
❌ **PPO or other RL algorithms** - DPO is supervised-like
❌ **Separate reward scoring step** - Implicit in denoising error

---

## What You DO Need

✅ **Preference pairs** - You already have this!
✅ **Reference model** - Frozen copy of original Wan2.2
✅ **Policy model** - Trainable Wan2.2
✅ **DPO loss** - Implemented above
✅ **Video encoder** - VAE from Wan2.2

---

## Summary

**DPO is simpler than RLHF**:
1. Load preference pairs (already have)
2. Load policy + reference models
3. Compute denoising errors on preferred vs rejected videos
4. Apply DPO loss: prefer lower error on preferred videos
5. No reward model needed during training!

The key insight: **Denoising error ≈ negative log probability**, so DPO can work directly with diffusion models using MSE loss as the implicit reward.
