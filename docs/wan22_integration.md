# Integrating DPO Reward Model with Wan2.2 T2V Model

## Overview

After training your reward model (BT or DPO), the next step is to use it to fine-tune the actual Wan-AI/Wan2.2-T2V-A14B video generation model.

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   DPO Training Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Train Reward Model                                     │
│     Input: (prompt, video) → Output: preference score      │
│     ✓ Already done with your current notebook             │
│                                                             │
│  2. Use Reward for T2V Fine-tuning                        │
│     Sample videos from Wan2.2 → Score with reward →        │
│     → Backprop through diffusion model                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Wan2.2 T2V Model Structure                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Text → T5 Encoder → text_embeds                           │
│                            ↓                                │
│                      [DiT Denoiser]                        │
│                            ↓                                │
│                    Latent z (noisy) → VAE Decoder          │
│                            ↓                                │
│                      Generated Video                        │
│                            ↓                                │
│                    [Your Reward Model]                      │
│                            ↓                                │
│                    Preference Score                         │
│                            ↓                                │
│              Backprop to optimize DiT                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Load Wan2.2 Model

```python
from diffusers import DiffusionPipeline
import torch

# Load Wan2.2 T2V pipeline
pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

# Access components
vae = pipe.vae               # Video VAE encoder/decoder
text_encoder = pipe.text_encoder  # T5 text encoder
unet = pipe.unet             # DiT denoising transformer
scheduler = pipe.scheduler   # Noise scheduler
```

---

## Step 2: Load Your Trained Reward Model

```python
# Load your trained BT or DPO reward model
reward_model = RewardModel(feat_dim=512, n_prompts=len(p2i), ...)
reward_model.load_state_dict(torch.load("dpo_output/best_reward_model.pt"))
reward_model.eval()
reward_model.to("cuda")

# Freeze reward model (we're not training it further)
for p in reward_model.parameters():
    p.requires_grad = False
```

---

## Step 3: DPO Fine-tuning Loop

### Option A: Decode + Re-encode (Simpler, less efficient)

```python
def dpo_finetune_wan22(pipe, reward_model, prompts, epochs=10):
    """Fine-tune Wan2.2 with DPO using reward model."""

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-6)

    for epoch in range(epochs):
        for prompt in prompts:
            # Generate two videos from same prompt (needed for DPO)
            with torch.no_grad():
                # Sample 1
                video1 = pipe(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(epoch * 100)
                ).frames[0]

                # Sample 2 (different noise)
                video2 = pipe(
                    prompt=prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(epoch * 100 + 1)
                ).frames[0]

            # Encode videos with your feature extractor (CLIP or X-CLIP)
            feat1 = encode_video(video1)
            feat2 = encode_video(video2)

            # Get preference scores from reward model
            prompt_idx = torch.tensor([p2i[prompt]]).to("cuda")
            score1 = reward_model.reward(prompt_idx, feat1.to("cuda"))
            score2 = reward_model.reward(prompt_idx, feat2.to("cuda"))

            # DPO loss: prefer higher-scored video
            # If score1 > score2, video1 is "preferred"
            if score1 > score2:
                r_w, r_l = score1, score2
                preferred_video = video1
            else:
                r_w, r_l = score2, score1
                preferred_video = video2

            # Now re-generate preferred video with gradients enabled
            optimizer.zero_grad()

            # Forward pass through denoising process
            latents = encode_video_to_latent(preferred_video, pipe.vae)
            noise = torch.randn_like(latents)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,))

            noisy_latents = scheduler.add_noise(latents, noise, timestep)

            # Predict noise
            text_embeds = pipe._encode_prompt(prompt, ...)
            noise_pred = pipe.unet(noisy_latents, timestep, text_embeds).sample

            # Reconstruction loss (standard diffusion)
            diff_loss = F.mse_loss(noise_pred, noise)

            # DPO reward loss
            reward_loss = -torch.log(torch.sigmoid(r_w - r_l))

            # Combined loss
            loss = diff_loss + 0.1 * reward_loss

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Prompt: {prompt[:30]}, Loss: {loss.item():.4f}")

        # Save checkpoint
        pipe.save_pretrained(f"wan22_dpo_epoch{epoch}")
```

### Option B: Latent-space DPO (More efficient, advanced)

```python
def dpo_finetune_latent(pipe, reward_model, prompts, epochs=10):
    """Fine-tune in latent space without decoding."""

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-6)

    for epoch in range(epochs):
        for prompt in prompts:
            # Encode prompt
            text_embeds = pipe._encode_prompt(prompt, device="cuda", ...)

            # Sample two different noise initializations
            latent_shape = (1, pipe.unet.config.in_channels, 16, 64, 64)  # T, H, W
            noise1 = torch.randn(latent_shape, device="cuda")
            noise2 = torch.randn(latent_shape, device="cuda")

            # Denoise to get two different videos
            with torch.no_grad():
                latents1 = pipe.scheduler.add_noise(noise1, timestep=0)
                latents2 = pipe.scheduler.add_noise(noise2, timestep=0)

                for t in pipe.scheduler.timesteps:
                    noise_pred1 = pipe.unet(latents1, t, text_embeds).sample
                    latents1 = pipe.scheduler.step(noise_pred1, t, latents1).prev_sample

                    noise_pred2 = pipe.unet(latents2, t, text_embeds).sample
                    latents2 = pipe.scheduler.step(noise_pred2, t, latents2).prev_sample

                # Decode to videos
                video1 = pipe.vae.decode(latents1).sample
                video2 = pipe.vae.decode(latents2).sample

            # Score with reward model (same as before)
            feat1 = encode_video_features(video1)
            feat2 = encode_video_features(video2)

            score1 = reward_model.reward(prompt_idx, feat1)
            score2 = reward_model.reward(prompt_idx, feat2)

            # Determine preferred latent
            if score1 > score2:
                preferred_latent = latents1
                r_w, r_l = score1, score2
            else:
                preferred_latent = latents2
                r_w, r_l = score2, score1

            # Now train on preferred latent (with gradients)
            optimizer.zero_grad()

            # Re-run denoising on preferred
            timestep = torch.randint(0, len(pipe.scheduler.timesteps), (1,)).to("cuda")
            noise = torch.randn_like(preferred_latent)
            noisy_latent = pipe.scheduler.add_noise(preferred_latent, noise, timestep)

            noise_pred = pipe.unet(noisy_latent, timestep, text_embeds).sample

            # Loss
            diff_loss = F.mse_loss(noise_pred, noise)
            reward_loss = -torch.log(torch.sigmoid(r_w - r_l))

            loss = diff_loss + 0.1 * reward_loss
            loss.backward()
            optimizer.step()
```

---

## Step 4: Practical Implementation Tips

### Memory Optimization

```python
# Use gradient checkpointing
pipe.unet.enable_gradient_checkpointing()
pipe.vae.enable_gradient_checkpointing()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_dpo_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Processing

```python
# Process multiple prompts in parallel
batch_size = 4
for batch_prompts in DataLoader(prompts, batch_size=batch_size):
    videos = pipe(batch_prompts, ...)
    scores = reward_model(batch_prompts, videos)
    # ... DPO loss computation
```

### Checkpoint Management

```python
# Save checkpoints periodically
if epoch % 10 == 0:
    checkpoint = {
        "epoch": epoch,
        "unet_state": pipe.unet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_reward": best_val_reward,
    }
    torch.save(checkpoint, f"wan22_dpo_ckpt_epoch{epoch}.pt")
```

---

## Step 5: Evaluation

```python
def evaluate_model(pipe, reward_model, val_prompts):
    """Evaluate fine-tuned model."""
    total_reward = 0
    num_samples = 0

    for prompt in val_prompts:
        # Generate video
        video = pipe(prompt, num_inference_steps=50).frames[0]

        # Score with reward model
        feat = encode_video(video)
        score = reward_model.reward(prompt_idx, feat)

        total_reward += score.item()
        num_samples += 1

    avg_reward = total_reward / num_samples
    print(f"Average reward: {avg_reward:.4f}")
    return avg_reward
```

---

## Important Considerations

### 1. Reward Model Feature Extractor Must Match

If you trained your reward model with **CLIP features**, you must use CLIP during fine-tuning.

```python
# Ensure consistency
if trained_with_clip:
    encode_video = encode_video_clip
elif trained_with_xclip:
    encode_video = encode_video_xclip
elif trained_with_vae:
    encode_video = encode_video_vae
```

### 2. KL Regularization with Reference Model

To prevent the model from drifting too far, keep a frozen reference:

```python
# Before fine-tuning
reference_unet = copy.deepcopy(pipe.unet)
reference_unet.eval()
for p in reference_unet.parameters():
    p.requires_grad = False

# During training, add KL penalty
ref_noise_pred = reference_unet(noisy_latent, timestep, text_embeds).sample
kl_loss = F.mse_loss(noise_pred, ref_noise_pred)

total_loss = diff_loss + 0.1 * reward_loss + 0.01 * kl_loss
```

### 3. Sampling Strategy

DPO needs pairs. You can:
- **On-policy**: Generate from current model
- **Off-policy**: Use pre-generated videos from dataset
- **Mixed**: Combine both

### 4. Computational Requirements

Fine-tuning Wan2.2 (14B params) requires:
- GPU: A100 80GB or H100 (minimum)
- RAM: 128GB+
- Storage: ~100GB for checkpoints

Consider using:
- **LoRA** for parameter-efficient fine-tuning
- **Gradient accumulation** for smaller GPUs
- **Distributed training** (multi-GPU)

---

## Alternative: LoRA Fine-tuning (Recommended)

```python
from peft import LoraConfig, get_peft_model

# Add LoRA adapters (much more efficient)
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
)

pipe.unet = get_peft_model(pipe.unet, lora_config)

# Now only ~100M params to train instead of 14B
trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params / 1e6:.1f}M")

# Fine-tune as before, but much faster and less memory
```

---

## Next Steps

1. **Start small**: Fine-tune on 10-20 prompts first
2. **Validate**: Check if generated videos improve according to reward
3. **Scale up**: Gradually increase dataset size
4. **Compare**: A/B test original vs fine-tuned model
5. **Iterate**: Refine reward model based on failure cases

---

## Resources

- Diffusers DPO example: https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_dpo
- LoRA paper: https://arxiv.org/abs/2106.09685
- DPO for diffusion: https://arxiv.org/abs/2311.12908 (Diffusion-DPO)

---

## Summary

1. ✅ Train reward model (done)
2. ✅ Load Wan2.2 pipeline
3. ✅ Generate video pairs
4. ✅ Score with reward model
5. ✅ Backprop through diffusion process
6. ✅ Save fine-tuned model

The key insight: **Your reward model provides the learning signal for what makes a "good" video according to human preferences.**
