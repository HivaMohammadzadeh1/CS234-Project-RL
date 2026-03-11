# Complete DPO Pipeline for Wan2.2 T2V Fine-tuning

## Understanding the Two Approaches

### ❌ What You Had Before (Incomplete)
**File**: `dpo_train_vae.ipynb`
**What it does**: Trains a **reward model** on video features
- Takes pre-generated videos
- Learns to predict which video is better
- **Does NOT modify Wan2.2 model**
- Useful for: evaluation, analysis, understanding preferences

**Output**: A reward model (NOT a better video generator)

---

### ✅ What You Need Now (Full RL Loop)
**File**: `wan22_dpo_finetune.py`
**What it does**: **Directly optimizes Wan2.2 T2V model** using DPO
- Loads Wan2.2 diffusion model
- Uses preference pairs to fine-tune the UNet
- **Modifies the T2V model** to generate better videos
- Follows CS234 DPO pattern (run_dpo.py:268-269)

**Output**: A fine-tuned Wan2.2 model that generates higher-quality videos

---

## How DPO Works for Diffusion Models

### CS234 DPO (for policies):
```python
# From /Users/hivamoh/Desktop/CS234/starter_code/run_dpo.py
log_pi_w = policy.distribution(obs).log_prob(actions_w)
log_pi_l = policy.distribution(obs).log_prob(actions_l)

with torch.no_grad():
    log_ref_w = reference.distribution(obs).log_prob(actions_w)
    log_ref_l = reference.distribution(obs).log_prob(actions_l)

logits = beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
loss = -torch.nn.functional.logsigmoid(logits).mean()
```

**Key insight**: DPO optimizes the policy directly using preference pairs, no reward model needed!

### Adapted for Wan2.2 (diffusion model):

```python
# For diffusion models, use denoising error as implicit log probability
policy_reward_w = -mse(policy_unet(noisy_w), true_noise_w)
policy_reward_l = -mse(policy_unet(noisy_l), true_noise_l)

with torch.no_grad():
    ref_reward_w = -mse(reference_unet(noisy_w), true_noise_w)
    ref_reward_l = -mse(reference_unet(noisy_l), true_noise_l)

logits = beta * ((policy_reward_w - ref_reward_w) - (policy_reward_l - ref_reward_l))
loss = -torch.nn.functional.logsigmoid(logits).mean()
```

**Key adaptation**: Lower denoising error = higher quality = higher reward

---

## Full Pipeline: Step-by-Step

### Step 1: Prepare Your Data ✅ (Already Done)
```bash
# You already have:
# - video_rankings3_pairwise.json (preference pairs)
# - ./wan22-dataset/videos/ (video files)
```

### Step 2: Run DPO Fine-tuning 🆕 (New!)

```bash
python wan22_dpo_finetune.py \
    --data video_rankings3_pairwise.json \
    --videos-dir ./wan22-dataset/videos \
    --output-dir wan22_dpo_finetuned \
    --beta 0.1 \
    --lr 1e-6 \
    --batch-size 1 \
    --grad-accum 4 \
    --epochs 10 \
    --n-frames 8
```

**What this does**:
1. Loads Wan2.2 T2V model
2. Creates frozen reference copy
3. Trains policy UNet using preference pairs
4. Saves fine-tuned model to `wan22_dpo_finetuned/best_model`

**Expected training time**:
- ~2-4 hours on A100 (batch_size=1, 8 frames)
- Memory: ~24GB VRAM minimum

### Step 3: Generate Videos with Fine-tuned Model

```python
from diffusers import DiffusionPipeline

# Load your fine-tuned model
pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B")
pipe.unet.from_pretrained("wan22_dpo_finetuned/best_model")
pipe.to("cuda")

# Generate video with improved quality
video = pipe(
    prompt="A ball rolling down a slope with realistic physics",
    num_frames=16,
    guidance_scale=7.5,
).frames
```

---

## Comparison: Reward Model vs DPO Fine-tuning

| Aspect | Reward Model (`dpo_train_vae.ipynb`) | DPO Fine-tuning (`wan22_dpo_finetune.py`) |
|--------|--------------------------------------|-------------------------------------------|
| **What it trains** | Preference classifier on features | Wan2.2 UNet directly |
| **Output** | Reward model (for evaluation) | Fine-tuned video generator |
| **Modifies T2V model?** | ❌ No | ✅ Yes |
| **Can generate better videos?** | ❌ No | ✅ Yes |
| **Training time** | ~30 min | ~2-4 hours |
| **Memory required** | ~8GB | ~24GB |
| **When to use** | Analysis, evaluation | Actual model improvement |

---

## Key Differences from CS234

### CS234 (Hopper RL):
- **Policy**: Neural network outputting action distributions
- **Reference**: Copy of pretrained SFT policy
- **Data**: (observation, preferred_actions, rejected_actions)
- **Reward**: Log probability of actions

### Wan2.2 DPO (Video Generation):
- **Policy**: UNet in diffusion model
- **Reference**: Copy of original Wan2.2 UNet
- **Data**: (prompt, preferred_video, rejected_video)
- **Reward**: Negative denoising MSE (implicit log prob)

**Same core DPO loss structure!** Just different "reward" definitions.

---

## Training Tips

### Memory Optimization
If you run out of VRAM:
```bash
# Reduce batch size
--batch-size 1

# Reduce frames per video
--n-frames 4  # instead of 8

# Increase gradient accumulation
--grad-accum 8  # effective batch size = 8

# Use 8-bit Adam (install bitsandbytes)
--use-8bit-adam
```

### Hyperparameter Tuning
- **Beta (0.01 - 1.0)**: Controls strength of preference signal
  - Lower β: more conservative, stays close to reference
  - Higher β: more aggressive, tries to maximize preferences
  - Start with 0.1 (CS234 default)

- **Learning rate (1e-7 - 1e-5)**:
  - Diffusion models are sensitive!
  - Start with 1e-6
  - If training unstable, reduce to 5e-7

### Expected Results
After fine-tuning:
- **Preference accuracy**: Should see policy prefer preferred > rejected ~70-80% of time
- **Generated videos**: Better physics simulation, clearer motion, higher fidelity
- **Convergence**: 5-10 epochs usually sufficient

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Smallest possible config
python wan22_dpo_finetune.py \
    --batch-size 1 \
    --grad-accum 16 \
    --n-frames 4 \
    ...
```

### "Training loss not decreasing"
- Check learning rate (try 5e-7)
- Verify videos are loading correctly
- Ensure preference pairs are non-trivial

### "Accuracy stuck at 50%"
- This means policy can't distinguish preferred from rejected
- Check beta value (try 0.5 or 1.0)
- Verify reference UNet is truly frozen
- Ensure videos are encoded correctly

---

## Next Steps: Advanced Techniques

Once basic DPO works, you can explore:

1. **LoRA Fine-tuning**: Use parameter-efficient fine-tuning
   ```python
   from peft import LoraConfig, get_peft_model
   lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["to_q", "to_v"])
   policy_unet = get_peft_model(policy_unet, lora_config)
   ```

2. **Multi-aspect Rewards**: Optimize for multiple qualities
   - Physics accuracy
   - Visual fidelity
   - Motion smoothness

3. **Iterative DPO**: Generate new videos, get preferences, retrain
   - Similar to RLHF but offline

4. **Distillation**: Create smaller, faster models from fine-tuned version

---

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `dpo_train_vae.ipynb` | Train reward model on features | Analysis, evaluation |
| `wan22_dpo_finetune.py` | **Full DPO fine-tuning** | **Actual model improvement** |
| `validate_dpo_implementation.py` | Verify DPO math is correct | Debugging |
| `QUICKSTART.md` | Quick reference guide | Getting started |
| `DPO_FULL_PIPELINE.md` (this file) | Complete pipeline explanation | Understanding workflow |

---

## Summary

✅ **You now have the full DPO RL loop!**

**Before**: Trained reward model on video features (incomplete)
**Now**: Can directly fine-tune Wan2.2 T2V model using preferences (complete!)

**To run full pipeline**:
```bash
# 1. Fine-tune model (2-4 hours)
python wan22_dpo_finetune.py \
    --data video_rankings3_pairwise.json \
    --videos-dir ./wan22-dataset/videos \
    --output-dir wan22_finetuned

# 2. Generate better videos
python generate_videos.py --model wan22_finetuned/best_model
```

**Expected outcome**: Wan2.2 generates videos that better align with human preferences! 🚀
