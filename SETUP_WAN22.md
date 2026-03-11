# Wan2.2 Model Setup Guide

## Issues Fixed

### ❌ Problem: Wan2.2 Model Not Loading
```
⚠️  Could not load Wan2.2 VAE: Wan-AI/Wan2.2-T2V-A14B does not appear to have a file named config.json
Falling back to Stable Diffusion VAE (similar architecture)
```

### ✅ Solution: Use Correct Model Path and Classes

The issue was using incorrect model identifiers and classes. Here's what changed:

| Component | ❌ Incorrect | ✅ Correct |
|-----------|-------------|-----------|
| **Model Path** | `Wan-AI/Wan2.2-T2V-A14B` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| **Pipeline** | `DiffusionPipeline` | `WanPipeline` |
| **VAE Class** | `AutoencoderKL` | `AutoencoderKLWan` |
| **Model Component** | `pipe.unet` (UNet) | `pipe.transformer` (Transformer MoE) |

---

## Installation Requirements

### 1. Install Latest Diffusers from Main Branch

Wan2.2 support requires the latest diffusers (not yet in stable release):

```bash
pip install git+https://github.com/huggingface/diffusers
```

### 2. Install Required Dependencies

```bash
pip install torch torchvision
pip install transformers accelerate
pip install opencv-python pillow
pip install numpy matplotlib scipy
pip install imageio
```

### 3. Optional: 8-bit Optimizer (for memory efficiency)

```bash
pip install bitsandbytes
```

---

## Correct Usage

### Loading Wan2.2 VAE (for feature extraction)

```python
from diffusers import AutoencoderKLWan

# ✅ Correct
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.float32  # Use float32 for encoding stability
)

# ❌ Incorrect (old code)
# vae = AutoencoderKL.from_pretrained(
#     "Wan-AI/Wan2.2-T2V-A14B",  # Wrong path
#     subfolder="vae",
# )
```

### Loading Full Wan2.2 Pipeline

```python
from diffusers import WanPipeline, AutoencoderKLWan
import torch

# Load VAE
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.float32
)

# Load pipeline
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    vae=vae,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for MoE transformer
)
pipe.to("cuda")

# Generate video
output = pipe(
    prompt="A ball rolling down a slope with realistic physics",
    height=720,
    width=1280,
    num_frames=81,  # 5 seconds at 24fps
    guidance_scale=4.0,
    num_inference_steps=40,
)

frames = output.frames[0]
```

### Accessing Model Components

```python
# ✅ Correct - Wan2.2 uses transformer (MoE architecture)
transformer = pipe.transformer
vae = pipe.vae
text_encoder = pipe.text_encoder

# ❌ Incorrect - Wan2.2 does NOT have a UNet
# unet = pipe.unet  # AttributeError!
```

---

## Model Architecture Details

### Wan2.2 T2V A14B Specifications

- **Architecture**: Mixture-of-Experts (MoE) Transformer
- **Total Parameters**: 27B (two 14B experts)
- **Active Parameters**: 14B per inference step
- **Experts**:
  - High-noise expert: Handles early denoising (layout)
  - Low-noise expert: Handles later denoising (details)
- **VAE**: Custom Wan2.2-VAE (4×16×16 compression)
- **Resolutions**: 480P and 720P support
- **Frame Rate**: 24fps
- **Video Length**: Up to 5 seconds (81 frames)

### Why Two Experts?

The MoE design switches between experts based on the signal-to-noise ratio:
- **Early timesteps** (high noise): Use high-noise expert for overall composition
- **Late timesteps** (low noise): Use low-noise expert for detail refinement

This makes training more efficient than a single 27B model!

---

## Hardware Requirements

### Minimum (Inference)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Resolution**: 720P
- **Frames**: 81 (5 seconds)

### Recommended (Training/Fine-tuning)
- **GPU**: A100 80GB or H100
- **VRAM**: 80GB minimum
- **Note**: DPO fine-tuning requires loading 2 copies of transformer (policy + reference)

### Memory Optimization Tips

1. **Reduce batch size**: Start with `batch_size=1`
2. **Reduce frames**: Use `n_frames=4` or `8` instead of `16`
3. **Use gradient accumulation**: `grad_accum=16` for effective batch size
4. **Enable 8-bit AdamW**: Requires `bitsandbytes` package
5. **Use bfloat16**: More memory efficient than float32

```python
# Optimized config for 24GB GPU
python wan22_dpo_finetune.py \
    --batch-size 1 \
    --grad-accum 16 \
    --n-frames 4 \
    --lr 1e-6
```

---

## Updated Files

All files have been updated to use the correct model paths and classes:

### ✅ Fixed Files:
1. **`dpo_train_vae.ipynb`**: Notebook for reward model training
   - Updated model path to `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
   - Changed `AutoencoderKL` → `AutoencoderKLWan`
   - Added dtype fix (float16 → float32)

2. **`wan22_dpo_finetune.py`**: Full DPO fine-tuning script
   - Updated model path
   - Changed `DiffusionPipeline` → `WanPipeline`
   - Changed `AutoencoderKL` → `AutoencoderKLWan`
   - Changed `unet` → `transformer` throughout
   - Added 8-bit optimizer support

3. **`generate_with_finetuned.py`**: Video generation script
   - Updated to use `WanPipeline`
   - Added correct parameters (height, width, num_frames)
   - Default: 720P, 81 frames

---

## Testing Your Setup

### Test 1: VAE Loading (from notebook)
```python
from diffusers import AutoencoderKLWan
import torch

vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.float32
)
print(f"✓ VAE loaded successfully: {type(vae)}")
```

**Expected output**:
```
✓ VAE loaded successfully: <class 'diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan'>
```

### Test 2: Full Pipeline Loading
```python
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
print(f"✓ Pipeline loaded successfully")
print(f"✓ Transformer: {type(pipe.transformer)}")
```

**Expected output**:
```
✓ Pipeline loaded successfully
✓ Transformer: <class 'diffusers.models.transformers.wan_transformer_2d.WanTransformer2DModel'>
```

---

## Troubleshooting

### Issue: "No module named 'diffusers.models.autoencoders.autoencoder_kl_wan'"

**Solution**: Install latest diffusers from main branch:
```bash
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers
```

### Issue: "CUDA out of memory" during fine-tuning

**Solution**: Reduce memory usage:
```bash
python wan22_dpo_finetune.py \
    --batch-size 1 \
    --n-frames 4 \
    --grad-accum 16
```

### Issue: "Model not found: Wan-AI/Wan2.2-T2V-A14B-Diffusers"

**Solution**: Verify internet connection and HuggingFace access:
```bash
huggingface-cli login  # If private model
```

---

## Next Steps

Now that Wan2.2 is properly configured:

1. **Run reward model training**:
   ```bash
   jupyter notebook dpo_train_vae.ipynb
   # Should now load Wan2.2 VAE correctly
   ```

2. **Run full DPO fine-tuning**:
   ```bash
   python wan22_dpo_finetune.py \
       --data video_rankings3_pairwise.json \
       --videos-dir ./wan22-dataset/videos \
       --output-dir wan22_finetuned
   ```

3. **Generate videos with fine-tuned model**:
   ```bash
   python generate_with_finetuned.py \
       --model wan22_finetuned/best_model \
       --prompt "Your prompt here" \
       --output output.mp4
   ```

---

## References

- **Model**: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers
- **Diffusers Docs**: https://huggingface.co/docs/diffusers
- **CS234 DPO Reference**: `/Users/hivamoh/Desktop/CS234/starter_code/run_dpo.py`

**Sources:**
- [Wan-AI/Wan2.2-T2V-A14B · Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- [Wan-AI/Wan2.2-T2V-A14B-Diffusers · Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
