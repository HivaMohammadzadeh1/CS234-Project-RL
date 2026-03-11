# Better Video Encoding Strategies for T2V DPO

## Goal: Use DPO to improve Wan-AI/Wan2.2-T2V-A14B

Your current approach uses CLIP (image-based) features, which are suboptimal for training rewards for a T2V model. Here are better approaches, ranked from ideal to practical.

---

## 🥇 Ideal: Use Wan2.2's Native VAE Latent Space

**Why**: The T2V model internally encodes videos into a latent space using a Video VAE. Training your reward model in the **same latent space** ensures perfect alignment.

### Architecture Overview:
```
Text Prompt → T5 Text Encoder → [Diffusion Process] ← VAE Latent z
                                         ↓
                                  Generated Video

For DPO Reward Model:
Text Prompt + VAE Latent z → Reward Model → Preference Score
```

### Implementation:

```python
# 1. Load Wan2.2's VAE encoder (from their official repo/checkpoint)
from wan_model import WanVideoVAE  # hypothetical - check their repo

vae = WanVideoVAE.from_pretrained("Wan-AI/Wan2.2-T2V-A14B", subfolder="vae")
vae.eval()
vae.to(device)

@torch.no_grad()
def encode_video_vae(video_path, vae, n_frames=16):
    """Encode video using Wan2.2's VAE."""
    # Load video frames (B, T, C, H, W)
    frames = load_video_frames(video_path, n_frames=n_frames)
    frames = frames.to(device)

    # Encode to VAE latent space
    # Shape: (B, latent_dim, T', H', W') - compressed spatiotemporally
    latent = vae.encode(frames).latent_dist.sample()

    # Global pool or use latent statistics
    # Option 1: Mean pool across space-time
    video_feat = latent.mean(dim=[2, 3, 4])  # (B, latent_dim)

    # Option 2: Use latent statistics (mean + logvar)
    mean = latent.mean(dim=[2, 3, 4])
    var = latent.var(dim=[2, 3, 4])
    video_feat = torch.cat([mean, var], dim=-1)

    return video_feat
```

### Why this is best:
- ✅ Matches T2V model's internal representation
- ✅ Captures temporal dynamics (not just individual frames)
- ✅ Direct optimization signal for diffusion training
- ✅ Can plug reward gradients directly into VAE latents during DPO fine-tuning

### Challenge:
- Need access to Wan2.2's VAE weights (check if they're public)
- May need to reverse-engineer their model architecture

---

## 🥈 Good: Video-Native Encoders

Use models designed specifically for video understanding (not just images).

### Option 1: VideoMAE
```python
from transformers import VideoMAEModel, VideoMAEImageProcessor

model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

@torch.no_grad()
def encode_video_videomae(video_path, model, processor, n_frames=16):
    frames = load_video_frames(video_path, n_frames=n_frames)
    inputs = processor(list(frames), return_tensors="pt")

    outputs = model(**inputs.to(device))
    # Use CLS token or mean pool
    video_feat = outputs.last_hidden_state[:, 0]  # CLS token
    return video_feat
```

**Pros**:
- Pretrained on video datasets (Kinetics-400)
- Understands temporal dynamics
- 768-dim features

### Option 2: X-CLIP (Video-aware CLIP)
```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")

@torch.no_grad()
def encode_video_xclip(video_path, model, processor, n_frames=8):
    frames = load_video_frames(video_path, n_frames=n_frames)
    inputs = processor(videos=list(frames), return_tensors="pt")

    outputs = model.get_video_features(**inputs.to(device))
    return outputs  # 512-dim aligned with text
```

**Pros**:
- Video extension of CLIP
- Text-video alignment
- Good for T2V tasks

---

## 🥉 Better: Improved CLIP Usage

If sticking with CLIP, add temporal modeling.

### Option 1: Temporal Transformer over CLIP frames
```python
class TemporalCLIPEncoder(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=256, n_heads=8):
        super().__init__()
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=clip_dim, nhead=n_heads),
            num_layers=3
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, clip_dim))

    def forward(self, clip_features):
        # clip_features: (batch, n_frames, 512)
        B, T, D = clip_features.shape

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, clip_features], dim=1)  # (B, T+1, D)

        # Temporal attention
        x = self.temporal_attn(x)

        # Return CLS token
        return x[:, 0]  # (B, D)
```

### Option 2: 3D CNN on CLIP feature maps
```python
class SpatioTemporalCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model.vision_model
        self.conv3d = nn.Sequential(
            nn.Conv3d(768, 512, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=1),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, video_frames):
        # Extract CLIP features for each frame
        # Then apply 3D conv for temporal modeling
        ...
```

---

## 📊 Comparison Table

| Method | Temporal Modeling | Aligned with T2V | Complexity | Expected Performance |
|--------|------------------|------------------|------------|---------------------|
| **Wan2.2 VAE** | ✅ Native | ✅ Perfect | High | Best |
| **VideoMAE** | ✅ Native | ⚠️ Indirect | Medium | Very Good |
| **X-CLIP** | ✅ Native | ✅ Good | Medium | Very Good |
| **CLIP + Transformer** | ✅ Added | ⚠️ Indirect | Medium | Good |
| **CLIP Mean-pool** (current) | ❌ None | ❌ Poor | Low | Fair |

---

## 🎯 Recommended Implementation Path

### Phase 1: Quick Win (1-2 days)
Use **X-CLIP** - it's video-native and text-aligned:
```bash
pip install transformers
# Replace CLIP with X-CLIP in cell 11 of your notebook
```

### Phase 2: Better Alignment (1 week)
Add **temporal transformer** to aggregate frame features:
```python
# Keep CLIP features but add temporal attention
# Modify the reward model architecture
```

### Phase 3: Optimal (2-3 weeks)
Extract and use **Wan2.2's VAE encoder**:
```python
# Requires reverse-engineering their model
# Check their repo: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
```

---

## 💡 Additional Recommendations

### 1. Multi-Modal Features
Combine text and video features:
```python
class MultiModalReward(nn.Module):
    def __init__(self, video_dim, text_dim):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, video_feat, text_feat):
        v = self.video_proj(video_feat)
        t = self.text_proj(text_feat)
        return self.fusion(torch.cat([v, t], dim=-1))
```

### 2. Sample More Frames
Increase from 8 to 16-32 frames to capture more temporal information:
```python
cfg.n_frames = 32  # More frames = better temporal understanding
```

### 3. Use Optical Flow
Add motion features:
```python
import cv2

def compute_optical_flow(video_path):
    # Extract optical flow between consecutive frames
    # Aggregate flow statistics as features
    ...
```

---

## 🚀 Next Steps

1. **Immediate**: Replace CLIP with X-CLIP (minimal code change)
2. **Short-term**: Add temporal transformer over features
3. **Long-term**: Investigate Wan2.2 VAE access
4. **Parallel**: Experiment with VideoMAE for comparison

Would you like me to implement any of these approaches in your notebook?
