# TI2V Reference Image Fix

## Error Encountered

```
RuntimeError: Given groups=1, weight of size [3072, 48, 1, 2, 2], expected input[1, 96, 2, 16, 16] to have 48 channels, but got 96 channels instead
```

## Root Cause

The initial TI2V implementation incorrectly concatenated reference image latents with video latents in the channel dimension, doubling the channels from 48 to 96. However, the TI2V transformer's patch embedding layer expects exactly 48 input channels (same as T2V models).

**Key insight**: TI2V models don't concatenate reference images in latent space. Instead, they use the reference image as **the first frame of the video sequence**.

## Solution

### Before (BROKEN)
```python
# Extract reference images separately
pref_ref_image = extract_first_frame(pref_path)
rej_ref_image = extract_first_frame(rej_path)

# Encode video
pref_latent = encode_video(pref_path)  # [n_frames, C, H, W]

# Encode reference image
ref_latent = encode_image(pref_ref_image)  # [1, C, H, W]

# Concatenate in channel dimension ❌
combined = torch.cat([pref_latent, ref_latent.expand(n_frames, -1, -1, -1)], dim=1)  # [n_frames, 2*C, H, W]
```

### After (WORKING)
```python
# Extract first frame as reference
cap = cv2.VideoCapture(pref_path)
ret, ref_frame = cap.read()
cap.release()

# Prepend reference image to video sequence BEFORE encoding
def load_video_to_latent(video_path, vae, n_frames, device, reference_image=None):
    frames = []

    # Add reference image as first frame
    if reference_image is not None:
        ref_frame = cv2.resize(reference_image, (256, 256))
        frames.append(ref_frame)

    # Sample (n_frames - 1) frames from video
    # ... video loading code ...

    # Encode entire sequence (reference + video) through VAE
    video_tensor = stack_and_normalize(frames)  # [n_frames, C, H, W]
    latents = vae.encode(video_tensor)  # [n_frames, latent_C, latent_H, latent_W]

    return latents  # ✅ Correct number of channels
```

## Changes Made

### 1. Modified `load_video_to_latent()` (lines 301-359)

**Signature change**:
```python
# Before
def load_video_to_latent(video_path, vae, n_frames=16, device="cuda", return_first_frame=False):

# After
def load_video_to_latent(video_path, vae, n_frames=16, device="cuda", reference_image=None):
```

**Behavior change**:
- **Before**: Optionally returned first frame as separate tensor
- **After**: Accepts reference image and prepends it to video sequence before encoding

**Key logic**:
```python
# For TI2V: Sample (n_frames - 1) from video, prepend reference as frame 0
frames_to_sample = n_frames - 1 if reference_image is not None else n_frames

frames = []
if reference_image is not None:
    ref_frame = cv2.resize(reference_image, (256, 256))
    frames.append(ref_frame)  # First frame = reference image

# Then add video frames
for idx in indices:
    # ... load frame ...
    frames.append(frame)

# Encode entire sequence together
video_array = np.stack(frames)  # [n_frames, H, W, C]
# ... VAE encoding ...
```

### 2. Modified `compute_dpo_loss()` (lines 414-470)

**Removed**:
- Separate reference image extraction
- Reference image VAE encoding
- Temporal expansion of reference latents
- Channel concatenation logic
- Output channel splitting logic

**Added**:
```python
if is_ti2v_model:
    # Extract first frame from each video
    cap = cv2.VideoCapture(pref_path)
    ret, pref_ref_frame = cap.read()
    cap.release()
    if ret:
        pref_ref_frame = cv2.cvtColor(pref_ref_frame, cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(rej_path)
    ret, rej_ref_frame = cap.read()
    cap.release()
    if ret:
        rej_ref_frame = cv2.cvtColor(rej_ref_frame, cv2.COLOR_BGR2RGB)

    # Load video with reference frame prepended as first frame
    pref_latent = load_video_to_latent(pref_path, vae, args.n_frames, device, reference_image=pref_ref_frame)
    rej_latent = load_video_to_latent(rej_path, vae, args.n_frames, device, reference_image=rej_ref_frame)
```

## How TI2V Works Now

### Video Sequence Structure

**T2V (Text-to-Video)**:
```
Input: [frame_1, frame_2, ..., frame_n]
VAE encoding: → [latent_1, latent_2, ..., latent_n]
Channels: 48
```

**TI2V (Text+Image-to-Video)**:
```
Input: [reference_image, frame_1, frame_2, ..., frame_{n-1}]
VAE encoding: → [ref_latent, latent_1, latent_2, ..., latent_{n-1}]
Channels: 48 (same as T2V!)
```

The reference image becomes the first frame in the temporal sequence, naturally providing conditioning through the transformer's temporal attention mechanisms.

### Training Flow

1. **Extract reference image**: First frame of video (in pixel space)
2. **Prepend to sequence**: [ref_image, video_frames[1:]]
3. **Encode through VAE**: Entire sequence encoded together
4. **Transformer forward**: Processes sequence with temporal attention
5. **First frame provides context**: Reference image latent influences all subsequent frames through attention

## Expected Output

### During Training
```
TI2V model detected - will use first frame as reference image
Encoding videos with reference images prepended...

Epoch 1/10
Step 1: loss=0.6931, accuracy=0.5000, reward_margin=0.0000
✓ Training progressing normally
```

### Tensor Shapes
```
Video frames: [n_frames, C=3, H=256, W=256]
VAE latents: [n_frames, C=48, H=16, W=16]  ✅ Correct channel count
Transformer input: [B=1, C=48, T=n_frames, H=16, W=16]  ✅ Correct format
```

## Benefits of This Approach

1. **Correct architecture**: Matches how TI2V models are designed
2. **Natural conditioning**: Reference image influences generation through temporal attention
3. **No channel mismatch**: Maintains expected 48 channels throughout
4. **Efficient**: Single VAE encoding pass for entire sequence
5. **Flexible**: Same pipeline handles both T2V and TI2V with simple flag

## Files Modified

1. **`azure/train_dpo_azure.py`**:
   - Modified `load_video_to_latent()` function signature and logic
   - Simplified `compute_dpo_loss()` to use prepending instead of concatenation
   - Removed all concatenation and splitting logic

## Verification

The fix has been applied and the code is ready to run. The error should no longer occur because:
- ✅ Reference image is prepended to video sequence in pixel space
- ✅ VAE encodes entire sequence together
- ✅ Output has correct 48 channels
- ✅ Transformer receives properly formatted input

## Next Steps

Run training:
```bash
bash azure/submit_job.sh
```

The training should now start successfully with TI2V-5B-Diffusers model using first-frame reference images.

## Status

✅ **FIX APPLIED AND READY**

The TI2V implementation now correctly handles reference images by prepending them to the video sequence before VAE encoding, rather than concatenating in latent space.
