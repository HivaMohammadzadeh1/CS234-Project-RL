# Quick Start: Running DPO Training with Qwen3-VL-2B

This guide shows you exactly how to run DPO training with Qwen3-VL-2B as the text encoder.

## Option 1: Azure ML (Recommended for Large-Scale Training)

### Step 1: Configure the Submission Script

Edit `azure/submit_job.sh` and set:

```bash
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"  # Use Qwen3-VL for text encoding
```

The default configuration is already set to use Qwen3-VL!

### Step 2: Submit the Job

```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
bash azure/submit_job.sh
```

This will:
1. Connect to your Azure ML workspace
2. Upload data if needed (or skip if `SKIP_UPLOAD=true`)
3. Submit training job with Qwen3-VL-2B as text encoder
4. Start training on GPU compute

### Step 3: Monitor Training

Watch the job progress in Azure ML Studio:
```
https://ml.azure.com
```

Look for your experiment: `wan22-dpo-training-<timestamp>`

---

## Option 2: Local/Direct Python (For Testing)

If you want to run locally or directly on a GPU machine:

### Basic Command

```bash
python azure/train_dpo_azure.py \
  --data video_rankings3_pairwise.json \
  --videos-dir /path/to/videos \
  --output-dir ./outputs \
  --text-encoder Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 1 \
  --grad-accum 4 \
  --epochs 10 \
  --beta 0.1 \
  --lr 1e-6
```

### With Train/Test Split

```bash
python azure/train_dpo_azure.py \
  --data video_rankings3_pairwise.json \
  --videos-dir /path/to/videos \
  --output-dir ./outputs \
  --text-encoder Qwen/Qwen3-VL-2B-Instruct \
  --train-only \
  --batch-size 1 \
  --epochs 10
```

---

## Option 3: Without Qwen3-VL (Use Default Text Encoder)

If you want to use the default text encoder from Wan2.2 instead:

### Azure ML
Edit `azure/submit_job.sh`:
```bash
TEXT_ENCODER=""  # Empty = use default
```

### Direct Python
Simply omit the `--text-encoder` argument:
```bash
python azure/train_dpo_azure.py \
  --data video_rankings3_pairwise.json \
  --videos-dir /path/to/videos \
  --output-dir ./outputs \
  --batch-size 1 \
  --epochs 10
```

---

## What Happens During Training?

When using Qwen3-VL-2B:

1. **Model Loading**:
   - Downloads `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (~28GB)
   - Downloads `Qwen/Qwen3-VL-2B-Instruct` (~4GB)
   - Creates learned projection layer (896 → 4096 dims)

2. **Training**:
   - Wan2.2 transformer: **TRAINS** (policy network)
   - Text projector: **TRAINS** (embedding adapter)
   - Qwen3-VL: **FROZEN** (text encoding only)
   - VAE: **FROZEN** (video encoding only)

3. **Checkpoints**:
   - `checkpoint-N/`: Transformer weights
   - `checkpoint-N/text_projector.pt`: Projection layer weights
   - `best_model/`: Best model based on accuracy
   - `final_model/`: Final model after all epochs

---

## Key Configuration Options

### Model Settings
- `--text-encoder`: Custom text encoder (default: None = use Wan2.2's encoder)
- `--model`: Video generation model (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)

### Training Settings
- `--batch-size`: Batch size per GPU (default: 1, increase if memory allows)
- `--grad-accum`: Gradient accumulation steps (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-6)
- `--beta`: DPO temperature (default: 0.1)

### Video Settings
- `--n-frames`: Frames per video (default: 8, reduce if OOM)
- `--num-inference-steps`: Diffusion steps for generation (default: 20)

### Data Settings
- `--train-only`: Use only training videos (excludes test set)
- `--test-indices`: Comma-separated test video indices

---

## Troubleshooting

### "Out of Memory"
```bash
# Reduce batch size and frames
python azure/train_dpo_azure.py \
  --batch-size 1 \
  --n-frames 4 \
  --grad-accum 8 \
  ...
```

### "transformers version too old"
```bash
pip install --upgrade transformers>=4.57.0
```

### "Model not found"
Make sure you have internet access for downloading from HuggingFace. The models will be cached in `/tmp/huggingface_cache` on Azure ML.

### "No valid pairs found"
Check that:
1. Videos directory path is correct
2. Video files exist and match names in preference JSON
3. Preference JSON is properly formatted

---

## Expected Output

```
======================================================================
  Loading Wan2.2 T2V Model
======================================================================
✓ Pipeline loaded

Loading custom text encoder: Qwen/Qwen3-VL-2B-Instruct
Loading Qwen3-VL model...
✓ Qwen3-VL text encoder loaded
✓ Text embedding projector created (896 -> 4096)

Creating reference transformer (frozen)...
Trainable parameters: 14,288,494,656 (transformer: 14,288,491,584, projector: 3,072)

======================================================================
  Starting DPO Training
======================================================================
Epoch 1/10: 100%|██████████| loss=0.6234, acc=0.652, margin=0.123
Epoch 1 Summary:
  Loss: 0.6234
  Accuracy: 0.652
  Reward Margin: 0.123
  ✓ New best model saved (accuracy: 0.652)
...
```

---

## Next Steps

After training completes:
1. Check `outputs/best_model/` for best checkpoint
2. Use the trained model for video generation
3. Evaluate on held-out test videos
4. Optionally fine-tune with different hyperparameters

See `QWEN_TEXT_ENCODER_GUIDE.md` for more technical details.
