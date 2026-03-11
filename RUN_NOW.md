# 🚀 Run DPO Training Now (Simple Guide)

You already have:
- ✅ Azure ML workspace (ChemEngTraining)
- ✅ GPU quota
- ✅ Preference data (video_rankings3_pairwise.json)

Just 2 steps to start training!

---

## Step 1: Download Videos (5 minutes)

```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
./download_data.sh
```

**What this does**: Downloads 384 videos (~1.2 GB) from HuggingFace

**Progress**: You'll see:
```
Downloading from HuggingFace Hub...
Dataset: hivamoh/wan22-physics-videos
...
✓ Download complete!
  Videos: 384 files
```

---

## Step 2: Submit to Azure ML (Just like your fal.ai setup!)

```bash
./azure/submit_job.sh
```

**What this does**:
1. Uploads your data to Azure (5-10 min)
2. Starts A100 GPU (5 min)
3. Trains model (2-4 hours)
4. Saves fine-tuned model automatically

**You'll get a portal URL** like:
```
Portal URL: https://ml.azure.com/runs/wan22-dpo-training-...
```

**Click it** to watch training live! Just like monitoring your fal.ai generations.

---

## That's It! 🎉

Azure ML handles everything automatically:
- ✅ GPU provisioning
- ✅ Environment setup
- ✅ Training
- ✅ Checkpointing
- ✅ Model saving

**Cost**: ~$11 (3 hours on A100)
**Time**: 3-4 hours total

---

## While Training Runs

Monitor in Azure ML Studio (click the portal URL):
- Live metrics (loss, accuracy)
- GPU utilization
- Logs streaming
- Auto-saved checkpoints

---

## After Training

### Download Your Model
```bash
# Get RUN_ID from portal or terminal output
az ml job download \
    --name YOUR_RUN_ID \
    --output-name outputs \
    --download-path ./wan22_finetuned
```

### Generate Better Videos
```python
from diffusers import WanPipeline

# Your fine-tuned model!
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    transformer="./wan22_finetuned/best_model"
)

video = pipe("A ball rolling down a slope").frames
```

---

## Need Help?

### Check Setup First (Optional)
```bash
python azure/check_setup.py --config azure/config.json
```

### If Data Already Exists
Skip download if you already have videos:
```bash
# Just run this:
./azure/submit_job.sh
```

### Troubleshooting

**"Data not found"**
```bash
./download_data.sh  # Download first
```

**"Quota exceeded"**
- You said you have quota, but double-check: Azure Portal → Subscriptions → Usage + quotas → Search "A100"

**"Job failed"**
- Check logs in Azure ML Studio portal
- Look at "Outputs + logs" tab

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./download_data.sh` | Download training videos |
| `./azure/submit_job.sh` | Submit training to Azure |
| `python azure/check_setup.py` | Verify everything works |

---

**Ready?** Run this now:

```bash
./download_data.sh && ./azure/submit_job.sh
```

This downloads data THEN submits the job automatically! 🚀
