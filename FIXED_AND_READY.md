# ✅ ALL FIXED - Ready to Run!

## What Was Fixed

1. ✅ **Model format issue** - Now uses `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
2. ✅ **Azure ML data path resolution** - Fixed DataReferenceConfiguration handling
3. ✅ **Data upload disabled** - Re-enabled data upload to Azure Blob Storage
4. ✅ **Qwen3-VL integration** - Added support for custom text encoder

## Run Training NOW

Just one command:

```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
bash azure/submit_job.sh
```

That's it! The script will:
- ✅ Upload your videos and preference data (if needed)
- ✅ Create GPU compute cluster (if needed)
- ✅ Download Wan2.2-T2V-A14B-Diffusers (~28GB)
- ✅ Download Qwen3-VL-2B-Instruct (~4GB)
- ✅ Start DPO training with learned projection layer

## What's Training

**Video Generation**: Wan2.2-T2V-A14B-Diffusers
**Text Encoding**: Qwen3-VL-2B-Instruct (NEW!)
**Trainable**: Transformer (14B) + Projector (3M)

## Monitor Progress

Go to: https://ml.azure.com

Navigate to:
- Workspace: `ChemEngTraining`
- Experiments → `wan22-dpo-training-<timestamp>`

You'll see:
- Real-time logs
- Training metrics (loss, accuracy, reward margin)
- GPU utilization

## Configuration

Edit `azure/submit_job.sh` if you want to change:

```bash
TEXT_ENCODER="Qwen/Qwen3-VL-2B-Instruct"  # or "" for default
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=10
BETA=0.1
LR=1e-6
N_FRAMES=8
```

## Expected Output

Training will produce:
- Checkpoints every 500 steps
- Best model based on accuracy
- Final model after all epochs
- All saved to Azure Blob Storage

## Need Help?

📖 **Detailed guide**: See `RUN_WITH_QWEN.md`
🔧 **Technical details**: See `QWEN_TEXT_ENCODER_GUIDE.md`
🐛 **What was fixed**: See `AZURE_FIX_SUMMARY.md`

## Quick Troubleshooting

**Out of memory?**
```bash
# In submit_job.sh, set:
BATCH_SIZE=1
N_FRAMES=4
GRAD_ACCUM=8
```

**Want default text encoder instead of Qwen?**
```bash
# In submit_job.sh, set:
TEXT_ENCODER=""
```

**Data already uploaded?**
```bash
# In submit_job.sh, set:
SKIP_UPLOAD=true
```

## You're Ready!

Everything is configured and fixed. Just run:

```bash
bash azure/submit_job.sh
```

Training will start automatically. Check Azure ML Studio for progress!

Good luck with your DPO training! 🚀
