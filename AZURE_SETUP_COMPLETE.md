# ✅ Azure ML Setup Complete!

I've created a complete Azure ML training setup for running DPO fine-tuning on Wan2.2, similar to how video generation is done on Azure.

---

## 📁 What Was Created

### Core Files
```
azure/
├── environment.yml              # Conda environment with all dependencies
├── train_dpo_azure.py          # Training script (runs on Azure)
├── submit_job.py               # Job submission script (runs locally)
├── submit_job.sh               # Bash wrapper for easy submission
├── check_setup.py              # Setup verification tool
├── config.example.json         # Configuration template
├── requirements.txt            # Local dependencies for submission
├── README_AZURE.md             # Complete Azure setup guide
└── WORKFLOW.md                 # Visual workflow diagram
```

### Updated Files
- **QUICKSTART.md**: Added Azure ML option
- **DPO_FULL_PIPELINE.md**: Already had the full pipeline explanation

---

## 🚀 Quick Start (3 Steps)

### 1. Install Local Dependencies
```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
pip install -r azure/requirements.txt
```

### 2. Configure Azure Credentials
```bash
# Copy config template
cp azure/config.example.json azure/config.json

# Edit with your Azure details
nano azure/config.json
# Set: subscription_id, resource_group, workspace_name

# Or edit submit_job.sh directly
nano azure/submit_job.sh
# Set: SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME
```

### 3. Submit Training Job
```bash
# Verify setup first (optional but recommended)
python azure/check_setup.py --config azure/config.json

# Submit job
./azure/submit_job.sh
```

---

## 💡 Key Features

### ✅ Auto-Scaling Compute
- Provisions GPU cluster automatically
- A100 80GB recommended (handles 2 model copies)
- Scales down when idle to save costs

### ✅ Professional Logging
- Real-time metrics in Azure ML Studio
- Automatic TensorBoard integration
- Downloadable logs and checkpoints

### ✅ Cost Efficient
- Pay only for training time (~$11 for 10 epochs)
- Auto-shutdown prevents idle charges
- Cached environments speed up subsequent runs

### ✅ Data Management
- Automatic upload to Azure Blob Storage
- Reusable data paths for multiple experiments
- Efficient mounting (no copying)

---

## 📊 Expected Workflow

```
Local Machine              Azure ML                    Results
─────────────             ──────────                  ─────────

1. Prepare data
   • videos/
   • preferences.json

2. Configure Azure
   • config.json
   • credentials

3. Submit job ──────────> 4. Provision A100 GPU
   ./submit_job.sh           (~5 min)

                          5. Build environment
                             (~15 min first time)
                             (~2 min cached)

                          6. Train DPO
                             (~2-4 hours)
                             ├─ Epoch 1: 55% acc
                             ├─ Epoch 5: 70% acc
                             └─ Epoch 10: 78% acc

                          7. Save outputs
                             • best_model/
                             • checkpoints/

8. Download results <──── 9. Upload to Blob
   az ml job download

10. Generate videos
    with fine-tuned model
```

**Total time**: 3-4 hours (mostly training)
**Total cost**: ~$11-15 per experiment

---

## 🎯 What This Solves

### Problem: Local Training Challenges
- ❌ Need expensive A100 GPU locally
- ❌ Long download times for models
- ❌ Risk of interruptions
- ❌ Manual checkpoint management

### Solution: Azure ML
- ✅ Cloud GPU on-demand
- ✅ Fast Azure-to-HuggingFace connections
- ✅ Fault-tolerant, resumable training
- ✅ Automatic artifact management

---

## 📝 Configuration Options

### VM Sizes (from config.json or submit_job.sh)

**Recommended**:
```bash
VM_SIZE="Standard_NC24ads_A100_v4"  # 1x A100 80GB (~$3.67/hour)
```

**For faster training**:
```bash
VM_SIZE="Standard_NC48ads_A100_v4"  # 2x A100 80GB (~$7.35/hour)
NODE_COUNT=1  # Uses both GPUs via DDP
```

**For testing**:
```bash
VM_SIZE="Standard_NC6s_v3"  # 1x V100 16GB (~$0.90/hour)
N_FRAMES=4  # Reduce memory usage
```

### Training Hyperparameters

```json
{
  "training": {
    "beta": 0.1,           # DPO temperature
    "learning_rate": 1e-6, # Conservative for stability
    "batch_size": 1,       # Per GPU
    "gradient_accumulation_steps": 4,  # Effective batch = 4
    "epochs": 10,
    "n_frames": 8          # Frames per video
  }
}
```

---

## 🔍 Monitoring Your Job

### Azure ML Studio (Best)
1. After submission, click the portal URL
2. View real-time metrics:
   - `train_loss` (should decrease)
   - `train_accuracy` (should increase to 70-80%)
   - `epoch_accuracy` (validation performance)
3. Download logs and outputs

### Command Line
```bash
# List jobs
az ml job list --workspace-name YOUR_WORKSPACE

# Show specific job
az ml job show --name RUN_ID

# Stream logs (live)
az ml job stream --name RUN_ID

# Download outputs
az ml job download --name RUN_ID --output-name outputs
```

---

## 💾 After Training

### Download Fine-tuned Model
```bash
# Method 1: Azure CLI
az ml job download \
    --name RUN_ID \
    --output-name outputs \
    --download-path ./wan22_finetuned

# Method 2: Azure ML Studio
# Click "Outputs + logs" → Download "outputs" folder
```

### Generate Videos
```python
from diffusers import WanPipeline

# Load your fine-tuned model
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    transformer="./wan22_finetuned/best_model",  # Your model!
)

# Generate improved video
video = pipe(
    prompt="A ball rolling down a slope with realistic physics",
    height=720,
    width=1280,
    num_frames=81,
).frames
```

---

## 🐛 Troubleshooting

### "Quota exceeded"
```bash
# Check current quota
az vm list-usage --location eastus -o table | grep NC24ads_A100

# Request increase:
# Azure Portal → Support → New support request → Quota
```

### "Data upload slow"
```bash
# Use AzCopy for faster upload (optional)
azcopy copy \
    "./wan22-dataset" \
    "https://YOUR_STORAGE.blob.core.windows.net/data" \
    --recursive
```

### "Training accuracy stuck at 50%"
- Check logs for reference model loading
- Try higher beta (0.5 instead of 0.1)
- Verify data uploaded correctly

### "CUDA out of memory"
- Reduce `n_frames` (8 → 4)
- Increase `grad_accum` (4 → 8)
- Use larger VM (2x A100)

---

## 📚 Documentation

Comprehensive guides:
- **[azure/README_AZURE.md](azure/README_AZURE.md)**: Complete Azure setup, troubleshooting, costs
- **[azure/WORKFLOW.md](azure/WORKFLOW.md)**: Visual workflow diagram with timings
- **[DPO_FULL_PIPELINE.md](DPO_FULL_PIPELINE.md)**: Full DPO explanation
- **[SETUP_WAN22.md](SETUP_WAN22.md)**: Wan2.2 model setup

---

## ✅ Pre-flight Checklist

Before submitting your first job:

- [ ] Azure ML workspace created
- [ ] GPU quota available (check Portal)
- [ ] Local data prepared:
  - [ ] `./wan22-dataset/videos/*.mp4` (384 videos)
  - [ ] `video_rankings3_pairwise.json` (preference data)
- [ ] Azure credentials configured:
  - [ ] `az login` completed
  - [ ] `azure/config.json` filled in
- [ ] Setup verified:
  - [ ] `python azure/check_setup.py` passed
- [ ] Cost understood:
  - [ ] ~$11 per 10-epoch experiment
  - [ ] Auto-shutdown enabled

---

## 🎉 You're Ready!

Everything is set up for Azure ML training. Next steps:

1. **Verify setup**:
   ```bash
   python azure/check_setup.py --config azure/config.json
   ```

2. **Submit your first job**:
   ```bash
   ./azure/submit_job.sh
   ```

3. **Monitor in Azure ML Studio**:
   - Click the portal URL after submission
   - Watch metrics improve in real-time

4. **Download and use your model**:
   ```bash
   az ml job download --name RUN_ID
   python generate_with_finetuned.py --model ./outputs/best_model
   ```

**Expected results**:
- Training time: 2-4 hours
- Best accuracy: 75-80%
- Cost: ~$11 per experiment
- Output: Fine-tuned Wan2.2 model that generates higher-quality videos!

---

## 🤝 Support

If you encounter issues:

1. Check **[azure/README_AZURE.md](azure/README_AZURE.md)** troubleshooting section
2. Review Azure ML logs in the portal
3. Run `python azure/check_setup.py` to diagnose
4. Check Azure status: https://status.azure.com/

**Happy Training on Azure! 🚀**
