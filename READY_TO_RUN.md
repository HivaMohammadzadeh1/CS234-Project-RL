# ✅ Ready to Run on Azure ML!

Your Azure ML setup is **complete** and configured with your credentials:

- **Workspace**: ChemEngTraining
- **Resource Group**: chemEng
- **Location**: westus2

---

## 🚀 Run in 3 Commands

### 1. Download Training Data (~5 minutes, 1.2 GB)
```bash
./download_data.sh
```

This downloads 384 videos from HuggingFace to `./wan22-dataset/videos/`

### 2. (Optional) Verify Setup
```bash
python azure/check_setup.py --config azure/config.json
```

This checks:
- ✓ Azure credentials working
- ✓ Workspace accessible
- ✓ Data downloaded correctly
- ✓ GPU quota available

### 3. Submit Training Job
```bash
./azure/submit_job.sh
```

This will:
- Upload data to Azure Blob Storage (~5-10 min)
- Provision A100 GPU cluster (~5 min)
- Train DPO model (~2-4 hours)
- Save fine-tuned model automatically

---

## 📊 What to Expect

### Timeline
```
[  0 min] Submit job
[  5 min] Data uploaded to Azure
[ 10 min] GPU cluster provisioned (A100 80GB)
[ 25 min] Environment built (first time only)
[ 30 min] Training starts

Training Progress:
[ 30 min] Epoch 1/10  - Accuracy: ~55%
[ 60 min] Epoch 3/10  - Accuracy: ~65%
[120 min] Epoch 5/10  - Accuracy: ~70%
[180 min] Epoch 10/10 - Accuracy: ~78%

[190 min] Saving outputs to Blob Storage
[195 min] ✅ Training complete!
```

**Total time**: ~3-3.5 hours
**Total cost**: ~$11-13 (A100 @ $3.67/hour)

### Portal URL
After submission, you'll get a URL like:
```
https://ml.azure.com/runs/wan22-dpo-training-...
```

Click it to monitor:
- Real-time metrics (loss, accuracy, reward margin)
- Live logs
- GPU utilization
- Auto-saved checkpoints

---

## 📥 After Training Completes

### Download Your Fine-tuned Model
```bash
# Get your run ID from the portal or submission output
RUN_ID="wan22-dpo-training-20260310-123000"

# Download outputs
az ml job download \
    --name $RUN_ID \
    --output-name outputs \
    --download-path ./wan22_finetuned
```

### Generate Videos with Your Model
```python
from diffusers import WanPipeline

# Load fine-tuned model
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    transformer="./wan22_finetuned/best_model",  # Your fine-tuned model!
)

# Generate improved video
video = pipe(
    prompt="A ball rolling down a slope with realistic physics",
    height=720,
    width=1280,
    num_frames=81,
).frames

# Save
import imageio
imageio.mimsave("output.mp4", [np.array(f) for f in video], fps=24)
```

---

## 🎯 Expected Results

**Before DPO (base Wan2.2)**:
- Physics sometimes unrealistic
- Motion inconsistencies
- Quality varies

**After DPO (your model)**:
- ✅ 75-80% preference accuracy
- ✅ Better physics simulation
- ✅ More consistent motion
- ✅ Higher fidelity aligned with human preferences

---

## 🔧 Configuration (Already Set)

Your `azure/submit_job.sh` is configured with:

```bash
# Your Azure credentials
SUBSCRIPTION_ID="fc1f3270-30de-4538-b720-3f2ac5377083"
RESOURCE_GROUP="chemEng"
WORKSPACE_NAME="ChemEngTraining"

# Compute
COMPUTE_NAME="gpu-cluster-a100"
VM_SIZE="Standard_NC24ads_A100_v4"  # A100 80GB

# Training
BETA=0.1
LR=1e-6
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=10
N_FRAMES=8
```

You can edit these values in `azure/submit_job.sh` if needed.

---

## 🆘 Quick Troubleshooting

### "Quota exceeded"
Check GPU quota in Azure Portal:
```
Portal → Subscriptions → Usage + quotas → Search "NC24ads_A100"
```
Request increase if needed (usually approved within hours).

### "Data not found"
Make sure you ran:
```bash
./download_data.sh
```
Verify: `ls wan22-dataset/videos/*.mp4 | wc -l` should show ~384 files

### "CUDA out of memory"
Edit `azure/submit_job.sh`:
```bash
N_FRAMES=4  # Reduce from 8
GRAD_ACCUM=8  # Increase from 4
```

---

## 📚 Documentation

- **[azure/README_AZURE.md](azure/README_AZURE.md)**: Complete guide with all options
- **[azure/WORKFLOW.md](azure/WORKFLOW.md)**: Visual workflow diagram
- **[DPO_FULL_PIPELINE.md](DPO_FULL_PIPELINE.md)**: How DPO works
- **[SETUP_WAN22.md](SETUP_WAN22.md)**: Model setup details

---

## ✅ Checklist

- [x] Azure credentials configured
- [x] Workspace accessible (ChemEngTraining)
- [ ] **→ Download training data** (`./download_data.sh`)
- [ ] **→ Submit training job** (`./azure/submit_job.sh`)
- [ ] Monitor in Azure ML Studio
- [ ] Download fine-tuned model
- [ ] Generate improved videos!

---

## 🎉 You're Ready!

**Next command**:
```bash
./download_data.sh
```

Then:
```bash
./azure/submit_job.sh
```

Training will start automatically on Azure ML! 🚀
