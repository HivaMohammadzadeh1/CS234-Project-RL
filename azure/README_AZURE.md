# Azure ML Training Guide for Wan2.2 DPO Fine-tuning

This guide shows you how to run the full DPO fine-tuning on Azure ML, similar to how video generation is typically done on Azure.

## 📋 Prerequisites

### 1. Azure ML Workspace
- Active Azure subscription
- Azure ML workspace created
- Sufficient quota for GPU VMs (A100 recommended)

### 2. Local Setup
```bash
pip install azureml-core azureml-mlflow
```

### 3. Data Preparation
Ensure you have:
- `video_rankings3_pairwise.json` (preference data)
- `./wan22-dataset/videos/` (video files)

---

## 🚀 Quick Start

### Option 1: Using Bash Script (Easiest)

1. **Configure your Azure settings**:
   ```bash
   # Edit azure/submit_job.sh
   nano azure/submit_job.sh

   # Set these variables:
   SUBSCRIPTION_ID="your-subscription-id"
   RESOURCE_GROUP="your-resource-group"
   WORKSPACE_NAME="your-workspace-name"
   ```

2. **Make executable and run**:
   ```bash
   chmod +x azure/submit_job.sh
   ./azure/submit_job.sh
   ```

### Option 2: Using Python Script

```bash
python azure/submit_job.py \
    --subscription-id "YOUR_SUBSCRIPTION_ID" \
    --resource-group "YOUR_RESOURCE_GROUP" \
    --workspace-name "YOUR_WORKSPACE_NAME" \
    --experiment-name "wan22-dpo-training" \
    --compute-name "gpu-cluster-a100" \
    --vm-size "Standard_NC24ads_A100_v4" \
    --epochs 10 \
    --beta 0.1 \
    --lr 1e-6
```

---

## 💻 Recommended Azure VM Sizes

### For DPO Fine-tuning (Requires 2 model copies + optimizer state):

| VM Size | GPU | VRAM | Cost/Hour | Recommended For |
|---------|-----|------|-----------|-----------------|
| **Standard_NC24ads_A100_v4** | 1x A100 80GB | 80GB | ~$3.67 | **Best choice** (full model) |
| Standard_NC48ads_A100_v4 | 2x A100 80GB | 160GB | ~$7.35 | Distributed training |
| Standard_NC96ads_A100_v4 | 4x A100 80GB | 320GB | ~$14.69 | Large-scale experiments |
| Standard_ND96amsr_A100_v4 | 8x A100 80GB | 640GB | ~$32.77 | Multi-node training |

**Note**: DPO requires loading both policy and reference models, so 80GB A100 is recommended minimum.

### For Testing (Reduced frames/batch size):

| VM Size | GPU | VRAM | Cost/Hour | Recommended For |
|---------|-----|------|-----------|-----------------|
| Standard_NC24s_v3 | 4x V100 16GB | 64GB | ~$3.06 | Testing (reduce n_frames to 4) |
| Standard_NC12s_v3 | 2x V100 16GB | 32GB | ~$1.53 | Small experiments |

---

## 📁 File Structure

```
azure/
├── environment.yml           # Conda environment with dependencies
├── train_dpo_azure.py       # Main training script (runs on Azure)
├── submit_job.py            # Job submission script (runs locally)
├── submit_job.sh            # Bash wrapper for easy submission
└── README_AZURE.md          # This file
```

---

## 🔧 Configuration Options

### Training Hyperparameters

```bash
--beta 0.1                  # DPO temperature (0.01-1.0)
                            # Lower = more conservative
                            # Higher = more aggressive

--lr 1e-6                   # Learning rate
                            # A100: 1e-6 to 5e-6
                            # V100: 5e-7 to 1e-6

--batch-size 1              # Batch size per GPU
                            # A100 80GB: 1-2
                            # V100 16GB: 1

--grad-accum 4              # Gradient accumulation steps
                            # Effective batch = batch_size * grad_accum
                            # Increase for memory efficiency

--epochs 10                 # Training epochs
                            # Start with 5-10 epochs

--n-frames 8                # Frames per video
                            # A100 80GB: 8-16
                            # V100 16GB: 4-8
```

### Compute Settings

```bash
--compute-name "gpu-cluster-a100"
--vm-size "Standard_NC24ads_A100_v4"
--max-nodes 1               # Max nodes in cluster (auto-scale)
--node-count 1              # Nodes for this job
```

### Data Settings

```bash
--local-data-dir "./wan22-dataset"      # Local data to upload
--pref-file "video_rankings3_pairwise.json"
--data-path ""                          # Or use existing Azure path
```

---

## 📊 Monitoring Your Job

### Azure ML Studio (Recommended)
1. Click the portal URL printed after submission
2. View real-time metrics:
   - `train_loss`
   - `train_accuracy`
   - `train_reward_margin`
   - `epoch_accuracy`

### Azure CLI
```bash
# List recent runs
az ml job list --workspace-name YOUR_WORKSPACE

# Show specific job
az ml job show --name RUN_ID

# Stream logs
az ml job stream --name RUN_ID
```

### Python SDK
```python
from azureml.core import Workspace, Experiment

ws = Workspace.from_config()
exp = Experiment(ws, "wan22-dpo-training")
run = exp.get_runs().__next__()

# Get metrics
metrics = run.get_metrics()
print(f"Best accuracy: {metrics.get('best_accuracy')}")

# Download outputs
run.download_files(prefix="outputs/best_model", output_directory="./local_outputs")
```

---

## 💾 Output Structure

After training completes, outputs are saved to Azure Blob Storage:

```
outputs/
├── best_model/             # Best model based on validation accuracy
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── ...
├── final_model/            # Final model after all epochs
├── checkpoint-500/         # Periodic checkpoints
├── checkpoint-1000/
└── ...
```

### Download Outputs

```bash
# Using Azure ML CLI
az ml job download --name RUN_ID --output-name outputs --download-path ./downloads

# Or using Python
from azureml.core import Run
run = Run.get_context()
run.download_files(prefix="outputs", output_directory="./local_outputs")
```

---

## 🔥 Common Issues & Solutions

### Issue 1: "Quota exceeded"
**Error**: `The maximum number of Standard_NC24ads_A100_v4 cores is 0`

**Solution**: Request quota increase
```bash
# Check current quota
az vm list-usage --location eastus -o table | grep "NC24ads_A100"

# Request increase via Azure Portal:
# Support → New Support Request → Service and subscription limits (quotas)
```

### Issue 2: "Out of memory"
**Error**: `CUDA out of memory`

**Solutions**:
```bash
# Option 1: Reduce frames
--n-frames 4  # Instead of 8

# Option 2: Reduce batch size (already at 1)
--batch-size 1

# Option 3: Increase gradient accumulation
--grad-accum 8  # Instead of 4

# Option 4: Use larger VM
--vm-size "Standard_NC48ads_A100_v4"  # 2x A100
```

### Issue 3: "Environment build timeout"
**Error**: Environment taking too long to build

**Solution**: Pre-build and register environment
```python
from azureml.core import Workspace, Environment

ws = Workspace.from_config()
env = Environment.from_conda_specification(
    name="wan22-dpo",
    file_path="azure/environment.yml"
)
env.register(ws)
env.build(ws).wait_for_completion(show_output=True)
```

### Issue 4: "Data upload too slow"
**Error**: Uploading 1.2GB of videos is slow

**Solutions**:
```bash
# Option 1: Upload once, reuse data path
# First run: uploads data
python azure/submit_job.py ...

# Subsequent runs: use existing data
python azure/submit_job.py ... \
    --data-path "azureml://datastores/workspaceblobstore/paths/wan22_dpo_data"

# Option 2: Use AzCopy (much faster)
azcopy copy "./wan22-dataset" \
    "https://YOUR_STORAGE.blob.core.windows.net/wan22-data" \
    --recursive
```

### Issue 5: "Training accuracy stuck at 50%"
**Possible causes**:
1. Reference model not loading correctly
2. Beta too low
3. Data issues

**Solutions**:
```bash
# Try higher beta
--beta 0.5  # Instead of 0.1

# Check logs for reference model loading
az ml job stream --name RUN_ID | grep "reference"

# Verify data uploaded correctly
az storage blob list --account-name ACCOUNT --container-name CONTAINER
```

---

## 🎯 Expected Results

### Training Time
- **10 epochs on A100 80GB**: ~2-4 hours
- **10 epochs on 2x A100 80GB**: ~1-2 hours
- **10 epochs on V100 16GB** (n_frames=4): ~4-6 hours

### Metrics
- **Epoch 1**: Accuracy ~55-60% (learning)
- **Epoch 5**: Accuracy ~65-70% (improving)
- **Epoch 10**: Accuracy ~70-80% (converged)
- **Best accuracy**: 75-80% expected

### Cost Estimates
- **A100 80GB**: ~$3.67/hour × 3 hours = ~$11 per experiment
- **2x A100 80GB**: ~$7.35/hour × 1.5 hours = ~$11 per experiment

---

## 🔄 Distributed Training (Multi-GPU)

For faster training with multiple GPUs:

```bash
python azure/submit_job.py \
    --vm-size "Standard_NC48ads_A100_v4" \
    --node-count 1 \
    --batch-size 2 \
    --grad-accum 2 \
    ...
```

**Note**: Distributed training is automatically configured via PyTorchConfiguration.

---

## 📝 Advanced: Custom Environment

If you need additional packages:

1. **Edit `environment.yml`**:
   ```yaml
   dependencies:
     - pip:
         - your-custom-package
   ```

2. **Force rebuild**:
   ```bash
   python azure/submit_job.py ... --force-rebuild-env
   ```

---

## 🧪 Testing Your Setup

### Quick Test Job (5 minutes, ~$0.30)

```bash
python azure/submit_job.py \
    --subscription-id "YOUR_ID" \
    --resource-group "YOUR_RG" \
    --workspace-name "YOUR_WS" \
    --vm-size "Standard_NC6s_v3" \
    --epochs 1 \
    --n-frames 2 \
    --batch-size 1
```

This runs a minimal job to verify:
- ✅ Compute provisioning works
- ✅ Environment builds successfully
- ✅ Data uploads correctly
- ✅ Training script runs

---

## 📚 Additional Resources

- **Azure ML Documentation**: https://docs.microsoft.com/azure/machine-learning/
- **VM Pricing**: https://azure.microsoft.com/pricing/details/machine-learning/
- **Quota Requests**: https://docs.microsoft.com/azure/azure-portal/supportability/per-vm-quota-requests
- **Cost Management**: https://docs.microsoft.com/azure/cost-management-billing/

---

## ✅ Checklist

Before submitting your job:

- [ ] Azure ML workspace created and accessible
- [ ] GPU quota sufficient (check `az vm list-usage`)
- [ ] Local data prepared (`./wan22-dataset/`, `video_rankings3_pairwise.json`)
- [ ] Azure credentials configured
- [ ] `azure/submit_job.sh` configured with your workspace details
- [ ] Test job completed successfully

---

## 🎉 Next Steps After Training

1. **Download fine-tuned model**:
   ```bash
   az ml job download --name RUN_ID --output-name outputs
   ```

2. **Generate videos with fine-tuned model**:
   ```python
   from diffusers import WanPipeline

   pipe = WanPipeline.from_pretrained(
       "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
       transformer="./outputs/best_model"
   )
   video = pipe("Your prompt here").frames
   ```

3. **Compare base vs fine-tuned**:
   ```bash
   python generate_with_finetuned.py \
       --model ./outputs/best_model \
       --prompt "A ball rolling down a slope" \
       --compare
   ```

---

## 💬 Support

If you encounter issues:

1. Check Azure ML logs in the portal
2. Review this README's troubleshooting section
3. Check Azure ML service health: https://status.azure.com/
4. File an issue with job details and error logs

---

**Happy Training! 🚀**
