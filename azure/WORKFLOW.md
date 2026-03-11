# Azure ML DPO Training Workflow

## 📊 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR LOCAL MACHINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Prepare Data                                                │
│     └─ video_rankings3_pairwise.json                           │
│     └─ ./wan22-dataset/videos/*.mp4                            │
│                                                                 │
│  2. Configure Azure                                             │
│     └─ azure/config.json (credentials)                         │
│     └─ azure/submit_job.sh (parameters)                        │
│                                                                 │
│  3. Verify Setup                                                │
│     └─ python azure/check_setup.py                             │
│                                                                 │
│  4. Submit Job                                                  │
│     └─ ./azure/submit_job.sh                                   │
│          │                                                      │
└──────────┼──────────────────────────────────────────────────────┘
           │
           │ Upload data + Submit job
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AZURE ML WORKSPACE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5. Provision Compute                                           │
│     └─ Create/Scale GPU cluster                                │
│     └─ VM: Standard_NC24ads_A100_v4 (A100 80GB)                │
│                                                                 │
│  6. Build Environment                                           │
│     └─ Create Docker image from environment.yml                │
│     └─ Install: diffusers, torch, transformers, etc.           │
│                                                                 │
│  7. Mount Data                                                  │
│     └─ Mount uploaded videos from Blob Storage                 │
│     └─ Mount preference JSON                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │
           │ Start training
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING COMPUTE NODE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  8. Load Wan2.2 Model                                           │
│     └─ Download from HuggingFace                               │
│     └─ Load VAE (AutoencoderKLWan)                             │
│     └─ Load Pipeline (WanPipeline)                             │
│     └─ Create reference copy (frozen)                          │
│                                                                 │
│  9. Training Loop (10 epochs)                                   │
│     ┌─────────────────────────────┐                            │
│     │ For each epoch:             │                            │
│     │   For each batch:           │                            │
│     │     • Load video pair       │                            │
│     │     • Encode to latents     │                            │
│     │     • Compute DPO loss      │                            │
│     │     • Backprop + update     │                            │
│     │                             │                            │
│     │   Log metrics to Azure ML:  │                            │
│     │     • train_loss            │                            │
│     │     • train_accuracy        │                            │
│     │     • reward_margin         │                            │
│     │                             │                            │
│     │   Save checkpoint (if best) │                            │
│     └─────────────────────────────┘                            │
│                                                                 │
│  10. Save Outputs                                               │
│      └─ best_model/                                             │
│      └─ final_model/                                            │
│      └─ checkpoints/                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │
           │ Upload outputs to Blob Storage
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AZURE BLOB STORAGE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  11. Stored Outputs                                             │
│      └─ wan22_dpo_outputs/                                      │
│          ├─ best_model/                                         │
│          │   ├─ config.json                                     │
│          │   ├─ diffusion_pytorch_model.safetensors            │
│          │   └─ ...                                             │
│          ├─ final_model/                                        │
│          └─ checkpoint-*/                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │
           │ Download outputs
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR LOCAL MACHINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  12. Download Trained Model                                     │
│      └─ az ml job download --name RUN_ID                       │
│      └─ Or download via Azure ML Studio                        │
│                                                                 │
│  13. Generate Videos                                            │
│      └─ python generate_with_finetuned.py \                    │
│            --model ./outputs/best_model \                       │
│            --prompt "Your prompt"                               │
│                                                                 │
│  14. Compare Results                                            │
│      └─ Base model vs Fine-tuned model                         │
│      └─ Improved physics, motion, quality!                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Parallel View: What Happens Where

| Step | Location | Action | Time |
|------|----------|--------|------|
| **Setup** | Local | Configure credentials | 5 min |
| **Verify** | Local | Check setup script | 1 min |
| **Submit** | Local → Azure | Upload data + Submit job | 5-10 min |
| **Provision** | Azure | Start GPU cluster | 3-5 min |
| **Build** | Azure | Build environment (first time) | 10-15 min |
| **Build** | Azure | Use cached environment (subsequent) | 1-2 min |
| **Train** | Azure GPU | DPO training (10 epochs) | 2-4 hours |
| **Save** | Azure → Storage | Upload checkpoints | 2-5 min |
| **Download** | Storage → Local | Download trained model | 5-10 min |
| **Generate** | Local | Create videos with fine-tuned model | Varies |

**Total time (first run)**: ~3-4 hours
**Total time (subsequent runs)**: ~2-3 hours

---

## 💰 Cost Breakdown

For a typical 10-epoch training run:

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| A100 80GB GPU | $3.67/hour | 3 hours | $11.01 |
| Blob Storage | $0.02/GB/month | 5 GB | $0.10 |
| Data Transfer | $0.087/GB | 2 GB | $0.17 |
| **Total** | | | **~$11.28** |

**Cost optimization tips**:
- Use cached environment (saves 10-15 min)
- Use existing data path (saves upload time)
- Enable auto-shutdown (avoids idle costs)
- Use spot instances (50-80% discount, but may be preempted)

---

## 📊 Monitoring Dashboard

While training runs, monitor via Azure ML Studio:

```
Azure ML Studio Dashboard
├─ Overview
│   ├─ Status: Running
│   ├─ Duration: 01:23:45
│   └─ Compute: Standard_NC24ads_A100_v4
│
├─ Metrics (Live)
│   ├─ train_loss: 0.4523 ↓
│   ├─ train_accuracy: 0.732 ↑
│   ├─ epoch_accuracy: 0.756 ↑
│   └─ best_accuracy: 0.768
│
├─ Logs (Live Stream)
│   ├─ azureml-logs/
│   ├─ driver/
│   └─ system/
│
├─ Outputs
│   ├─ best_model/ (auto-saved)
│   ├─ checkpoint-500/
│   └─ checkpoint-1000/
│
└─ Child Runs (if distributed)
    └─ None (single GPU)
```

---

## 🔧 Troubleshooting Flow

```
Job Failed?
   │
   ├─ Check Status in Azure ML Studio
   │   │
   │   ├─ "Provisioning Failed"
   │   │   └─ → Check quota: az vm list-usage
   │   │   └─ → Request increase via Support
   │   │
   │   ├─ "Environment Build Failed"
   │   │   └─ → Check environment.yml syntax
   │   │   └─ → Try: env.build(ws).wait_for_completion()
   │   │
   │   ├─ "Data Not Found"
   │   │   └─ → Verify data uploaded: az storage blob list
   │   │   └─ → Check data path in arguments
   │   │
   │   └─ "CUDA Out of Memory"
   │       └─ → Reduce --n-frames (8→4)
   │       └─ → Increase --grad-accum (4→8)
   │       └─ → Use larger VM (2x A100)
   │
   └─ Check Logs
       └─ Download: az ml job download --name RUN_ID
       └─ View: cat azureml-logs/70_driver_log.txt
```

---

## ✅ Success Checklist

Before submitting:
- [ ] Azure credentials configured (`az login`)
- [ ] Workspace accessible (`python azure/check_setup.py`)
- [ ] Data prepared locally (videos + JSON)
- [ ] GPU quota sufficient (check Azure Portal)
- [ ] Config file updated (`azure/config.json`)

After training:
- [ ] Job completed successfully (check Studio)
- [ ] Metrics look good (accuracy >70%)
- [ ] Outputs saved to Blob Storage
- [ ] Model downloaded locally
- [ ] Test generation works

---

## 🚀 Quick Commands Reference

```bash
# Setup
pip install -r azure/requirements.txt
az login
python azure/check_setup.py

# Submit
./azure/submit_job.sh

# Monitor
az ml job show --name RUN_ID
az ml job stream --name RUN_ID

# Download
az ml job download --name RUN_ID --output-name outputs

# Generate
python generate_with_finetuned.py \
    --model ./outputs/best_model \
    --prompt "Your prompt"
```

---

## 📚 Additional Resources

- **Azure ML Studio**: https://ml.azure.com/
- **Pricing Calculator**: https://azure.microsoft.com/pricing/calculator/
- **Quota Requests**: https://portal.azure.com/#blade/Microsoft_Azure_Capacity/QuotaMenuBlade
- **Documentation**: https://docs.microsoft.com/azure/machine-learning/
