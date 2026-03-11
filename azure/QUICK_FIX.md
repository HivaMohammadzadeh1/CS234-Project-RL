# Quick Fix for Azure ML Submission

## Issue
Missing package: `azureml-dataset-runtime`

## Solution

**Option 1: Install the missing package (Quickest)**
```bash
pip install azureml-dataset-runtime --upgrade
```

Then rerun:
```bash
bash azure/submit_job.sh
```

---

## Updated Approach (Simplified)

I've updated the code to use a simpler data mounting approach that should work better:

1. **Data is already uploaded** - Your files are in Azure Blob Storage at `wan22_dpo_data/`
2. **Mount the datastore** - The script now mounts the entire `wan22_dpo_data` folder
3. **Paths are relative** - Training script receives paths like `workspaceblobstore/wan22_dpo_data/videos`

---

## Alternative: Bypass Azure Dataset API

If you continue to have issues, you can:

1. **Pass direct blob URLs** to the training script
2. **Download at runtime** using Azure Blob Storage SDK
3. **Use shared file system** if available

---

## Model Note

Based on your notebooks (`Video Generation.ipynb`), I see you're using **CogVideoX-5b** for generation, not Wan2.2.

Should we:
- **Keep Wan2.2 for DPO training** (current approach)
- **Switch to CogVideoX-5b** to match your generation pipeline?

Let me know and I can update the training script accordingly!
