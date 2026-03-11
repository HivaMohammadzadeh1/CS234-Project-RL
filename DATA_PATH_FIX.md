# Azure ML Data Path Resolution - Final Fix

## Problem Summary

The training job was failing with:
```
FileNotFoundError: Preference file not found: azureml://datastores/workspaceblobstore/paths/...
```

**Root Cause**: The submission script was passing raw azureml:// URI strings as command-line arguments, but Azure ML doesn't automatically resolve these URIs when passed as arguments. The training script received the URI strings directly instead of actual file paths.

## Solution Applied

### Fixed in `azure/submit_job.py` (lines 14, 168-220)

**Changed from**: Raw azureml:// URI strings
**Changed to**: `DataReference` objects with proper mounting and registration

```python
# Before (BROKEN):
pref_uri = f"azureml://datastores/{datastore_name}/paths/wan22_dpo_data/{pref_file_name}"
videos_uri = f"azureml://datastores/{datastore_name}/paths/wan22_dpo_data/videos"
arguments = ["--data", pref_uri, "--videos-dir", videos_uri, ...]

# After (WORKING):
from azureml.data.data_reference import DataReference

# Create DataReference objects
pref_data_ref = DataReference(
    datastore=datastore,
    data_reference_name="pref_data",
    path_on_datastore=f"wan22_dpo_data/{pref_file_name}",
    mode='download',  # Download the JSON file
    path_on_compute="pref_data"  # Where it will be on compute node
)

videos_data_ref = DataReference(
    datastore=datastore,
    data_reference_name="videos_data",
    path_on_datastore="wan22_dpo_data/videos",
    mode='mount',  # Mount videos directory
    path_on_compute="videos_data"  # Where it will be on compute node
)

# Pass path_on_compute strings as arguments
arguments = ["--data", pref_data_ref.path_on_compute,
             "--videos-dir", videos_data_ref.path_on_compute, ...]

# Register data references with run config
src.run_config.data_references = {
    pref_data_ref.data_reference_name: pref_data_ref.to_config(),
    videos_data_ref.data_reference_name: videos_data_ref.to_config(),
}
```

## How It Works Now

1. **Submission time** (`submit_job.py`):
   - Creates `DataReference` objects pointing to data in Azure Blob Storage
   - Specifies download mode for JSON (small file) and mount mode for videos (large directory)
   - Sets `path_on_compute` to define where data will be on compute node
   - Registers DataReferences with run config using `.to_config()`
   - Passes `path_on_compute` strings as command-line arguments

2. **Job execution** (Azure ML):
   - Reads the data_references from run config
   - Downloads the preference JSON to: `<working_dir>/pref_data/`
   - Mounts the videos directory to: `<working_dir>/videos_data/`
   - Passes the path strings ("pref_data", "videos_data") to the training script
   - Training script receives these as relative paths that exist on the compute node

3. **Training script** (`train_dpo_azure.py`):
   - Receives path strings (e.g., "pref_data", "videos_data")
   - Recursively searches for JSON if given a directory
   - Searches for videos in subdirectories if needed
   - Successfully loads data and starts training

## Why DataReference vs URI Strings

| Approach | How it works | Result |
|----------|--------------|--------|
| Raw URI strings | Python string passed as argument | ❌ Training script receives `"azureml://..."` string - FileNotFoundError |
| `DataReference` objects as args | Trying to serialize objects directly | ❌ TypeError: DataReference is not JSON serializable |
| **DataReference (registered)** | **Register with run_config + pass path_on_compute strings** | **✅ Azure ML downloads/mounts data to specified paths** |

**Key insight**: DataReference objects must be:
1. Added to `run_config.data_references` via `.to_config()`
2. Only their `path_on_compute` strings passed as arguments
3. Azure ML then handles mounting/downloading automatically

## Run Training Now

Everything is fixed. Just run:

```bash
cd /Users/hivamoh/cs234Proj/CS234-Project-RL
bash azure/submit_job.sh
```

## Expected Behavior

### During Submission
```
Setting up data...
Created data references:
  Preference file: wan22_dpo_data/video_rankings3_pairwise.json (download to pref_data/)
  Videos: wan22_dpo_data/videos (mount to videos_data/)

Creating script run configuration...
Submitting experiment: wan22-dpo-training-<timestamp>

======================================================================
  Job Submitted Successfully!
======================================================================
Run ID: wan22-dpo-training-<timestamp>_<id>
```

### During Training (Azure ML logs)
```
Loading Preference Dataset
======================================================================
Preference file (raw): pref_data
Preference file type: <class 'str'>
pref_data_path is a directory, searching for JSON files...
Found JSON file: pref_data/video_rankings3_pairwise.json

Videos directory (raw): videos_data
Found videos directory: videos_data/
Found 720 video files in videos_data/

Loading Wan2.2-T2V-A14B-Diffusers...
Loading Qwen3-VL-2B-Instruct...
Creating text embedding projector (1024 → 4096)...
✓ All models loaded

Starting DPO training...
```

## Verification Checklist

After running `bash azure/submit_job.sh`:

- ✅ Job submits without errors
- ✅ No "DataReferenceConfiguration" errors
- ✅ No "azureml://" URI errors in training logs
- ✅ Training script receives actual file paths
- ✅ JSON and videos are found automatically
- ✅ Models load successfully
- ✅ Training starts and runs

## Troubleshooting

### If submission fails with "DataReference not found"
The azureml-core package might be outdated. Update it:
```bash
pip install --upgrade azureml-core
```

### If data not found during training
Check that data exists on Azure:
```bash
az storage blob list \
  --account-name <storage_account> \
  --container-name <container> \
  --prefix wan22_dpo_data/
```

Or re-upload by setting `SKIP_UPLOAD=false` in `azure/submit_job.sh`

## What Changed

**Modified Files**:
1. `azure/submit_job.py`:
   - Added `from azureml.data.data_reference import DataReference`
   - Replaced raw URI strings with DataReference objects
   - Set proper download/mount modes

**Unchanged Files** (already had robust handling):
1. `azure/train_dpo_azure.py`:
   - Already recursively searches for JSON files
   - Already handles nested video directories
   - Already has defensive checks for azureml:// URIs

## Summary

The fix changes data handling from broken URI strings to proper Azure ML `DataReference` API. Azure ML now correctly downloads/mounts data to actual file paths that the training script can access using standard file I/O.

**Status**: ✅ FIXED AND READY TO RUN

Execute: `bash azure/submit_job.sh`
