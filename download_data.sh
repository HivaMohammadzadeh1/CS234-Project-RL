#!/bin/bash
# Download full video dataset for DPO training

echo "=========================================="
echo "  Downloading Wan2.2 DPO Training Data"
echo "=========================================="
echo ""

# Check if data already exists
if [ -d "wan22-dataset/videos" ]; then
    VIDEO_COUNT=$(ls wan22-dataset/videos/*.mp4 2>/dev/null | wc -l)
    if [ "$VIDEO_COUNT" -ge 380 ]; then
        echo "✓ Data already downloaded ($VIDEO_COUNT videos found)"
        echo ""
        echo "Ready to submit training job!"
        exit 0
    fi
fi

# Download from HuggingFace
echo "Downloading from HuggingFace Hub..."
echo "Dataset: hivamoh/wan22-physics-videos"
echo "Size: ~1.2 GB"
echo ""

# Try downloading with existing huggingface_hub first (no upgrade to avoid conflicts)
python << 'PYTHON_SCRIPT'
import sys

try:
    # Try with existing huggingface_hub
    from huggingface_hub import snapshot_download

    print("Downloading videos from HuggingFace...")
    snapshot_download(
        repo_id="hivamoh/wan22-physics-videos",
        repo_type="dataset",
        local_dir="wan22-dataset",
        allow_patterns="videos/*.mp4"
    )
    print("Download complete!")

except ImportError:
    print("ERROR: huggingface_hub not found in conda environment")
    print("Installing in conda environment...")
    sys.exit(1)

except Exception as e:
    print(f"ERROR: Download failed: {e}")
    sys.exit(1)
PYTHON_SCRIPT

# If Python script failed, try installing huggingface_hub in conda
if [ $? -ne 0 ]; then
    echo ""
    echo "Installing huggingface_hub via conda..."
    conda install -y -c conda-forge huggingface_hub

    # Try again
    python << 'PYTHON_SCRIPT2'
from huggingface_hub import snapshot_download

print("Downloading videos from HuggingFace (retry)...")
snapshot_download(
    repo_id="hivamoh/wan22-physics-videos",
    repo_type="dataset",
    local_dir="wan22-dataset",
    allow_patterns="videos/*.mp4"
)
print("Download complete!")
PYTHON_SCRIPT2
fi

# Verify download
if [ -d "wan22-dataset/videos" ]; then
    VIDEO_COUNT=$(ls wan22-dataset/videos/*.mp4 | wc -l)
    echo ""
    echo "=========================================="
    echo "✓ Download complete!"
    echo "  Videos: $VIDEO_COUNT files"
    echo "  Location: ./wan22-dataset/videos/"
    echo "=========================================="
    echo ""
    echo "Next step: Submit training job"
    echo "  ./azure/submit_job.sh"
else
    echo ""
    echo "✗ Download failed"
    echo "  Please check your internet connection and try again"
    exit 1
fi
