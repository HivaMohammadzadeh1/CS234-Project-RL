#!/bin/bash
# Simple download using git (no Python dependencies needed)

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

echo "Downloading from HuggingFace Hub..."
echo "Dataset: hivamoh/wan22-physics-videos"
echo "Size: ~1.2 GB"
echo ""

# Install git-lfs if needed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs via conda..."
    conda install -y -c conda-forge git-lfs
    git lfs install
fi

# Clone repository (this will download all videos)
echo "Cloning dataset repository..."
git clone https://huggingface.co/datasets/hivamoh/wan22-physics-videos wan22-dataset-temp

# Move videos to correct location
mkdir -p wan22-dataset
mv wan22-dataset-temp/videos wan22-dataset/
rm -rf wan22-dataset-temp

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
