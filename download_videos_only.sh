#!/bin/bash
# Download ONLY videos (not latents) to save disk space

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

# Check available disk space (need at least 2 GB)
AVAILABLE_GB=$(df -g . | awk 'NR==2 {print $4}')
echo "Available disk space: ${AVAILABLE_GB} GB"
if [ "$AVAILABLE_GB" -lt 2 ]; then
    echo ""
    echo "❌ ERROR: Not enough disk space!"
    echo "   Need: 2 GB minimum"
    echo "   Available: ${AVAILABLE_GB} GB"
    echo ""
    echo "Please free up some disk space and try again."
    exit 1
fi

echo "Downloading from HuggingFace Hub..."
echo "Dataset: hivamoh/wan22-physics-videos"
echo "Downloading: videos/ only (~1.2 GB)"
echo ""

# Clone with sparse checkout (videos only)
echo "Setting up sparse checkout (videos only)..."
git clone --filter=blob:none --sparse https://huggingface.co/datasets/hivamoh/wan22-physics-videos wan22-dataset-temp

cd wan22-dataset-temp
git sparse-checkout set videos
cd ..

# Move videos to correct location
echo ""
echo "Extracting videos..."
mkdir -p wan22-dataset
mv wan22-dataset-temp/videos wan22-dataset/
rm -rf wan22-dataset-temp

# Verify download
if [ -d "wan22-dataset/videos" ]; then
    VIDEO_COUNT=$(ls wan22-dataset/videos/*.mp4 2>/dev/null | wc -l)
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
