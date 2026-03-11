#!/bin/bash
# One-command DPO training on Azure ML
# Similar to your fal.ai video generation setup!

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Wan2.2 DPO Fine-tuning on Azure ML                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "video_rankings3_pairwise.json" ]; then
    echo "❌ Error: video_rankings3_pairwise.json not found"
    echo "   Please run this from the project root directory"
    exit 1
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "⚠️  Azure CLI not installed"
    echo "   Installing azureml-core Python package instead..."
    pip install -q azureml-core azureml-mlflow
fi

# Step 1: Download videos if needed
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking training data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "wan22-dataset/videos" ]; then
    VIDEO_COUNT=$(ls wan22-dataset/videos/*.mp4 2>/dev/null | wc -l | tr -d ' ')
    if [ "$VIDEO_COUNT" -ge 380 ]; then
        echo "✅ Videos already downloaded ($VIDEO_COUNT files)"
        echo "   Location: ./wan22-dataset/videos/"
    else
        echo "📥 Downloading videos from HuggingFace..."
        ./download_data.sh
    fi
else
    echo "📥 Downloading videos from HuggingFace..."
    echo "   Dataset: hivamoh/wan22-physics-videos"
    echo "   Size: ~1.2 GB (384 videos)"
    echo ""
    ./download_data.sh
fi

# Verify download
if [ ! -d "wan22-dataset/videos" ]; then
    echo ""
    echo "❌ Error: Failed to download videos"
    echo "   Please check your internet connection and try again"
    exit 1
fi

VIDEO_COUNT=$(ls wan22-dataset/videos/*.mp4 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "✅ Data ready: $VIDEO_COUNT videos"
echo ""

# Step 2: Submit to Azure ML
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Submitting to Azure ML..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Azure ML Workspace:"
echo "  📍 ChemEngTraining (chemEng)"
echo "  🖥️  GPU: A100 80GB"
echo "  ⏱️  Training time: ~2-4 hours"
echo "  💰 Cost: ~\$11"
echo ""

# Submit job
./azure/submit_job.sh

# Check if submission was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Training job submitted successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📊 Monitor your training:"
    echo "   1. Click the Portal URL above"
    echo "   2. Watch metrics: loss, accuracy, reward_margin"
    echo "   3. View live logs in 'Outputs + logs' tab"
    echo ""
    echo "⏳ Expected timeline:"
    echo "   [  0 min] Data uploading to Azure"
    echo "   [ 10 min] GPU provisioning"
    echo "   [ 25 min] Environment building (first time)"
    echo "   [ 30 min] Training starts"
    echo "   [180 min] Training completes"
    echo "   [190 min] Model saved to Blob Storage"
    echo ""
    echo "📥 After training completes, download with:"
    echo "   az ml job download --name YOUR_RUN_ID --output-name outputs"
    echo ""
    echo "🎉 Your fine-tuned model will generate better videos!"
    echo ""
else
    echo ""
    echo "❌ Job submission failed"
    echo "   Check the error messages above"
    echo "   Try: python azure/check_setup.py --config azure/config.json"
    echo ""
    exit 1
fi
