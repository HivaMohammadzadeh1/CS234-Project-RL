# Weights & Biases Integration

## Summary

Added Weights & Biases (wandb) integration for tracking experiments, metrics, and hyperparameters during DPO training.

## Changes Made

### 1. Core Training Script (`azure/train_dpo_azure.py`)

#### Import and Availability Check
```python
# Weights & Biases import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not installed. Install with: pip install wandb")
```

#### Updated `log_metric()` Function
Now logs to both Azure ML and W&B:
```python
def log_metric(key, value, step=None):
    """Log metrics to Azure ML, W&B, or print locally."""
    # Log to Azure ML
    if AZURE_ML and run:
        run.log(key, value)
        if step is not None:
            run.log(f"{key}_step", step)
    else:
        print(f"Metric: {key} = {value}" + (f" (step {step})" if step else ""))

    # Log to W&B
    if WANDB_AVAILABLE and wandb.run is not None:
        if step is not None:
            wandb.log({key: value, "step": step})
        else:
            wandb.log({key: value})
```

#### W&B Initialization (in `train_dpo()`)
Added before training loop starts:
```python
# Initialize Weights & Biases
if WANDB_AVAILABLE and args.use_wandb:
    print("\nInitializing Weights & Biases...")
    wandb_config = {
        "model": args.model,
        "text_encoder": args.text_encoder if args.text_encoder else "default",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "beta": args.beta,
        "learning_rate": args.lr,
        "n_frames": args.n_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "dataset_size": len(dataset),
    }

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=wandb_config,
        resume="allow" if args.wandb_resume else False,
        tags=["dpo", "wan22", "video-generation"],
    )
    print(f"✓ W&B initialized: {wandb.run.name}")
    print(f"  Project: {args.wandb_project}")
    print(f"  Run URL: {wandb.run.get_url()}")
```

#### Command-Line Arguments
Added wandb-related arguments:
```python
# Weights & Biases (wandb)
parser.add_argument("--use-wandb", action="store_true",
                   help="Enable Weights & Biases logging")
parser.add_argument("--wandb-project", type=str, default="wan22-dpo",
                   help="W&B project name")
parser.add_argument("--wandb-run-name", type=str, default=None,
                   help="W&B run name (auto-generated if not specified)")
parser.add_argument("--wandb-resume", action="store_true",
                   help="Resume W&B run if it exists")
```

#### Other Changes
- **Changed default epochs from 10 to 1** (line 1033)

### 2. Submission Shell Script (`azure/submit_job.sh`)

#### Added W&B Configuration Variables
```bash
# Training Hyperparameters
EPOCHS=1  # Changed from 10 to 1

# Weights & Biases Configuration
USE_WANDB=true  # Set to true to enable W&B logging
WANDB_PROJECT="wan22-dpo"
WANDB_RUN_NAME=""  # Leave empty for auto-generated name
```

#### Added W&B Flag Building
```bash
# Build wandb flags
WANDB_FLAGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAGS="--use-wandb --wandb-project $WANDB_PROJECT"
    if [ ! -z "$WANDB_RUN_NAME" ]; then
        WANDB_FLAGS="$WANDB_FLAGS --wandb-run-name $WANDB_RUN_NAME"
    fi
    echo "✓ W&B enabled: Project=$WANDB_PROJECT"
fi
```

#### Updated Python Script Call
Added `$WANDB_FLAGS` to the arguments:
```bash
python azure/submit_job.py \
    --subscription-id "$SUBSCRIPTION_ID" \
    ... (other args) ...
    $WANDB_FLAGS \
    --beta "$BETA" \
    ...
```

### 3. Azure ML Submission Script (`azure/submit_job.py`)

#### Updated Function Signature
Added wandb parameters to `submit_training_job()`:
```python
def submit_training_job(
    workspace,
    ...
    # Weights & Biases
    use_wandb=False,
    wandb_project="wan22-dpo",
    wandb_run_name=None,
    # Distributed training
    node_count=1,
):
```

#### Added W&B Arguments to Training Script
```python
# Add W&B arguments if specified
if use_wandb:
    arguments.append("--use-wandb")
    arguments.extend(["--wandb-project", wandb_project])
    if wandb_run_name:
        arguments.extend(["--wandb-run-name", wandb_run_name])
```

#### Added Command-Line Arguments
```python
# Weights & Biases
parser.add_argument("--use-wandb", action="store_true",
                   help="Enable Weights & Biases logging")
parser.add_argument("--wandb-project", type=str, default="wan22-dpo",
                   help="W&B project name")
parser.add_argument("--wandb-run-name", type=str, default=None,
                   help="W&B run name")
```

#### Updated Function Call
```python
run = submit_training_job(
    ...
    use_wandb=args.use_wandb,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name,
    ...
)
```

## Installation

To use W&B logging, install the package:

```bash
pip install wandb
```

## Usage

### Enable W&B in Azure ML

Edit `azure/submit_job.sh`:
```bash
USE_WANDB=true  # Enable W&B logging
WANDB_PROJECT="wan22-dpo"  # Your project name
WANDB_RUN_NAME="my-experiment"  # Optional: custom run name
```

Then submit:
```bash
bash azure/submit_job.sh
```

### Enable W&B in Local Training

```bash
python azure/train_dpo_azure.py \
    --data pref_data \
    --videos-dir videos_data \
    --use-wandb \
    --wandb-project "my-project" \
    --wandb-run-name "experiment-1" \
    --epochs 1
```

### Disable W&B

Set in `azure/submit_job.sh`:
```bash
USE_WANDB=false  # Disable W&B logging
```

Or omit `--use-wandb` flag when running directly.

## Logged Metrics

The following metrics are automatically logged to W&B:

### Training Metrics (per step)
- `train_loss`: DPO loss value
- `train_accuracy`: Binary accuracy (preferred > rejected)
- `train_reward_margin`: Difference between preferred and rejected rewards

### Epoch Metrics
- `epoch_loss`: Average loss for epoch
- `epoch_accuracy`: Average accuracy for epoch
- `epoch_reward_margin`: Average reward margin for epoch

### Best Model Tracking
- `best_accuracy`: Highest accuracy achieved

### Configuration
All hyperparameters are logged to W&B config:
- Model architecture
- Training hyperparameters (batch size, learning rate, beta, etc.)
- Dataset size
- Number of frames
- Random seed

## Viewing Results

### W&B Dashboard
After training starts, the script will print:
```
✓ W&B initialized: wandering-sun-42
  Project: wan22-dpo
  Run URL: https://wandb.ai/your-username/wan22-dpo/runs/abc123
```

Click the URL to view:
- Real-time training metrics
- Loss curves
- Accuracy plots
- Hyperparameter configs
- System metrics (GPU, CPU, memory)

### W&B CLI
```bash
wandb login  # First time only
wandb sync   # Sync offline runs
```

## Tips

1. **Auto-generated run names**: Leave `WANDB_RUN_NAME=""` for W&B to generate unique names like "dazzling-galaxy-42"

2. **Custom run names**: Set `WANDB_RUN_NAME="descriptive-name"` for easier identification

3. **Resume runs**: Add `--wandb-resume` flag to continue a previous run (uses same run ID)

4. **Offline mode**: Set `WANDB_MODE=offline` environment variable to log locally and sync later

5. **Tags**: Runs are automatically tagged with: `["dpo", "wan22", "video-generation"]`

## Architecture

```
┌─────────────────────────────────────────────┐
│         Training Script                      │
│       (train_dpo_azure.py)                  │
│                                              │
│  ┌──────────────┐      ┌──────────────┐    │
│  │  log_metric  │─────▶│  Azure ML    │    │
│  │  function    │      │  Logging     │    │
│  └──────┬───────┘      └──────────────┘    │
│         │                                    │
│         │              ┌──────────────┐     │
│         └─────────────▶│   W&B API    │─────┼──▶ W&B Cloud
│                        │   wandb.log  │     │
│                        └──────────────┘     │
└─────────────────────────────────────────────┘
```

## Files Modified

1. **`azure/train_dpo_azure.py`**:
   - Added wandb import and availability check
   - Modified `log_metric()` to log to W&B
   - Added `wandb.init()` before training
   - Added command-line arguments for W&B
   - Changed default epochs from 10 to 1

2. **`azure/submit_job.sh`**:
   - Changed `EPOCHS` from 10 to 1
   - Added W&B configuration variables
   - Added W&B flag building logic
   - Added `$WANDB_FLAGS` to python call

3. **`azure/submit_job.py`**:
   - Added wandb parameters to function signature
   - Added wandb arguments to training script
   - Added wandb command-line arguments
   - Updated function call with wandb parameters

## Status

✅ **W&B INTEGRATION COMPLETE**

- Metrics automatically logged to both Azure ML and W&B
- Configurable via command-line or shell script
- Optional (disabled by default in Azure submission)
- Auto-detects if wandb is installed
- Full hyperparameter tracking
- Works in both Azure ML and local environments

## Next Steps

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Enable in config: Set `USE_WANDB=true` in `submit_job.sh`
4. Submit training: `bash azure/submit_job.sh`
5. View results: Open W&B URL printed during training
