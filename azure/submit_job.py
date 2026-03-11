#!/usr/bin/env python3
"""
Azure ML Job Submission Script for Wan2.2 DPO Fine-tuning

This script submits a training job to Azure ML with:
- GPU compute cluster (A100 recommended)
- Custom environment with Wan2.2 dependencies
- Data mounting from Azure Blob Storage or uploading
- Distributed training support (optional)
"""

import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference


def get_or_create_compute(workspace, compute_name, vm_size="Standard_NC24ads_A100_v4", max_nodes=1):
    """Get existing compute or create new one."""
    try:
        compute_target = ComputeTarget(workspace=workspace, name=compute_name)
        print(f"Found existing compute: {compute_name}")
    except ComputeTargetException:
        print(f"Creating new compute: {compute_name}")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size,
            max_nodes=max_nodes,
            idle_seconds_before_scaledown=1800,  # 30 minutes
        )
        compute_target = ComputeTarget.create(workspace, compute_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    return compute_target


def get_or_create_environment(workspace, env_name="wan22-dpo"):
    """Get existing environment or create from YAML."""
    try:
        env = Environment.get(workspace=workspace, name=env_name)
        print(f"Found existing environment: {env_name}")
    except Exception:
        print(f"Creating new environment: {env_name}")
        env = Environment.from_conda_specification(
            name=env_name,
            file_path="azure/environment.yml"
        )
        # Use Docker with GPU support
        env.docker.enabled = True
        env.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
        env.python.user_managed_dependencies = False

        # Register environment
        env.register(workspace=workspace)

    return env


def upload_data_to_azure(workspace, local_path, pref_file, datastore_name="workspaceblobstore"):
    """Upload local data to Azure Blob Storage (.mp4 files and preference JSON)."""
    from azureml.core import Datastore
    import glob
    import shutil
    import tempfile
    import os

    datastore = Datastore.get(workspace, datastore_name)

    # Create temp directory with videos and preference file
    print(f"Preparing data from {local_path}...")
    temp_dir = tempfile.mkdtemp(prefix="wan22_upload_")
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir)

    # Copy .mp4 files
    video_files = glob.glob(os.path.join(local_path, "*.mp4"))
    print(f"Found {len(video_files)} video files (.mp4 only)")
    print(f"Copying videos to temporary directory...")

    for video_file in video_files:
        shutil.copy2(video_file, videos_dir)

    # Copy preference JSON file
    if os.path.exists(pref_file):
        print(f"Copying preference file: {pref_file}")
        shutil.copy2(pref_file, temp_dir)
    else:
        raise FileNotFoundError(f"Preference file not found: {pref_file}")

    print(f"Uploading {len(video_files)} videos + preference file to Azure Blob Storage...")
    try:
        datastore.upload(
            src_dir=temp_dir,
            target_path="wan22_dpo_data",
            overwrite=False,
            show_progress=True
        )
    finally:
        # Clean up temp directory
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

    # Return datastore paths
    videos_path = f"azureml://datastores/{datastore_name}/paths/wan22_dpo_data/videos"
    pref_file_name = os.path.basename(pref_file)
    pref_path = f"azureml://datastores/{datastore_name}/paths/wan22_dpo_data/{pref_file_name}"

    return videos_path, pref_path


def submit_training_job(
    workspace,
    experiment_name="wan22-dpo-training",
    compute_name="gpu-cluster-a100",
    vm_size="Standard_NC24ads_A100_v4",
    max_nodes=1,
    # Data paths
    data_path=None,  # If None, will upload from local
    local_data_dir="./wan22-dataset",
    pref_file="video_rankings3_pairwise.json",
    # Model configuration
    model=None,  # Base video generation model (e.g., Wan-AI/Wan2.2-TI2V-5B-Diffusers)
    text_encoder=None,  # Custom text encoder (e.g., Qwen/Qwen3-VL-2B-Instruct)
    # Training hyperparameters
    beta=0.1,
    lr=1e-6,
    batch_size=1,
    grad_accum=4,
    epochs=10,
    n_frames=8,
    num_inference_steps=20,
    # Weights & Biases
    use_wandb=False,
    wandb_project="wan22-dpo",
    wandb_run_name=None,
    # Distributed training
    node_count=1,
):
    """Submit DPO training job to Azure ML."""

    print("=" * 70)
    print("  Azure ML DPO Training Job Submission")
    print("=" * 70)
    print()

    # Get or create compute
    print("Setting up compute...")
    compute_target = get_or_create_compute(workspace, compute_name, vm_size, max_nodes)

    # Get or create environment
    print("\nSetting up environment...")
    env = get_or_create_environment(workspace)

    # Handle data
    print("\nSetting up data...")
    datastore = workspace.get_default_datastore()

    if data_path is None:
        if not os.path.exists(local_data_dir):
            raise ValueError(f"Local data directory not found: {local_data_dir}")
        if not os.path.exists(pref_file):
            raise ValueError(f"Preference file not found: {pref_file}")

        print("Uploading local data to Azure Blob Storage...")
        _, _ = upload_data_to_azure(workspace, local_data_dir, pref_file)
        print(f"✓ Videos uploaded")
        print(f"✓ Preference file uploaded")
    else:
        print("✓ Skipping upload - using existing data on Azure")

    # Create DataReference objects for proper data mounting
    pref_file_name = os.path.basename(pref_file)

    # Setup output directory
    output = OutputFileDatasetConfig(
        name="outputs",
        destination=(workspace.get_default_datastore(), "wan22_dpo_outputs")
    )

    # Create DataReference for preference file (will be downloaded to compute)
    pref_data_ref = DataReference(
        datastore=datastore,
        data_reference_name="pref_data",
        path_on_datastore=f"wan22_dpo_data/{pref_file_name}",
        mode='download',  # Download the JSON file
        path_on_compute="pref_data"  # Local path on compute node
    )

    # Create DataReference for videos directory (will be downloaded to compute)
    # Note: Using download instead of mount for better reliability
    videos_data_ref = DataReference(
        datastore=datastore,
        data_reference_name="videos_data",
        path_on_datastore="wan22_dpo_data/videos",
        mode='download',  # Download videos directory (mount was unreliable)
        path_on_compute="videos_data"  # Local path on compute node
    )

    print(f"Created data references:")
    print(f"  Preference file: wan22_dpo_data/{pref_file_name} (download to pref_data/)")
    print(f"  Videos: wan22_dpo_data/videos (download to videos_data/)")

    # Prepare arguments with path_on_compute strings
    # Azure ML will mount/download data to these paths and pass them to the script
    arguments = [
        "--data", pref_data_ref.path_on_compute,
        "--videos-dir", videos_data_ref.path_on_compute,
        "--output-dir", output,
        "--beta", beta,
        "--lr", lr,
        "--batch-size", batch_size,
        "--grad-accum", grad_accum,
        "--epochs", epochs,
        "--n-frames", n_frames,
        "--num-inference-steps", num_inference_steps,
    ]

    # Add model if specified
    if model:
        arguments.extend(["--model", model])

    # Add text encoder if specified
    if text_encoder:
        arguments.extend(["--text-encoder", text_encoder])

    # Add W&B arguments if specified
    if use_wandb:
        arguments.append("--use-wandb")
        arguments.extend(["--wandb-project", wandb_project])
        if wandb_run_name:
            arguments.extend(["--wandb-run-name", wandb_run_name])

    # Create script run config
    print("\nCreating script run configuration...")
    src = ScriptRunConfig(
        source_directory=".",
        script="azure/train_dpo_azure.py",
        arguments=arguments,
        compute_target=compute_target,
        environment=env,
    )

    # Add data references to run config (this tells Azure ML to mount/download the data)
    src.run_config.data_references = {
        pref_data_ref.data_reference_name: pref_data_ref.to_config(),
        videos_data_ref.data_reference_name: videos_data_ref.to_config(),
    }

    # Setup distributed training if using multiple nodes
    if node_count > 1:
        print(f"Configuring distributed training with {node_count} nodes...")
        distributed_config = PyTorchConfiguration(node_count=node_count)
        src.run_config.distributed_job_config = distributed_config

    # Submit experiment
    print(f"\nSubmitting experiment: {experiment_name}")
    print("(This may take 30-60 seconds...)")

    try:
        experiment = Experiment(workspace=workspace, name=experiment_name)
        run = experiment.submit(src)
    except Exception as e:
        print(f"\n❌ Job submission failed!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Azure credentials: az login")
        print("2. Verify compute cluster is running: az ml compute show -n gpu-h100-2x")
        print("3. Check workspace quotas in Azure Portal")
        raise

    print("\n" + "=" * 70)
    print("  Job Submitted Successfully!")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Run ID: {run.id}")
    print(f"Portal URL: {run.get_portal_url()}")
    print("=" * 70)
    print()
    print("Monitor your job:")
    print(f"  - Azure ML Studio: {run.get_portal_url()}")
    print(f"  - Or run: az ml job show --name {run.id}")
    print()

    return run


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Submit Wan2.2 DPO training to Azure ML")

    # Azure ML settings
    parser.add_argument("--subscription-id", type=str, required=True,
                       help="Azure subscription ID")
    parser.add_argument("--resource-group", type=str, required=True,
                       help="Azure resource group")
    parser.add_argument("--workspace-name", type=str, required=True,
                       help="Azure ML workspace name")
    parser.add_argument("--experiment-name", type=str, default="wan22-dpo-training",
                       help="Experiment name")

    # Compute settings
    parser.add_argument("--compute-name", type=str, default="gpu-cluster-a100",
                       help="Compute cluster name")
    parser.add_argument("--vm-size", type=str, default="Standard_NC24ads_A100_v4",
                       help="VM size (A100 recommended)")
    parser.add_argument("--max-nodes", type=int, default=1,
                       help="Max nodes in cluster")
    parser.add_argument("--node-count", type=int, default=1,
                       help="Number of nodes for this job")

    # Data settings
    parser.add_argument("--data-path", type=str, default=None,
                       help="Azure data path (if None, uploads from local)")
    parser.add_argument("--local-data-dir", type=str, default="./wan22-dataset",
                       help="Local data directory to upload")
    parser.add_argument("--pref-file", type=str, default="video_rankings3_pairwise.json",
                       help="Preference data JSON file")

    # Model configuration
    parser.add_argument("--model", type=str, default=None,
                       help="Base video generation model (e.g., Wan-AI/Wan2.2-TI2V-5B-Diffusers)")
    parser.add_argument("--text-encoder", type=str, default=None,
                       help="Custom text encoder model (e.g., Qwen/Qwen3-VL-2B-Instruct)")

    # Training hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO temperature")
    parser.add_argument("--lr", type=float, default=1e-6,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--n-frames", type=int, default=8,
                       help="Frames per video")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                       help="Number of diffusion steps (for evaluation/generation)")

    # Upload control
    parser.add_argument("--skip-upload", action="store_true",
                       help="Skip data upload (use when data already exists on Azure)")

    # Weights & Biases
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="wan22-dpo",
                       help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="W&B run name")

    # Action
    parser.add_argument("--wait", action="store_true",
                       help="Wait for job to complete")

    args = parser.parse_args()

    # Connect to workspace
    print("Connecting to Azure ML workspace...")
    workspace = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    print(f"✓ Connected to workspace: {workspace.name}")
    print()

    # Submit job
    effective_data_path = args.data_path if args.data_path else ("skip" if args.skip_upload else None)
    run = submit_training_job(
        workspace=workspace,
        experiment_name=args.experiment_name,
        compute_name=args.compute_name,
        vm_size=args.vm_size,
        max_nodes=args.max_nodes,
        data_path=effective_data_path,
        local_data_dir=args.local_data_dir,
        pref_file=args.pref_file,
        model=args.model,
        text_encoder=args.text_encoder,
        beta=args.beta,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        n_frames=args.n_frames,
        num_inference_steps=args.num_inference_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        node_count=args.node_count,
    )

    # Optionally wait for completion
    if args.wait:
        print("Waiting for job to complete...")
        run.wait_for_completion(show_output=True)
        print("\nJob completed!")
        print(f"Status: {run.get_status()}")


if __name__ == "__main__":
    main()
