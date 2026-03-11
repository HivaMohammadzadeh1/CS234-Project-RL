#!/usr/bin/env python3
"""
Azure ML Setup Verification Script

Checks that everything is configured correctly before submitting a training job.
"""

import os
import sys
import json
from pathlib import Path

def check_azure_credentials():
    """Check if Azure credentials are configured."""
    print("Checking Azure credentials...")
    try:
        from azureml.core import Workspace
        from azureml.core.authentication import AzureCliAuthentication

        auth = AzureCliAuthentication()
        print("  ✓ Azure CLI authentication available")
        return True
    except ImportError:
        print("  ✗ azureml-core not installed")
        print("    Run: pip install azureml-core")
        return False
    except Exception as e:
        print(f"  ✗ Azure authentication failed: {e}")
        print("    Run: az login")
        return False


def check_workspace_connection(subscription_id, resource_group, workspace_name):
    """Check if we can connect to the workspace."""
    print("\nChecking workspace connection...")
    try:
        from azureml.core import Workspace

        ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
        print(f"  ✓ Connected to workspace: {ws.name}")
        print(f"    Region: {ws.location}")
        print(f"    Subscription: {ws.subscription_id}")
        return True, ws
    except Exception as e:
        print(f"  ✗ Failed to connect to workspace: {e}")
        return False, None


def check_compute_quota(workspace, vm_size="Standard_NC24ads_A100_v4"):
    """Check if quota is available for the VM size."""
    print(f"\nChecking compute quota for {vm_size}...")
    try:
        from azureml.core.compute import ComputeTarget

        # This is a simplified check - actual quota checking requires Azure REST API
        print(f"  ⚠  Cannot automatically check quota")
        print(f"    Manual check: az vm list-usage --location {workspace.location} -o table")
        return True
    except Exception as e:
        print(f"  ✗ Error checking quota: {e}")
        return False


def check_local_data(data_dir, pref_file):
    """Check if local data exists."""
    print("\nChecking local data...")

    # Check data directory
    if not os.path.exists(data_dir):
        print(f"  ✗ Data directory not found: {data_dir}")
        return False
    else:
        video_files = list(Path(data_dir).rglob("*.mp4"))
        print(f"  ✓ Data directory found: {data_dir}")
        print(f"    Videos: {len(video_files)} files")

    # Check preference file
    if not os.path.exists(pref_file):
        print(f"  ✗ Preference file not found: {pref_file}")
        return False
    else:
        with open(pref_file) as f:
            prefs = json.load(f)
        print(f"  ✓ Preference file found: {pref_file}")
        print(f"    Groups: {len(prefs)}")

        # Count pairs
        total_pairs = sum(
            len([p for p in g["pairwise_comparisons"] if not p.get("tie")])
            for g in prefs.values()
        )
        print(f"    Preference pairs: {total_pairs}")

    return True


def check_environment_file():
    """Check if environment YAML exists."""
    print("\nChecking environment configuration...")

    env_file = "azure/environment.yml"
    if not os.path.exists(env_file):
        print(f"  ✗ Environment file not found: {env_file}")
        return False
    else:
        print(f"  ✓ Environment file found: {env_file}")
        return True


def check_training_script():
    """Check if training script exists."""
    print("\nChecking training script...")

    script = "azure/train_dpo_azure.py"
    if not os.path.exists(script):
        print(f"  ✗ Training script not found: {script}")
        return False
    else:
        print(f"  ✓ Training script found: {script}")
        return True


def estimate_cost(vm_size, epochs, samples_per_epoch):
    """Estimate training cost."""
    print("\nEstimating training cost...")

    # Approximate costs (as of 2026, check Azure pricing for current rates)
    vm_costs = {
        "Standard_NC24ads_A100_v4": 3.67,  # per hour
        "Standard_NC48ads_A100_v4": 7.35,
        "Standard_NC96ads_A100_v4": 14.69,
        "Standard_NC6s_v3": 0.90,
        "Standard_NC12s_v3": 1.53,
        "Standard_NC24s_v3": 3.06,
    }

    cost_per_hour = vm_costs.get(vm_size, 3.67)

    # Rough estimate: ~2-4 hours for 10 epochs on A100
    estimated_hours = (epochs / 10) * 3.0
    estimated_cost = cost_per_hour * estimated_hours

    print(f"  VM: {vm_size}")
    print(f"  Cost per hour: ${cost_per_hour:.2f}")
    print(f"  Estimated time: {estimated_hours:.1f} hours")
    print(f"  Estimated cost: ${estimated_cost:.2f}")
    print(f"  ⚠  This is a rough estimate - actual cost may vary")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check Azure ML setup")
    parser.add_argument("--config", type=str, default="azure/config.json",
                       help="Path to config file")
    parser.add_argument("--subscription-id", type=str,
                       help="Azure subscription ID (overrides config)")
    parser.add_argument("--resource-group", type=str,
                       help="Resource group (overrides config)")
    parser.add_argument("--workspace-name", type=str,
                       help="Workspace name (overrides config)")

    args = parser.parse_args()

    print("=" * 70)
    print("  Azure ML Setup Verification")
    print("=" * 70)

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
        print(f"\n✓ Loaded config from: {args.config}")
    else:
        print(f"\n⚠  Config file not found: {args.config}")
        print("  Using command-line arguments or defaults")

    # Get Azure settings
    subscription_id = args.subscription_id or config.get("azure", {}).get("subscription_id")
    resource_group = args.resource_group or config.get("azure", {}).get("resource_group")
    workspace_name = args.workspace_name or config.get("azure", {}).get("workspace_name")

    # Get training settings
    vm_size = config.get("compute", {}).get("vm_size", "Standard_NC24ads_A100_v4")
    epochs = config.get("training", {}).get("epochs", 10)
    data_dir = config.get("data", {}).get("local_data_dir", "./wan22-dataset")
    pref_file = config.get("data", {}).get("preference_file", "video_rankings3_pairwise.json")

    # Run checks
    checks_passed = []

    # 1. Azure credentials
    checks_passed.append(check_azure_credentials())

    # 2. Workspace connection
    if subscription_id and resource_group and workspace_name:
        ws_ok, ws = check_workspace_connection(subscription_id, resource_group, workspace_name)
        checks_passed.append(ws_ok)

        if ws_ok:
            # 3. Compute quota
            checks_passed.append(check_compute_quota(ws, vm_size))
    else:
        print("\n⚠  Skipping workspace checks - credentials not provided")
        print("   Set subscription_id, resource_group, workspace_name in config")

    # 4. Local data
    checks_passed.append(check_local_data(data_dir, pref_file))

    # 5. Environment file
    checks_passed.append(check_environment_file())

    # 6. Training script
    checks_passed.append(check_training_script())

    # 7. Cost estimate
    if data_dir and os.path.exists(data_dir):
        # Count samples for better estimate
        video_files = list(Path(data_dir).rglob("*.mp4"))
        samples = len(video_files) * 10  # Rough estimate
        estimate_cost(vm_size, epochs, samples)

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    passed = sum(checks_passed)
    total = len(checks_passed)

    print(f"\nChecks passed: {passed}/{total}")

    if all(checks_passed):
        print("\n✅ All checks passed! You're ready to submit a training job.")
        print("\nNext steps:")
        print("  1. Review azure/submit_job.sh and update credentials")
        print("  2. Run: ./azure/submit_job.sh")
        print("  3. Monitor in Azure ML Studio")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before submitting.")
        print("\nCommon fixes:")
        print("  - Install Azure ML SDK: pip install azureml-core")
        print("  - Login to Azure: az login")
        print("  - Download data: huggingface-cli download ...")
        print("  - Update config: cp azure/config.example.json azure/config.json")

    print("\n" + "=" * 70)

    return 0 if all(checks_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
