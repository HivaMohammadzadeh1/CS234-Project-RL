import json
import glob
from pathlib import Path

def remove_widgets_from_notebook(notebook_path):
    """Remove the 'widgets' section from notebook metadata."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Remove widgets from metadata if it exists
        if 'metadata' in notebook and 'widgets' in notebook['metadata']:
            del notebook['metadata']['widgets']
            print(f"✓ Removed 'widgets' from {Path(notebook_path).name}")
        else:
            print(f"- No 'widgets' section in {Path(notebook_path).name}")
        
        # Save the notebook back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
    except Exception as e:
        print(f"✗ Error processing {Path(notebook_path).name}: {e}")

# Usage - process all .ipynb files in RLHF directory
rlhf_dir = Path(__file__).parent.parent  # Navigate to RLHF directory
notebook_files = list(rlhf_dir.glob("*.ipynb"))

if notebook_files:
    print(f"Found {len(notebook_files)} notebook(s) to process:\n")
    for notebook_path in notebook_files:
        remove_widgets_from_notebook(str(notebook_path))
    print(f"\nCompleted processing {len(notebook_files)} notebook(s)")
else:
    print("No .ipynb files found in RLHF directory")
