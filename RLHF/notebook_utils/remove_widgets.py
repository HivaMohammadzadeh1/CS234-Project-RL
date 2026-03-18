import json

def remove_widgets_from_notebook(notebook_path):
    """Remove the 'widgets' section from notebook metadata."""
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Remove widgets from metadata if it exists
    if 'metadata' in notebook and 'widgets' in notebook['metadata']:
        del notebook['metadata']['widgets']
        print(f"Removed 'widgets' section from metadata")
    else:
        print("No 'widgets' section found in metadata")
    
    # Save the notebook back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Notebook saved to {notebook_path}")

# Usage
notebook_path = "/Users/dkoffical/Documents/GitHub/CS234-Project-RL/RLHF/[Final]_RLHF_PPO_Training.ipynb"
remove_widgets_from_notebook(notebook_path)
