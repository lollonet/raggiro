#!/usr/bin/env python3
"""
Launcher script for the Raggiro GUI
This script handles PyTorch compatibility issues when running Streamlit
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import tempfile

def create_temp_module():
    """Create a temporary module that monkeypatches Streamlit"""
    temp_dir = tempfile.mkdtemp(prefix="raggiro_streamlit_")
    init_file = Path(temp_dir) / "__init__.py"
    
    with open(init_file, "w") as f:
        f.write("""
# Monkeypatch for Streamlit to handle PyTorch modules properly
import sys
from types import ModuleType
import importlib.util

# Store original __import__ function
original_import = __import__

# Define a custom importer that handles torch specially
def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    # If trying to import torch._classes, provide a fake module
    if name == 'torch._classes' or name.startswith('torch._classes.'):
        if 'torch._classes' not in sys.modules:
            # Create a fake module for torch._classes
            module = ModuleType('torch._classes')
            module.__path__ = ModuleType('torch._classes.__path__')
            # Return a fake __path__ attribute that doesn't cause errors
            module.__path__._path = []
            sys.modules['torch._classes'] = module
            sys.modules['torch._classes.__path__'] = module.__path__
        return sys.modules['torch._classes']
        
    # For other imports, use the original import function
    return original_import(name, globals, locals, fromlist, level)

# Replace the built-in __import__ function with our custom one
sys.__import__ = custom_import
""")
    
    return temp_dir

def main():
    """Main entry point for the GUI launcher"""
    print("Launching Raggiro GUI...")
    
    # Print current directory and file info for debugging
    current_file = Path(__file__).resolve()
    print(f"Current file: {current_file}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Use a more robust method to find the repository root
    # Look for key files/directories that indicate the repo root
    current_dir = current_file.parent
    while True:
        # If we find the raggiro module directory, we're at the right level
        if (current_dir / "raggiro").exists() and (current_dir / "raggiro" / "gui").exists():
            repo_dir = current_dir
            break
        # If we've gone too far up without finding it
        if current_dir.parent == current_dir:  # Reached root of filesystem
            # Fall back to two levels up from script
            repo_dir = Path(__file__).parent.parent.parent.resolve()
            print("WARNING: Could not find repository root by detection, using fallback path")
            break
        # Move up one directory
        current_dir = current_dir.parent
    
    # Create the monkeypatch module
    monkeypatch_dir = create_temp_module()
    
    # Add the monkeypatch directory to the beginning of sys.path
    os.environ["PYTHONPATH"] = f"{monkeypatch_dir}:{repo_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set Streamlit configuration
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILEWATCH_TYPE"] = "none"
    
    # Define the app path
    app_path = repo_dir / "raggiro" / "gui" / "streamlit_app.py"
    
    # Debug info
    print(f"Repository directory: {repo_dir}")
    print(f"Streamlit app path: {app_path}")
    print(f"Path exists: {app_path.exists()}")
    
    # Create the command to run streamlit directly
    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
sys.path.insert(0, "{monkeypatch_dir}")
import streamlit
from pathlib import Path
app_path = Path("{app_path}")
streamlit._is_running_with_streamlit = True
sys.argv = ["streamlit", "run", str(app_path)]
from raggiro.gui.streamlit_app import run_app
run_app()
"""
    ]
    
    # Print information
    print(f"Running Streamlit app: {app_path}")
    print(f"Using Python: {sys.executable}")
    print(f"Using monkeypatch: {monkeypatch_dir}")
    
    # Run the command
    subprocess.run(cmd, env=os.environ)

if __name__ == "__main__":
    main()