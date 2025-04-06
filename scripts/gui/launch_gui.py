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
    
    # Get the repository root directory
    script_dir = Path(__file__).parent.resolve()
    
    # Create the monkeypatch module
    monkeypatch_dir = create_temp_module()
    
    # Add the monkeypatch directory to the beginning of sys.path
    os.environ["PYTHONPATH"] = f"{monkeypatch_dir}:{script_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set Streamlit configuration
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILEWATCH_TYPE"] = "none"
    
    # Define the app path
    app_path = script_dir / "raggiro" / "gui" / "streamlit_app.py"
    
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