#!/usr/bin/env python3
"""
Ultra-simple direct launcher for Raggiro GUI
This script uses a simplified approach to launch the Streamlit app
with proper configuration to avoid conflicts with PyTorch
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variables first
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_FILEWATCH_TYPE"] = "none"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Print current directory and file location for debugging
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

# Add repo to Python path
sys.path.insert(0, str(repo_dir))

# The streamlit app path
app_path = repo_dir / "raggiro" / "gui" / "streamlit_app.py"

# Debug information
print(f"Repository directory: {repo_dir}")
print(f"Streamlit app path: {app_path}")
print(f"Path exists: {app_path.exists()}")

print("Launching Raggiro GUI directly...")

# Run the app using subprocess to ensure a clean environment
try:
    # Command to run streamlit directly with all necessary flags
    cmd = [
        "streamlit", "run", 
        str(app_path),
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
except Exception as e:
    print(f"Error running app: {e}")
    import traceback
    traceback.print_exc()