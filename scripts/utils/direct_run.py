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

# Get the repository root directory (3 levels up: scripts/utils -> scripts -> root)
repo_dir = Path(__file__).parent.parent.parent.resolve()

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