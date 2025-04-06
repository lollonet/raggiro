#!/usr/bin/env python3
"""
Ultra-simple direct launcher for Raggiro GUI
This script bypasses Streamlit completely and just runs the app directly
"""

import os
import sys
from pathlib import Path

# Get the repository root directory
repo_dir = Path(__file__).parent.resolve()

# Add repo to Python path
sys.path.insert(0, str(repo_dir))

# Load Streamlit first to avoid import errors
import streamlit as st

# Set required Streamlit variables
st._is_running_with_streamlit = True
st._config.set_option("server.fileWatcherType", "none")

# Force disable Streamlit's file watcher
import streamlit.watcher.path_watcher
streamlit.watcher.path_watcher._is_watchable_module = lambda _: False

# Import and run the app
try:
    from raggiro.gui.streamlit_app import run_app
    print("Launching Raggiro GUI directly...")
    run_app()
except Exception as e:
    print(f"Error running app: {e}")
    import traceback
    traceback.print_exc()