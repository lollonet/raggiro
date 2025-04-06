#!/usr/bin/env python3
"""
Ultra-simple direct launcher for Raggiro GUI
This script bypasses Streamlit's main script runner and launches it directly
"""

import os
import sys
from pathlib import Path

# Get the repository root directory
repo_dir = Path(__file__).parent.resolve()

# Add repo to Python path
sys.path.insert(0, str(repo_dir))

# Create a fake ScriptRunContext to avoid the missing context error
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
import threading
from streamlit.runtime.scriptrunner.script_run_context import ScriptRunContext, get_script_run_ctx

# Create and set a fake context if needed 
if not get_script_run_ctx():
    # Create minimal fake context with required attributes
    ctx = ScriptRunContext(
        session_id="fake_session_id",
        _enqueue=lambda msg: None,  # Dummy enqueue function
        query_string="",
        session_state={},
        uploaded_file_mgr=None
    )
    # Set the context for the current thread
    add_script_run_ctx(threading.current_thread(), ctx)

# Set required Streamlit variables
st._is_running_with_streamlit = True
st._config.set_option("server.fileWatcherType", "none")
os.environ["STREAMLIT_SERVER_FILEWATCH_TYPE"] = "none"

# Force disable Streamlit's file watcher
import streamlit.watcher.path_watcher
streamlit.watcher.path_watcher._is_watchable_module = lambda _: False

# Import and run Streamlit CLI
try:
    # Set up minimum argv for streamlit.cli.main()
    sys.argv = ["streamlit", "run", str(repo_dir / "raggiro" / "gui" / "streamlit_app.py")]
    
    # Run Streamlit CLI
    import streamlit.web.cli as cli
    cli._main_run_cloned = True  # Tell Streamlit we're already inside an executed script
    
    print("Launching Raggiro GUI directly...")
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run("/home/ubuntu/raggiro/raggiro/gui/streamlit_app.py", "", [], flag_options={})
    
except Exception as e:
    print(f"Error running app: {e}")
    import traceback
    traceback.print_exc()