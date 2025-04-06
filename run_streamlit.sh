#!/bin/bash
# Streamlit launcher script for Raggiro
# This script helps to properly launch the Streamlit UI with correct environment setup

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path to include the project directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Ensure we're using the correct configuration
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE="light"

# Check if the streamlit command is available
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing..."
    pip install streamlit
fi

# Version-aware streamlit launcher
echo "Launching Raggiro Streamlit interface..."

# Check for module mode vs script mode
if python -c "import importlib.util; print(importlib.util.find_spec('raggiro') is not None)" | grep -q "True"; then
    # Module mode - use the raggiro package
    echo "Using module mode"
    streamlit run -m raggiro.gui.streamlit_app
else
    # Script mode - use direct file path
    echo "Using script mode"
    streamlit run "$SCRIPT_DIR/raggiro/gui/streamlit_app.py"
fi