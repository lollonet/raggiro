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

# Always use the direct file path - more compatible with all versions
STREAMLIT_APP_PATH="$SCRIPT_DIR/raggiro/gui/streamlit_app.py"
echo "Running: $STREAMLIT_APP_PATH"

# Launch Streamlit
streamlit run "$STREAMLIT_APP_PATH"