#!/bin/bash
# Streamlit launcher script for Raggiro
# This script helps to properly launch the Streamlit UI with correct environment setup

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path to include the project directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Set Streamlit variables to avoid warnings
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Check if the streamlit command is available
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing..."
    pip install streamlit
fi

# Launch the streamlit app with proper arguments
streamlit run "$SCRIPT_DIR/raggiro/gui/streamlit_app.py" "$@" --browser.gatherUsageStats=false