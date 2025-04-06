#!/bin/bash
# Multiple approach launcher script for Raggiro
# This script tries several methods to run the GUI reliably

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path to include the project directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Make sure all scripts are executable
chmod +x "$SCRIPT_DIR/launch_gui.py"
chmod +x "$SCRIPT_DIR/../utils/direct_run.py"
chmod +x "$SCRIPT_DIR/../../raggiro/gui/streamlit_app.py"

# Ensure we're using the correct configuration
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_THEME_BASE="light"
export STREAMLIT_SERVER_FILEWATCH_TYPE=none

# Check if the user specified a method
METHOD=${1:-"direct"}

echo "Launching Raggiro GUI using $METHOD method..."

case "$METHOD" in
    "direct")
        # Try the direct Python method (most compatible with PyTorch)
        echo "Using direct Python launcher..."
        exec python "$SCRIPT_DIR/../utils/direct_run.py"
        ;;
    "monkey")
        # Try the monkeypatching method
        echo "Using monkeypatching launcher..."
        exec "$SCRIPT_DIR/launch_gui.py"
        ;;
    "streamlit")
        # Try the regular Streamlit method
        echo "Using standard Streamlit launcher..."
        exec streamlit run "$SCRIPT_DIR/../../raggiro/gui/streamlit_app.py" --server.fileWatcherType none
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available methods: direct, monkey, streamlit"
        exit 1
        ;;
esac