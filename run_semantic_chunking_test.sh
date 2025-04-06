#!/bin/bash
# Run the semantic chunking test script with proper environment setup

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Check if input file exists
if [ -z "$1" ]; then
    echo "Error: Please provide an input file path."
    echo "Usage: ./run_semantic_chunking_test.sh <input_file> [additional options]"
    echo "Example: ./run_semantic_chunking_test.sh tmp/Humanizar_it.pdf --queries \"What is humanization?\" \"Define the main concept\""
    exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Run the script directly
echo "Running test_semantic_chunking.py script..."
python3 examples/scripts/test_semantic_chunking.py "$@"

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "The script failed. This may be due to missing dependencies."
    echo "Make sure you have installed all requirements:"
    echo "pip install -e ."
    echo "pip install -r requirements.txt"
    echo ""
    echo "If you're using a virtual environment, make sure it's activated."
fi