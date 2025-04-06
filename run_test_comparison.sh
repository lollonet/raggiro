#!/bin/bash
# Run the RAG comparison test script with proper environment setup

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
    echo "Usage: ./run_test_comparison.sh <input_file> [--strategies <list of strategies>]"
    echo "Example: ./run_test_comparison.sh tmp/Humanizar_it.pdf --strategies size semantic hybrid"
    exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Run the script directly
echo "Running test_rag_comparison.py script..."
python3 examples/scripts/test_rag_comparison.py "$@"

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