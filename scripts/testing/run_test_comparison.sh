#!/bin/bash
# Run the RAG comparison test script with proper environment setup

# Note: We're using uv for environment management instead of traditional venv
# No manual activation is needed when using uv run
if ! command -v uv &> /dev/null; then
    echo "WARNING: uv is not installed. It's recommended to use uv for this project."
    echo "You can install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# Parse arguments
POSITIONAL_ARGS=()
INPUT_FILE=""
HAS_INPUT_ARG=false

# Process arguments to find or add --input
while [[ $# -gt 0 ]]; do
  case $1 in
    --input|-i)
      INPUT_FILE="$2"
      HAS_INPUT_ARG=true
      POSITIONAL_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# If no input argument was found and first argument exists and is a file
if [[ "$HAS_INPUT_ARG" = false && -n "${POSITIONAL_ARGS[0]}" && -f "${POSITIONAL_ARGS[0]}" ]]; then
    INPUT_FILE="${POSITIONAL_ARGS[0]}"
    # Remove first argument and add it as --input
    ARGS_WITHOUT_FIRST=("${POSITIONAL_ARGS[@]:1}")
    POSITIONAL_ARGS=("--input" "$INPUT_FILE" "${ARGS_WITHOUT_FIRST[@]}")
fi

# Check if input file exists
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Please provide an input file path."
    echo "Usage: ./scripts/testing/run_test_comparison.sh <input_file> [--strategies <list of strategies>]"
    echo "   or: ./scripts/testing/run_test_comparison.sh --input <input_file> [--strategies <list of strategies>]"
    echo "Example: ./scripts/testing/run_test_comparison.sh tmp/Humanizar_it.pdf --strategies size semantic hybrid"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Run the script directly
echo "Running test_rag_comparison.py script..."
echo "Command: uv run python examples/scripts/test_rag_comparison.py ${POSITIONAL_ARGS[*]}"
if command -v uv &> /dev/null; then
    uv run python examples/scripts/test_rag_comparison.py "${POSITIONAL_ARGS[@]}"
else
    # Fallback to regular python if uv is not available
    python3 examples/scripts/test_rag_comparison.py "${POSITIONAL_ARGS[@]}"
fi

# Check if the script executed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "The script failed. This may be due to missing dependencies."
    echo "Make sure you have installed all dependencies:"
    echo "uv pip install -e ."
    echo "or for development: uv pip install -e \".[dev]\""
    echo ""
    echo "If you're using uv, make sure commands are run with 'uv run'."
fi