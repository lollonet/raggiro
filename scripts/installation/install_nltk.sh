#!/bin/bash
# Script to install required NLTK data packages

echo "Installing required NLTK data packages for Raggiro..."

# Create Python script to download NLTK data
cat > /tmp/download_nltk.py << 'EOL'
import nltk
import sys

# Set download directory to project directory if possible
import os
try:
    # Try to find the project root
    current_dir = os.path.abspath(os.getcwd())
    project_root = current_dir
    
    # Look for typical project markers to find root
    markers = ['raggiro', 'pyproject.toml', 'setup.py']
    while project_root != '/' and not any(os.path.exists(os.path.join(project_root, m)) for m in markers):
        project_root = os.path.dirname(project_root)
    
    if project_root != '/':
        nltk_path = os.path.join(project_root, 'nltk_data')
        print(f"Setting NLTK data path to: {nltk_path}")
        nltk.data.path.append(nltk_path)
except Exception as e:
    print(f"Could not set custom NLTK path: {e}")

# List of required NLTK packages
packages = [
    'punkt',         # For sentence tokenization
    'stopwords',     # Common stopwords (optional, for improved summarization)
    'averaged_perceptron_tagger',  # For POS tagging (optional, for improved summarization)
    'punkt_tab'      # Additional punkt data for specific languages
]

# Download each package
for package in packages:
    try:
        print(f"Downloading {package}...")
        nltk.download(package, quiet=False)
        print(f"Successfully downloaded {package}")
    except Exception as e:
        print(f"Error downloading {package}: {e}", file=sys.stderr)
        sys.exit(1)

print("All NLTK packages installed successfully!")
EOL

# Run the Python script
if command -v uv &> /dev/null; then
    uv run python /tmp/download_nltk.py
else
    python /tmp/download_nltk.py
fi

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download NLTK data"
    exit 1
fi

echo "✅ NLTK data installation completed successfully"
echo "You can now use the text summarization features of Raggiro"