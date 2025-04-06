#!/bin/bash
# Install PromptFoo for Raggiro

echo "Installing PromptFoo for Raggiro..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "NPM is not installed. Please install Node.js and NPM first."
    echo "Visit https://nodejs.org/ for installation instructions."
    exit 1
fi

# Setup npm to install global packages in user directory to avoid permission issues
echo "Configuring npm to install packages locally..."
mkdir -p "$HOME/.npm-global"
npm config set prefix "$HOME/.npm-global"

# Add the npm bin directory to PATH temporarily for this session
export PATH="$HOME/.npm-global/bin:$PATH"

# Install PromptFoo locally
echo "Installing PromptFoo globally (in user directory)..."
npm install -g promptfoo

# Check if installation succeeded
if ! command -v promptfoo &> /dev/null; then
    echo "PromptFoo installation failed or not in PATH."
    echo "" 
    echo "Your npm packages were installed in: $HOME/.npm-global"
    echo ""
    echo "To make this permanent, add the following to your ~/.bashrc or ~/.profile:"
    echo "------------------------------------------------------------------"
    echo "export PATH=\"$HOME/.npm-global/bin:\$PATH\""
    echo "------------------------------------------------------------------"
    echo ""
    echo "Then reload your shell with: source ~/.bashrc (or ~/.profile)"
    exit 1
fi

echo "PromptFoo installed successfully!"
echo "Version: $(promptfoo --version 2>/dev/null || echo 'Unable to determine version')"
echo ""
echo "IMPORTANT: Make sure to add the npm bin directory to your PATH permanently:"
echo "------------------------------------------------------------------"
echo "export PATH=\"$HOME/.npm-global/bin:\$PATH\""
echo "------------------------------------------------------------------"
echo ""
echo "Add this line to your ~/.bashrc or ~/.profile and reload your shell."
echo "You can now use PromptFoo with Raggiro for RAG testing."