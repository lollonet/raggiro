#!/bin/bash
# Install PromptFoo for Raggiro

echo "Installing PromptFoo for Raggiro..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "NPM is not installed. Please install Node.js and NPM first."
    echo "Visit https://nodejs.org/ for installation instructions."
    exit 1
fi

# Install PromptFoo globally
echo "Installing PromptFoo globally via NPM..."
npm install -g promptfoo

# Check if installation succeeded
if ! command -v promptfoo &> /dev/null; then
    echo "PromptFoo installation failed or not in PATH."
    echo "You may need to add npm global bin directory to your PATH."
    
    # Detect npm prefix
    NPM_PREFIX=$(npm config get prefix)
    echo "Your npm prefix is: $NPM_PREFIX"
    echo "Try adding the following to your .bashrc or .zshrc file:"
    echo "export PATH=\"$NPM_PREFIX/bin:\$PATH\""
    exit 1
fi

echo "PromptFoo installed successfully!"
echo "Version: $(promptfoo --version)"
echo ""
echo "You can now use PromptFoo with Raggiro for RAG testing."