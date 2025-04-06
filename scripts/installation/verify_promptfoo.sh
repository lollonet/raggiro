#!/bin/bash
# Verify PromptFoo installation and provide troubleshooting steps

echo "=== PromptFoo Installation Verification ==="
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ NPM is not installed. Please install Node.js and NPM first."
    echo "Visit https://nodejs.org/ for installation instructions."
    exit 1
else
    NPM_VERSION=$(npm --version)
    echo "✅ npm is installed (version $NPM_VERSION)"
fi

# Check npm configuration
NPM_PREFIX=$(npm config get prefix)
echo "npm global prefix: $NPM_PREFIX"

# Check if PATH includes npm bin directory
if [[ ":$PATH:" == *":$NPM_PREFIX/bin:"* ]]; then
    echo "✅ PATH includes npm bin directory"
else
    echo "❌ PATH does not include npm bin directory ($NPM_PREFIX/bin)"
    echo "Add the following to your ~/.bashrc or ~/.profile:"
    echo "export PATH=\"$NPM_PREFIX/bin:\$PATH\""
fi

# Check if promptfoo is installed
echo ""
echo "Checking for PromptFoo installation..."

# 1. Check if it's in PATH
if command -v promptfoo &> /dev/null; then
    PROMPTFOO_VERSION=$(promptfoo --version 2>/dev/null || echo "Unable to determine version")
    echo "✅ promptfoo command is available in PATH"
    echo "   Version: $PROMPTFOO_VERSION"
else
    echo "❌ promptfoo command not found in PATH"
    
    # 2. Check if it's installed but not in PATH
    if [ -f "$NPM_PREFIX/bin/promptfoo" ]; then
        echo "   PromptFoo is installed at $NPM_PREFIX/bin/promptfoo but not in PATH"
        echo "   Add $NPM_PREFIX/bin to your PATH"
    else
        # 3. Check if it's installed in user directory
        if [ -f "$HOME/.npm-global/bin/promptfoo" ]; then
            echo "   PromptFoo is installed at $HOME/.npm-global/bin/promptfoo but not in PATH"
            echo "   Add the following to your ~/.bashrc or ~/.profile:"
            echo "   export PATH=\"$HOME/.npm-global/bin:\$PATH\""
        else
            echo "   PromptFoo is not installed. Run ./scripts/installation/install_promptfoo.sh to install it."
        fi
    fi
fi

# Check npm global list for promptfoo
echo ""
echo "Checking npm global packages..."
npm list -g promptfoo || echo "PromptFoo not found in npm global packages"

echo ""
echo "=== Troubleshooting Tips ==="
echo "1. If you just installed PromptFoo, you may need to restart your terminal"
echo "2. Make sure your PATH includes npm bin directory: $NPM_PREFIX/bin"
echo "3. Try running: source ~/.bashrc or source ~/.profile"
echo "4. If using a virtual environment, ensure it's activated"
echo "5. If all else fails, try installing PromptFoo locally in the project:"
echo "   npm install promptfoo --save-dev"
echo "   Then use: npx promptfoo instead of promptfoo"