#!/usr/bin/env python3
"""Check if PromptFoo is properly installed."""

import subprocess
import importlib.util
import sys

def is_package_installed(package_name):
    """Check if a package is installed via pip."""
    try:
        # Try to import the package
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            print(f"✅ Package '{package_name}' is installed as Python module")
            return True
        
        # If import fails, try to check if it exists as a command
        try:
            result = subprocess.run(
                [package_name, "--version"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(f"✅ Package '{package_name}' is installed as command line tool: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ Package '{package_name}' is not installed as command line tool")
            return False
    except Exception as e:
        print(f"❌ Error checking package '{package_name}': {e}")
        return False

def main():
    """Main function to check PromptFoo installation."""
    print("\n--- Checking PromptFoo Installation ---\n")
    
    # Check standard promptfoo package
    is_installed = is_package_installed("promptfoo")
    
    # Check where it's looking for the command
    print("\n--- Command Path ---\n")
    try:
        result = subprocess.run(
            ["which", "promptfoo"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"Command path: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("Command 'promptfoo' not found in PATH")
    
    # Check npm installation
    print("\n--- Checking NPM Installation ---\n")
    try:
        result = subprocess.run(
            ["npm", "list", "-g", "promptfoo"], 
            check=False,  # Don't fail if not installed
            capture_output=True, 
            text=True
        )
        print(f"NPM global installation: \n{result.stdout.strip()}")
    except FileNotFoundError:
        print("NPM not installed or not in PATH")
    
    print("\n--- Python Path ---\n")
    print('\n'.join(sys.path))

if __name__ == "__main__":
    main()