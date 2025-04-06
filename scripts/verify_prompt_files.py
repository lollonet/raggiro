#!/usr/bin/env python3
"""
Verify that all prompt files in the test_prompts directory use the new chunking-only format.
"""

import os
import sys
import yaml
from pathlib import Path

def verify_prompt_files(directory):
    """Verify all prompt files in the directory use the new chunking-only format."""
    print(f"Verifying prompt files in {directory}")
    
    problems = []
    files_checked = 0
    
    # Find all YAML files in the directory
    yaml_files = list(Path(directory).glob("*.yaml"))
    yaml_files.extend(list(Path(directory).glob("*.yml")))
    
    for yaml_file in yaml_files:
        files_checked += 1
        print(f"Checking {yaml_file.name}...")
        
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                
                # Check if the file has variants
                if "variants" not in data:
                    problems.append(f"{yaml_file.name}: Missing 'variants' section")
                    continue
                
                # Check each variant
                for i, variant in enumerate(data["variants"]):
                    if "config" not in variant:
                        problems.append(f"{yaml_file.name}: Variant {i} missing 'config' section")
                        continue
                    
                    config = variant["config"]
                    
                    # Check if the config has chunking_strategy
                    if "chunking_strategy" not in config:
                        problems.append(f"{yaml_file.name}: Variant {i} missing 'chunking_strategy' in config")
                    
                    # Check if the config has old Ollama settings
                    if "model" in config or "temperature" in config or "endpoint" in config:
                        problems.append(f"{yaml_file.name}: Variant {i} has old Ollama settings: {', '.join(config.keys())}")
                
        except Exception as e:
            problems.append(f"{yaml_file.name}: Error parsing file: {str(e)}")
    
    # Print results
    print("\nVerification complete!")
    print(f"Checked {files_checked} files")
    
    if problems:
        print(f"\nFound {len(problems)} problems:")
        for problem in problems:
            print(f"- {problem}")
        return False
    else:
        print("\nAll files use the correct format!")
        return True

if __name__ == "__main__":
    # Get the repository root directory
    repo_dir = Path(__file__).resolve().parent.parent
    test_prompts_dir = repo_dir / "test_prompts"
    
    if not test_prompts_dir.exists():
        print(f"Error: Directory not found: {test_prompts_dir}")
        sys.exit(1)
    
    success = verify_prompt_files(test_prompts_dir)
    sys.exit(0 if success else 1)