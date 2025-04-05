"""Configuration utilities."""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (TOML, JSON, or YAML)
        
    Returns:
        Configuration dictionary
    """
    # Default config
    default_config = {
        "processing": {
            "dry_run": False,
            "recursive": True,
        },
        "logging": {
            "log_level": "info",
            "log_to_file": True,
        },
        "extraction": {
            "ocr_enabled": True,
            "ocr_language": "eng",
        },
        "cleaning": {
            "remove_headers_footers": True,
            "normalize_whitespace": True,
            "remove_special_chars": True,
        },
        "segmentation": {
            "use_spacy": True,
            "spacy_model": "en_core_web_sm",
            "min_section_length": 100,
            "max_chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "export": {
            "formats": ["markdown", "json"],
            "include_metadata": True,
            "pretty_json": True,
        },
    }
    
    if config_path is None:
        # Look for config in standard locations
        config_paths = [
            Path("config.toml"),
            Path("config.json"),
            Path("config.yaml"),
            Path("config.yml"),
            Path.home() / ".raggiro" / "config.toml",
            Path.home() / ".raggiro" / "config.json",
            Path.home() / ".raggiro" / "config.yaml",
            Path.home() / ".raggiro" / "config.yml",
        ]
        
        for path in config_paths:
            if path.exists():
                config_path = str(path)
                break
    
    if config_path is None:
        # No config found, use defaults
        return default_config
    
    # Load config based on file extension
    config = {}
    config_path_obj = Path(config_path)
    
    try:
        if config_path_obj.suffix.lower() == ".toml":
            try:
                import toml
                with open(config_path_obj, "r", encoding="utf-8") as f:
                    config = toml.load(f)
            except ImportError:
                print("Error: TOML support requires 'toml' package. Install with: pip install toml")
                sys.exit(1)
                
        elif config_path_obj.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
                with open(config_path_obj, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            except ImportError:
                print("Error: YAML support requires 'pyyaml' package. Install with: pip install pyyaml")
                sys.exit(1)
                
        elif config_path_obj.suffix.lower() == ".json":
            import json
            with open(config_path_obj, "r", encoding="utf-8") as f:
                config = json.load(f)
                
        else:
            print(f"Error: Unsupported config file format: {config_path_obj.suffix}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)
    
    # Merge with defaults
    merged_config = default_config.copy()
    _update_nested_dict(merged_config, config)
    
    return merged_config

def _update_nested_dict(d: Dict, u: Dict) -> Dict:
    """Update a nested dictionary with another nested dictionary.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v
    return d