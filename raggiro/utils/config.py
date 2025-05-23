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
        "spelling": {
            "enabled": True,
            "language": "auto",
            "backend": "symspellpy",
            "max_edit_distance": 2,
            "always_correct": True,
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
                    try:
                        # First attempt: standard TOML parsing
                        config = toml.load(f)
                    except Exception as e:
                        # If standard TOML loading fails (likely due to variable interpolation)
                        # Print a more helpful message without the overwhelming error details
                        print(f"Note: Using custom TOML parser for interpolated variables. This is normal.")
                        
                        # Second attempt: Try loading section by section to avoid interpolation issues
                        try:
                            # Rewind file pointer to the beginning
                            f.seek(0)
                            
                            # Parse each section independently to avoid interpolation errors
                            import re
                            config = {}
                            current_section = ""
                            section_content = {}
                            
                            # Read line by line
                            for line in f:
                                line = line.strip()
                                
                                # Skip empty lines and comments
                                if not line or line.startswith('#'):
                                    continue
                                    
                                # Check if it's a section header
                                section_match = re.match(r'^\[(.*)\]$', line)
                                if section_match:
                                    # Save previous section if any
                                    if current_section and section_content:
                                        if current_section not in config:
                                            config[current_section] = {}
                                        config[current_section].update(section_content)
                                    
                                    # Start new section
                                    current_section = section_match.group(1)
                                    section_content = {}
                                    continue
                                
                                # Parse key-value pairs
                                if '=' in line and current_section:
                                    key, value = [part.strip() for part in line.split('=', 1)]
                                    
                                    # Skip interpolation variables
                                    if '${' in value:
                                        continue
                                        
                                    # Parse various value types
                                    try:
                                        # Try to evaluate the value (for booleans, numbers, lists, etc.)
                                        import ast
                                        section_content[key] = ast.literal_eval(value)
                                    except (SyntaxError, ValueError):
                                        # If that fails, treat it as a string (remove quotes)
                                        if value.startswith('"') and value.endswith('"'):
                                            section_content[key] = value[1:-1]
                                        elif value.startswith("'") and value.endswith("'"):
                                            section_content[key] = value[1:-1]
                                        else:
                                            section_content[key] = value
                            
                            # Add the last section
                            if current_section and section_content:
                                if current_section not in config:
                                    config[current_section] = {}
                                config[current_section].update(section_content)
                                
                        except Exception as inner_e:
                            print(f"Warning: Failed to parse TOML manually: {inner_e}")
                            
                        # Final step: ensure all essential sections exist and are properly populated
                        # LLM section
                        if "llm" not in config:
                            config["llm"] = {}
                            
                        llm_defaults = {
                            "provider": "ollama",
                            "ollama_base_url": "http://ollama:11434",
                            "ollama_timeout": 30,
                            "llamacpp_path": "",
                            "api_key": "",
                            "api_url": "",
                            "openai_model": "gpt-3.5-turbo"
                        }
                        
                        for key, value in llm_defaults.items():
                            if key not in config["llm"]:
                                config["llm"][key] = value
                            
                        # Rewriting section
                        if "rewriting" not in config:
                            config["rewriting"] = {}
                            
                        rewriting_defaults = {
                            "enabled": True,
                            "llm_type": config["llm"].get("provider", "ollama"),
                            "ollama_model": "llama3",
                            "llamacpp_model": "llama3",
                            "openai_model": config["llm"].get("openai_model", "gpt-3.5-turbo"),
                            "temperature": 0.1,
                            "max_tokens": 200,
                            "ollama_base_url": config["llm"].get("ollama_base_url", "http://ollama:11434"),
                            "llamacpp_path": config["llm"].get("llamacpp_path", ""),
                            "api_key": config["llm"].get("api_key", ""),
                            "api_url": config["llm"].get("api_url", "")
                        }
                        
                        for key, value in rewriting_defaults.items():
                            if key not in config["rewriting"]:
                                config["rewriting"][key] = value
                                
                        # Generation section
                        if "generation" not in config:
                            config["generation"] = {}
                            
                        generation_defaults = {
                            "llm_type": config["llm"].get("provider", "ollama"),
                            "ollama_model": "mistral",
                            "llamacpp_model": "mistral",
                            "openai_model": config["llm"].get("openai_model", "gpt-3.5-turbo"),
                            "temperature": 0.7,
                            "max_tokens": 1000,
                            "ollama_base_url": config["llm"].get("ollama_base_url", "http://ollama:11434"),
                            "llamacpp_path": config["llm"].get("llamacpp_path", ""),
                            "api_key": config["llm"].get("api_key", ""),
                            "api_url": config["llm"].get("api_url", "")
                        }
                        
                        for key, value in generation_defaults.items():
                            if key not in config["generation"]:
                                config["generation"][key] = value
                                
                        # Segmentation section
                        if "segmentation" not in config:
                            config["segmentation"] = {}
                            
                        segmentation_defaults = {
                            "use_spacy": True,
                            "spacy_model": "en_core_web_sm",
                            "min_section_length": 100,
                            "max_chunk_size": 500,
                            "chunk_overlap": 100,
                            "semantic_chunking": True,
                            "chunking_strategy": "hybrid",
                            "semantic_similarity_threshold": 0.65
                        }
                        
                        for key, value in segmentation_defaults.items():
                            if key not in config["segmentation"]:
                                config["segmentation"][key] = value
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