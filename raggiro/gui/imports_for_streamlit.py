"""Imports for streamlit app."""

import os
import json
import tempfile
import time
import yaml
import subprocess
import glob
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import streamlit as st
import pandas as pd
from tqdm import tqdm

# Raggiro imports
from raggiro.processor import DocumentProcessor
from raggiro.utils.config import load_config
from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.models.classifier import DocumentClassifier

def import_all():
    """Import all necessary modules for the Streamlit app."""
    pass

def get_processed_files():
    """Get a list of previously processed files."""
    # Default output directory from config or use a standard location
    config = load_config()
    output_dir = config.get("output", {}).get("output_dir", "test_output")
    
    # Make sure the directory exists
    if not os.path.exists(output_dir):
        return []
    
    # Look for processed files (MD and JSON outputs)
    processed_files = []
    for ext in ["*.md", "*.json"]:
        processed_files.extend(glob.glob(os.path.join(output_dir, ext)))
    
    # Also check for original files that might have been processed
    for ext in ["*.pdf", "*.docx", "*.txt"]:
        for file_path in glob.glob(os.path.join("tmp", ext)):
            processed_files.append(file_path)
    
    return sorted(processed_files)