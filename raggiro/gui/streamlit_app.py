#!/usr/bin/env python3
"""Streamlit-based GUI for Raggiro document processing."""

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

from raggiro.processor import DocumentProcessor
from raggiro.utils.config import load_config

def run_app():
    """Run the Streamlit app."""
    import streamlit as st

    # Set page configuration
    try:
        st.set_page_config(
            page_title="Raggiro - Document Processing for RAG",
            page_icon="📄",
            layout="wide",
        )
    except Exception as e:
        # This might be called twice in some Streamlit versions
        # Just ignore the error if it fails
        pass
    
    st.title("Raggiro - Document Processing for RAG")
    st.write("Process documents for Retrieval-Augmented Generation (RAG) systems")
    
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Process Documents", "OCR & Correction", "Test RAG", "View Results", "Configuration"])
    
    with tab1:
        process_documents_ui()
    
    with tab2:
        ocr_correction_ui()
    
    with tab3:
        test_rag_ui()
    
    with tab4:
        view_results_ui()
    
    with tab5:
        configuration_ui()

def process_documents_ui():
    """UI for processing documents."""
    st.header("Process Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        
        # File or directory input
        input_type = st.radio(
            "Input Type",
            options=["Upload Files", "Local Path"],
            index=0,
        )
        
        if input_type == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload documents to process",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "html", "htm", "rtf", "xlsx", "xls", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
            )
        else:
            input_path = st.text_input(
                "Local Path",
                placeholder="/path/to/documents",
            )
            recursive = st.checkbox("Process subdirectories", value=True)
    
    with col2:
        st.subheader("Output Options")
        
        # Output format selection
        output_formats = st.multiselect(
            "Output Formats",
            options=["markdown", "json", "txt"],
            default=["markdown", "json"],
        )
        
        # Output path
        use_temp_dir = st.checkbox("Use temporary directory", value=True)
        if not use_temp_dir:
            output_path = st.text_input(
                "Output Path",
                placeholder="/path/to/output",
            )
        else:
            output_path = None
        
        # Processing options
        with st.expander("Advanced Options"):
            # Chunking options
            st.subheader("Chunking Options")
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                options=["hybrid", "semantic", "size"],
                index=0,
            )
            
            semantic_chunking = st.checkbox("Enable semantic chunking", value=True)
            
            # OCR options
            st.subheader("OCR Options")
            ocr_enabled = st.checkbox("Enable OCR for scanned documents", value=True)
            ocr_language = st.selectbox(
                "OCR Language",
                options=["eng", "ita", "fra", "deu", "spa", "por", "nld", "rus", "jpn", "kor", "chi_sim"],
                index=0,
            )
            
            # Log level
            st.subheader("Logging")
            log_level = st.selectbox(
                "Log Level",
                options=["debug", "info", "warning", "error"],
                index=1,
            )
    
    # Process button
    process_button = st.button("Process Documents", type="primary")
    
    if process_button:
        if input_type == "Upload Files" and not uploaded_files:
            st.error("Please upload at least one file")
            return
        
        if input_type == "Local Path" and not input_path:
            st.error("Please enter a local path")
            return
        
        if not use_temp_dir and not output_path:
            st.error("Please enter an output path")
            return
        
        # Process documents
        with st.spinner("Processing documents..."):
            process_documents(
                input_type=input_type,
                uploaded_files=uploaded_files if input_type == "Upload Files" else None,
                input_path=input_path if input_type == "Local Path" else None,
                recursive=recursive if input_type == "Local Path" else False,
                output_formats=output_formats,
                output_path=output_path,
                use_temp_dir=use_temp_dir,
                chunking_strategy=chunking_strategy,
                semantic_chunking=semantic_chunking,
                ocr_enabled=ocr_enabled,
                ocr_language=ocr_language,
                log_level=log_level,
            )

def ocr_correction_ui():
    """UI for OCR and spelling/semantic correction."""
    st.header("OCR & Text Correction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        
        # File or directory input
        input_type = st.radio(
            "Input Type",
            options=["Upload Files", "Local Path"],
            index=0,
            key="ocr_input_type"
        )
        
        if input_type == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload documents for OCR/correction",
                accept_multiple_files=True,
                type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
                key="ocr_upload"
            )
        else:
            input_path = st.text_input(
                "Local Path",
                placeholder="/path/to/documents",
                key="ocr_input_path"
            )
            recursive = st.checkbox("Process subdirectories", value=True, key="ocr_recursive")
    
    with col2:
        st.subheader("Output Options")
        
        # Output format selection
        output_formats = st.multiselect(
            "Output Formats",
            options=["markdown", "json", "txt"],
            default=["markdown", "json"],
            key="ocr_output_formats"
        )
        
        # Output path
        use_temp_dir = st.checkbox("Use temporary directory", value=True, key="ocr_use_temp")
        if not use_temp_dir:
            output_path = st.text_input(
                "Output Path",
                placeholder="/path/to/output",
                key="ocr_output_path"
            )
        else:
            output_path = None
    
    # OCR settings section
    st.subheader("OCR Settings")
    ocr_enabled = st.checkbox("Enable OCR", value=True, key="ocr_enabled")
    
    # OCR language options
    ocr_language_options = [
        ("auto", "Auto-detect (Recommended)"),
        ("eng", "English"),
        ("ita", "Italian"),
        ("fra", "French"),
        ("deu", "German"),
        ("spa", "Spanish"),
        ("por", "Portuguese"),
        ("eng+ita", "English + Italian"),
        ("eng+fra+deu+spa", "Multiple European Languages")
    ]
    
    ocr_language = st.selectbox(
        "OCR Language",
        options=[code for code, _ in ocr_language_options],
        format_func=lambda x: next((name for code, name in ocr_language_options if code == x), x),
        index=0,
        key="ocr_language"
    )
    
    # OCR quality settings
    col1, col2 = st.columns(2)
    with col1:
        ocr_dpi = st.slider(
            "OCR Resolution (DPI)",
            min_value=150,
            max_value=600,
            value=300,
            step=50,
            help="Higher values provide better quality but require more memory and processing time",
            key="ocr_dpi"
        )
    
    with col2:
        ocr_batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Number of pages to process in each batch",
            key="ocr_batch"
        )
    
    # Spelling correction section
    st.subheader("Spelling Correction")
    
    spelling_enabled = st.checkbox("Enable spelling correction", value=True, key="spelling_enabled")
    
    # Create columns for spelling settings
    col1, col2 = st.columns(2)
    
    with col1:
        spelling_backends = {
            "symspellpy": "SymSpellPy (Fast, recommended)",
            "textblob": "TextBlob (Good multilingual support)",
            "wordfreq": "Wordfreq (Lightweight fallback)"
        }
        
        spelling_backend = st.selectbox(
            "Correction Engine",
            options=list(spelling_backends.keys()),
            format_func=lambda x: spelling_backends.get(x, x),
            key="spelling_backend"
        )
        
        spelling_languages = {
            "auto": "Auto-detect (Recommended)",
            "en": "English",
            "it": "Italian",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "pt": "Portuguese"
        }
        
        spelling_language = st.selectbox(
            "Correction Language",
            options=list(spelling_languages.keys()),
            format_func=lambda x: spelling_languages.get(x, x),
            key="spelling_language"
        )
    
    with col2:
        max_edit_distance = st.slider(
            "Max Edit Distance",
            min_value=1,
            max_value=3,
            value=2,
            help="Maximum number of character edits for correction suggestions",
            key="max_edit_distance"
        )
        
        always_correct = st.checkbox(
            "Apply to all documents (not just OCR)",
            value=True,
            help="Apply spelling correction to all documents, not just those processed with OCR",
            key="always_correct"
        )
    
    # Semantic chunking section
    st.subheader("Semantic Segmentation")
    
    semantic_chunking = st.checkbox("Enable semantic chunking", value=True, key="semantic_chunking")
    
    # Create columns for chunking settings
    col1, col2 = st.columns(2)
    
    with col1:
        chunking_strategies = {
            "hybrid": "Hybrid (Combines size and semantic approaches)",
            "semantic": "Semantic (Based on content similarity)",
            "size": "Size-based (Fixed size chunks)"
        }
        
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            options=list(chunking_strategies.keys()),
            format_func=lambda x: chunking_strategies.get(x, x),
            key="chunking_strategy"
        )
    
    with col2:
        semantic_similarity = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.4,
            max_value=0.9,
            value=0.55,
            step=0.05,
            help="Threshold for semantic similarity when creating chunks (lower = more chunks)",
            key="semantic_similarity"
        )
    
    # Add advanced options expander
    with st.expander("Advanced Chunking Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_chunk_size = st.number_input(
                "Max Chunk Size",
                min_value=500,
                max_value=3000,
                value=1500,
                step=100,
                help="Maximum character count per chunk",
                key="max_chunk_size"
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Number of characters to overlap between chunks",
                key="chunk_overlap"
            )
    
    # Process button
    process_button = st.button("Process Documents", type="primary", key="ocr_process_btn")
    
    if process_button:
        if input_type == "Upload Files" and not uploaded_files:
            st.error("Please upload at least one file")
            return
        
        if input_type == "Local Path" and not input_path:
            st.error("Please enter a local path")
            return
        
        if not use_temp_dir and not output_path:
            st.error("Please enter an output path")
            return
        
        # Create configuration for the processor
        config = {
            "processing": {
                "dry_run": False,
                "recursive": recursive if input_type == "Local Path" else False,
            },
            "logging": {
                "log_level": "info",
            },
            "extraction": {
                "ocr_enabled": ocr_enabled,
                "ocr_language": ocr_language,
                "ocr_dpi": ocr_dpi,
                "ocr_batch_size": ocr_batch_size,
            },
            "cleaning": {
                "remove_headers_footers": True,
                "normalize_whitespace": True,
                "remove_special_chars": True,
            },
            "spelling": {
                "enabled": spelling_enabled,
                "language": spelling_language,
                "backend": spelling_backend,
                "max_edit_distance": max_edit_distance,
                "always_correct": always_correct,
            },
            "segmentation": {
                "semantic_chunking": semantic_chunking,
                "chunking_strategy": chunking_strategy,
                "semantic_similarity_threshold": semantic_similarity,
                "max_chunk_size": max_chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            "export": {
                "formats": output_formats,
            },
        }
        
        # Process documents with the specialized OCR & correction workflow
        with st.spinner("Processing documents with OCR and text correction..."):
            process_documents(
                input_type=input_type,
                uploaded_files=uploaded_files if input_type == "Upload Files" else None,
                input_path=input_path if input_type == "Local Path" else None,
                recursive=recursive if input_type == "Local Path" else False,
                output_formats=output_formats,
                output_path=output_path,
                use_temp_dir=use_temp_dir,
                config=config,
                is_ocr_workflow=True,
            )

def test_rag_ui():
    """UI for testing RAG capabilities."""
    st.header("Test RAG System")
    
    # Check if promptfoo is installed and show a warning if not
    if not is_package_installed("promptfoo"):
        with st.warning("PromptFoo is not installed. Some test features will be unavailable."):
            st.markdown("### Install PromptFoo")
            
            # Find the installation script
            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            install_script_path = os.path.join(repo_dir, "scripts", "installation", "install_promptfoo.sh")
            verify_script_path = os.path.join(repo_dir, "scripts", "installation", "verify_promptfoo.sh")
            
            if os.path.exists(install_script_path):
                st.info("PromptFoo installation script found. Use one of these methods to install:")
                
                tab1, tab2 = st.tabs(["Method 1: User Installation", "Method 2: Global Installation"])
                
                with tab1:
                    st.markdown("#### User Installation (Recommended)")
                    st.markdown("This installs PromptFoo in your user directory, avoiding permission issues:")
                    st.code(f"chmod +x {install_script_path} && {install_script_path}", language="bash")
                    st.markdown("**After installation**, add this to your ~/.bashrc or ~/.profile:")
                    st.code("export PATH=\"$HOME/.npm-global/bin:$PATH\"", language="bash")
                    st.markdown("Then reload your terminal or run:")
                    st.code("source ~/.bashrc", language="bash")
                
                with tab2:
                    st.markdown("#### Global Installation (Alternative)")
                    st.markdown("This installs PromptFoo globally (may require sudo):")
                    st.code("npm install -g promptfoo", language="bash")
                
                if os.path.exists(verify_script_path):
                    st.markdown("#### Verify Installation")
                    st.markdown("After installing, you can verify your installation with:")
                    st.code(f"chmod +x {verify_script_path} && {verify_script_path}", language="bash")
            else:
                st.info("PromptFoo is a Node.js application used for advanced RAG testing. Install it with:")
                st.code("npm install -g promptfoo", language="bash")
            
            st.markdown("---")
    
    # Input directory (processed documents)
    st.subheader("Input")
    
    input_type = st.radio(
        "Select input type",
        options=["Select Processed Directory", "Use Last Processed Output"],
        index=0,
    )
    
    if input_type == "Select Processed Directory":
        input_path = st.text_input(
            "Processed Documents Directory",
            placeholder="/path/to/processed/documents",
        )
    else:
        # Try to get the last processed output directory from session state
        if "last_output_dir" in st.session_state:
            input_path = st.session_state["last_output_dir"]
            st.info(f"Using last processed output directory: {input_path}")
        else:
            st.warning("No processed output found in session. Please specify a directory.")
            input_path = st.text_input(
                "Processed Documents Directory",
                placeholder="/path/to/processed/documents",
            )
    
    # Test configuration
    st.subheader("Test Configuration")
    
    test_type = st.radio(
        "Test Type",
        options=["Use Predefined Test Prompts", "Upload Test Prompts", "Create Custom Prompts"],
        index=0,
    )
    
    if test_type == "Use Predefined Test Prompts":
        # Search for test prompt files in the test_prompts directory
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_prompt_location = os.path.join(repo_dir, "test_prompts")
        
        prompt_files = []
        found_dir = None
        
        # Search in the test_prompts directory
        if os.path.exists(test_prompt_location):
            # Look for YAML files in this directory
            yaml_files = [
                f for f in glob.glob(os.path.join(test_prompt_location, "*.yaml")) 
                if os.path.isfile(f)
            ]
            yaml_files.extend([
                f for f in glob.glob(os.path.join(test_prompt_location, "*.yml")) 
                if os.path.isfile(f)
            ])
            
            if yaml_files:
                prompt_files = [os.path.basename(f) for f in yaml_files]
                found_dir = test_prompt_location
        
        if prompt_files:
            st.success(f"Found {len(prompt_files)} test prompt files in {found_dir}")
            selected_prompt_file = st.selectbox(
                "Select Test Prompt File",
                options=prompt_files,
            )
            prompt_file_path = os.path.join(found_dir, selected_prompt_file)
            
            # Preview selected file
            try:
                with open(prompt_file_path, 'r') as f:
                    file_content = f.read()
                    with st.expander("Preview selected prompt file"):
                        st.code(file_content, language="yaml")
            except Exception as e:
                st.warning(f"Could not preview file: {e}")
        else:
            st.warning("No test prompt files found. Please upload a test prompt file or create custom prompts.")
            prompt_file_path = None
    
    elif test_type == "Upload Test Prompts":
        uploaded_prompt_file = st.file_uploader(
            "Upload YAML Test Prompts File",
            type=["yaml", "yml"],
        )
        
        if uploaded_prompt_file:
            # Save to temporary file
            temp_prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            temp_prompt_file.write(uploaded_prompt_file.getbuffer())
            prompt_file_path = temp_prompt_file.name
            
            # Display content of the uploaded file
            st.expander("View Uploaded Test Prompts").code(uploaded_prompt_file.getvalue().decode(), language="yaml")
        else:
            prompt_file_path = None
    
    else:  # Create Custom Prompts
        st.subheader("Create Custom Prompts")
        
        # Enter prompts
        custom_prompts = st.text_area(
            "Enter Prompts (one per line)",
            height=150,
            placeholder="What is the main topic of this document?\nSummarize the key points of this document.\nWhat are the recommendations in this document?"
        )
        
        if custom_prompts:
            # Split by lines
            prompts = [p.strip() for p in custom_prompts.split('\n') if p.strip()]
            
            # Create YAML structure
            test_config = {
                "prompts": prompts,
                "tests": [
                    {
                        "description": "Quality check",
                        "assert": [
                            {
                                "type": "contains-json",
                                "value": {"min_length": 100}
                            }
                        ]
                    }
                ],
                "variants": [
                    {
                        "name": "semantic_chunking",
                        "description": "Test with semantic chunking",
                        "config": {
                            "chunking_strategy": "semantic"
                        }
                    },
                    {
                        "name": "size_chunking",
                        "description": "Test with size-based chunking",
                        "config": {
                            "chunking_strategy": "size"
                        }
                    }
                ]
            }
            
            # Convert to YAML
            yaml_content = yaml.dump(test_config, default_flow_style=False, sort_keys=False)
            
            # Save to temporary file
            temp_prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
            temp_prompt_file.write(yaml_content.encode('utf-8'))
            prompt_file_path = temp_prompt_file.name
            
            # Display generated YAML
            st.expander("View Generated Test Prompts").code(yaml_content, language="yaml")
        else:
            prompt_file_path = None
    
    # Output directory
    st.subheader("Output")
    
    output_type = st.radio(
        "Select output type",
        options=["Use Custom Output Directory", "Use Input Directory"],
        index=1,
    )
    
    if output_type == "Use Custom Output Directory":
        output_path = st.text_input(
            "Test Results Directory",
            placeholder="/path/to/test/results",
        )
    else:
        output_path = input_path  # Use the same directory as input
    
    # Run test button
    run_test_button = st.button("Run RAG Tests", type="primary")
    
    if run_test_button:
        if not input_path:
            st.error("Please enter a processed documents directory")
            return
        
        if not prompt_file_path:
            st.error("No test prompts file specified")
            return
        
        if not output_path:
            st.error("Please enter an output directory")
            return
        
        # Run tests
        with st.spinner("Running RAG tests..."):
            run_rag_tests(
                input_path=input_path,
                prompt_file_path=prompt_file_path,
                output_path=output_path,
            )

def view_results_ui():
    """UI for viewing test results."""
    st.header("View Test Results")
    
    # Input directory for results
    st.subheader("Select Results Source")
    
    # Options for loading results
    results_source = st.radio(
        "Results Source",
        options=["Select Results Directory", "Use Last Test Results", "Browse Test History"],
        index=0,
    )
    
    results_path = None
    
    if results_source == "Select Results Directory":
        # Manual path input
        results_path = st.text_input(
            "Results Directory",
            placeholder="/path/to/test/results",
        )
    elif results_source == "Use Last Test Results":
        # Try to get the last test results directory from session state
        if "last_test_dir" in st.session_state:
            results_path = st.session_state["last_test_dir"]
            st.success(f"Using last test results from: {results_path}")
        else:
            st.warning("No recent test results found in this session. Please select a directory.")
            results_path = st.text_input(
                "Results Directory",
                placeholder="/path/to/test/results",
            )
    else:  # Browse Test History
        # Look for test result directories in common locations
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        common_result_locations = [
            os.path.join(repo_dir, "test_output"),
            os.path.join(repo_dir, "test_results"),
            os.path.join(repo_dir, "output"),
            os.path.join(repo_dir, "results"),
            os.path.join(repo_dir, "tmp"),
        ]
        
        test_dirs = []
        
        # Look for result directories
        for location in common_result_locations:
            if os.path.exists(location) and os.path.isdir(location):
                # Add this directory
                test_dirs.append(location)
                
                # Look for subdirectories that might contain test results
                subdirs = [os.path.join(location, d) for d in os.listdir(location) 
                          if os.path.isdir(os.path.join(location, d)) and 
                          (d.startswith("rag_test_") or d.startswith("test_"))]
                
                test_dirs.extend(subdirs)
        
        if test_dirs:
            # Sort by modification time (newest first)
            test_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Add timestamps and file counts for better selection
            labeled_dirs = []
            for directory in test_dirs:
                json_count = len(glob.glob(os.path.join(directory, "*.json")))
                md_count = len(glob.glob(os.path.join(directory, "*.md")))
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(directory)))
                
                if json_count > 0 or md_count > 0:
                    label = f"{os.path.basename(directory)} - {mod_time} ({json_count} JSON, {md_count} MD files)"
                    labeled_dirs.append((directory, label))
            
            if labeled_dirs:
                # Create a dictionary for the selectbox options
                dir_options = {label: path for path, label in labeled_dirs}
                
                selected_label = st.selectbox(
                    "Select Test Results",
                    options=list(dir_options.keys()),
                )
                
                results_path = dir_options[selected_label]
                st.success(f"Selected: {results_path}")
            else:
                st.warning("No test result directories with JSON or MD files found")
                results_path = st.text_input(
                    "Results Directory",
                    placeholder="/path/to/test/results",
                )
        else:
            st.warning("No test result directories found. Please enter a path manually.")
            results_path = st.text_input(
                "Results Directory",
                placeholder="/path/to/test/results",
            )
    
    # Load results button
    col1, col2 = st.columns([3, 1])
    with col1:
        load_results_button = st.button("Load Results", type="primary")
    with col2:
        refresh_button = st.button("Refresh", help="Refresh the page to update the list of test results")
    
    if load_results_button:
        if not results_path:
            st.error("Please enter a results directory")
            return
        
        # Check if the directory exists
        if not os.path.exists(results_path):
            st.error(f"Directory not found: {results_path}")
            return
        
        # Store in session state
        st.session_state["last_test_dir"] = results_path
        
        # Create a progress indicator
        progress = st.progress(0, "Processing results")
        
        # Look for JSON and Markdown result files
        json_files = glob.glob(os.path.join(results_path, "*.json"))
        md_files = glob.glob(os.path.join(results_path, "*.md"))
        
        if not json_files and not md_files:
            st.warning(f"No result files found in {results_path}")
            return
        
        # Update progress
        progress.progress(10)
        
        # Organize files by type
        result_files = {
            "PromptFoo Results": [],
            "RAG Query Results": [],
            "RAG Index Info": [],
            "Markdown Reports": [],
            "Other Results": []
        }
        
        # Process JSON files
        total_files = len(json_files) + len(md_files)
        processed = 0
        
        for json_file in json_files:
            try:
                file_name = os.path.basename(json_file)
                file_size = os.path.getsize(json_file) / 1024  # KB
                
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        
                        # Categorize the file based on content
                        if "prompts" in data or "summary" in data:
                            result_files["PromptFoo Results"].append({
                                "path": json_file,
                                "name": file_name,
                                "size": file_size,
                                "data": data
                            })
                        elif isinstance(data, list) and all("query" in item for item in data):
                            result_files["RAG Query Results"].append({
                                "path": json_file,
                                "name": file_name,
                                "size": file_size,
                                "data": data
                            })
                        elif "index_info" in data or "vector_count" in data:
                            result_files["RAG Index Info"].append({
                                "path": json_file,
                                "name": file_name,
                                "size": file_size,
                                "data": data
                            })
                        else:
                            result_files["Other Results"].append({
                                "path": json_file,
                                "name": file_name,
                                "size": file_size,
                                "type": "json"
                            })
                    except json.JSONDecodeError:
                        result_files["Other Results"].append({
                            "path": json_file,
                            "name": file_name,
                            "size": file_size,
                            "type": "json",
                            "error": "Invalid JSON format"
                        })
            except Exception as e:
                st.error(f"Error processing {os.path.basename(json_file)}: {str(e)}")
                
            processed += 1
            progress.progress(10 + int(40 * processed / total_files))
        
        # Process markdown files
        for md_file in md_files:
            try:
                file_name = os.path.basename(md_file)
                file_size = os.path.getsize(md_file) / 1024  # KB
                
                # Just add to markdown reports
                result_files["Markdown Reports"].append({
                    "path": md_file,
                    "name": file_name,
                    "size": file_size,
                    "type": "markdown"
                })
            except Exception as e:
                st.error(f"Error processing {os.path.basename(md_file)}: {str(e)}")
                
            processed += 1
            progress.progress(10 + int(40 * processed / total_files))
        
        # Update progress
        progress.progress(50)
            
        # Display results
        st.subheader("Test Results")
        
        # Calculate total counts
        total_count = sum(len(files) for files in result_files.values())
        
        if total_count > 0:
            # Create tabs for different types of results
            tab_names = []
            for category, files in result_files.items():
                if files:
                    tab_names.append(f"{category} ({len(files)})")
                else:
                    tab_names.append(category)
            
            result_tabs = st.tabs(tab_names)
            
            # Update progress
            progress.progress(60)
            
            # Fill tabs with content
            tab_idx = 0
            
            # PromptFoo Results
            with result_tabs[tab_idx]:
                if result_files["PromptFoo Results"]:
                    for i, file_info in enumerate(result_files["PromptFoo Results"]):
                        st.subheader(f"{i+1}. {file_info['name']} ({file_info['size']:.1f} KB)")
                        display_promptfoo_results(file_info["data"])
                        st.divider()
                else:
                    st.info("No PromptFoo test results found")
                    
            tab_idx += 1
            progress.progress(70)
            
            # RAG Query Results
            with result_tabs[tab_idx]:
                if result_files["RAG Query Results"]:
                    for i, file_info in enumerate(result_files["RAG Query Results"]):
                        st.subheader(f"{i+1}. {file_info['name']} ({file_info['size']:.1f} KB)")
                        display_rag_results(file_info["data"])
                        st.divider()
                else:
                    st.info("No RAG query results found")
                    
            tab_idx += 1
            progress.progress(80)
            
            # RAG Index Info
            with result_tabs[tab_idx]:
                if result_files["RAG Index Info"]:
                    for i, file_info in enumerate(result_files["RAG Index Info"]):
                        st.subheader(f"{i+1}. {file_info['name']} ({file_info['size']:.1f} KB)")
                        st.json(file_info["data"])
                        st.divider()
                else:
                    st.info("No RAG index information found")
                    
            tab_idx += 1
            progress.progress(90)
            
            # Markdown Reports
            with result_tabs[tab_idx]:
                if result_files["Markdown Reports"]:
                    for i, file_info in enumerate(result_files["Markdown Reports"]):
                        st.subheader(f"{i+1}. {file_info['name']} ({file_info['size']:.1f} KB)")
                        
                        try:
                            with open(file_info["path"], "r", encoding="utf-8") as f:
                                content = f.read()
                                st.markdown(content)
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                            
                        st.divider()
                else:
                    st.info("No markdown reports found")
                    
            tab_idx += 1
            
            # Other Results
            with result_tabs[tab_idx]:
                if result_files["Other Results"]:
                    for i, file_info in enumerate(result_files["Other Results"]):
                        st.subheader(f"{i+1}. {file_info['name']} ({file_info['size']:.1f} KB)")
                        
                        if file_info.get("type") == "json":
                            try:
                                if "data" in file_info:
                                    st.json(file_info["data"])
                                else:
                                    with open(file_info["path"], "r", encoding="utf-8") as f:
                                        content = f.read()
                                        
                                    if "error" in file_info:
                                        st.error(file_info["error"])
                                        st.text(content)
                                    else:
                                        try:
                                            data = json.loads(content)
                                            st.json(data)
                                        except json.JSONDecodeError:
                                            st.text(content)
                            except Exception as e:
                                st.error(f"Error displaying file: {str(e)}")
                        else:
                            try:
                                with open(file_info["path"], "r", encoding="utf-8") as f:
                                    content = f.read()
                                    st.text(content)
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                                
                        st.divider()
                else:
                    st.info("No other results found")
        else:
            st.warning("No valid result files found")
            
        # Complete the progress
        progress.progress(100)

def is_package_installed(package_name):
    """Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if the package is installed, False otherwise
    """
    # Special handling for promptfoo which is an npm package, not Python
    if package_name == "promptfoo":
        try:
            # Try global installation
            subprocess.run(["promptfoo", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try npx as a fallback
            try:
                subprocess.run(["npx", "promptfoo", "--version"], check=True, capture_output=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    else:
        # For Python packages
        try:
            # Try importing the package
            __import__(package_name)
            return True
        except ImportError:
            # Check if it's a command-line tool
            try:
                subprocess.run([package_name, "--version"], check=True, capture_output=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False

def get_ollama_models(base_url="http://ollama:11434"):
    """Get list of available models from Ollama server.
    
    Args:
        base_url: Ollama server base URL
        
    Returns:
        List of model names or empty list if unavailable
    """
    import requests
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                return [model["name"] for model in data["models"]]
        return []
    except Exception as e:
        st.warning(f"Could not connect to Ollama server at {base_url}: {str(e)}")
        return []

def configuration_ui():
    """UI for configuration."""
    st.header("Configuration")
    
    # Load default configuration
    default_config = load_config()
    
    # Choose configuration section
    config_section = st.radio(
        "Configuration Section",
        options=["LLM Settings", "Full Configuration"],
        index=0,
    )
    
    if config_section == "LLM Settings":
        st.subheader("Ollama Settings")
        
        # Get current Ollama settings
        llm_config = default_config.get("llm", {})
        rewriting_config = default_config.get("rewriting", {})
        generation_config = default_config.get("generation", {})
        
        # Ollama base URL
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value=llm_config.get("ollama_base_url", "http://ollama:11434"),
            help="Base URL for Ollama API server",
        )
        
        # Query available models
        ollama_models = get_ollama_models(ollama_base_url)
        if ollama_models:
            st.success(f"Found {len(ollama_models)} models on Ollama server")
        else:
            st.warning("Could not retrieve models from Ollama server. Using default options.")
            ollama_models = ["llama3", "llama3.2-vision", "mistral", "dolphin-phi3", "phi3"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rewriting model
            rewriting_model = st.selectbox(
                "Rewriting Model",
                options=ollama_models,
                index=ollama_models.index(rewriting_config.get("ollama_model", "llama3")) if rewriting_config.get("ollama_model", "llama3") in ollama_models else 0,
                help="Model used for query rewriting"
            )
            
            # Rewriting temperature
            rewriting_temp = st.slider(
                "Rewriting Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(rewriting_config.get("temperature", 0.1)),
                step=0.1,
                help="Temperature for query rewriting (lower = more focused)"
            )
        
        with col2:
            # Generation model
            generation_model = st.selectbox(
                "Generation Model",
                options=ollama_models,
                index=ollama_models.index(generation_config.get("ollama_model", "mistral")) if generation_config.get("ollama_model", "mistral") in ollama_models else 0,
                help="Model used for response generation"
            )
            
            # Generation temperature
            generation_temp = st.slider(
                "Generation Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(generation_config.get("temperature", 0.7)),
                step=0.1,
                help="Temperature for response generation (higher = more creative)"
            )
        
        # Max tokens
        max_tokens = st.slider(
            "Max Response Tokens",
            min_value=100,
            max_value=2000,
            value=int(generation_config.get("max_tokens", 1000)),
            step=100,
            help="Maximum number of tokens in the generated response"
        )
        
        # Create updated config
        updated_config = default_config.copy()
        
        # Update LLM settings
        if "llm" not in updated_config:
            updated_config["llm"] = {}
        updated_config["llm"]["provider"] = "ollama"
        updated_config["llm"]["ollama_base_url"] = ollama_base_url
        
        # Update rewriting settings
        if "rewriting" not in updated_config:
            updated_config["rewriting"] = {}
        updated_config["rewriting"]["llm_type"] = "ollama"
        updated_config["rewriting"]["ollama_model"] = rewriting_model
        updated_config["rewriting"]["temperature"] = rewriting_temp
        updated_config["rewriting"]["ollama_base_url"] = ollama_base_url
        
        # Update generation settings
        if "generation" not in updated_config:
            updated_config["generation"] = {}
        updated_config["generation"]["llm_type"] = "ollama"
        updated_config["generation"]["ollama_model"] = generation_model
        updated_config["generation"]["temperature"] = generation_temp
        updated_config["generation"]["max_tokens"] = max_tokens
        updated_config["generation"]["ollama_base_url"] = ollama_base_url
        
        # Save button
        if st.button("Apply Settings", type="primary"):
            try:
                # Find config path
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.toml")
                
                # Read existing config to preserve structure and comments
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        existing_config = f.read()
                    
                    # Update only the relevant sections
                    import re
                    import toml
                    
                    # Convert to string for writing back to file
                    # We'll need to update specific sections
                    new_llm_section = f"""[llm]
                    provider = "ollama"  # "ollama", "llamacpp", "openai"
                    # Make sure this URL is directly accessible from your execution environment
                    ollama_base_url = "{ollama_base_url}"  # Ollama API URL 
                    ollama_timeout = 30  # Timeout in seconds
                    llamacpp_path = ""  # Path to llama.cpp executable
                    """
                    
                    new_rewriting_section = f"""[rewriting]
                    enabled = true
                    llm_type = "{{{{llm.provider}}}}"  # Inherit from llm section
                    temperature = {rewriting_temp}
                    max_tokens = 200
                    
                    # Model names by provider type
                    ollama_model = "{rewriting_model}"  # Model name for Ollama
                    llamacpp_model = "llama3"  # Model name for LLaMA.cpp
                    openai_model = "{{{{llm.openai_model}}}}"  # Inherit from llm section
                    
                    # Provider-specific settings (inherited from llm section)
                    ollama_base_url = "{{{{llm.ollama_base_url}}}}"
                    """
                    
                    new_generation_section = f"""[generation]
                    llm_type = "{{{{llm.provider}}}}"  # Inherit from llm section
                    temperature = {generation_temp}
                    max_tokens = {max_tokens}
                    
                    # Model names by provider type
                    ollama_model = "{generation_model}"  # Model name for Ollama
                    llamacpp_model = "mistral"  # Model name for LLaMA.cpp
                    openai_model = "{{{{llm.openai_model}}}}"  # Inherit from llm section
                    
                    # Provider-specific settings (inherited from llm section)
                    ollama_base_url = "{{{{llm.ollama_base_url}}}}"
                    """
                    
                    # Now generate a new TOML config that merges existing sections with our updated ones
                    import tempfile
                    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.toml') as tf:
                        temp_path = tf.name
                        # Write updated TOML
                        with open(config_path, 'w') as f:
                            # Replace sections with our updated versions
                            # This is a simple approach - a more robust solution would be to parse and modify the TOML
                            sections = re.split(r'\[([^\]]+)\]', existing_config)
                            for i in range(1, len(sections), 2):
                                section_name = sections[i].strip()
                                if section_name == "llm":
                                    sections[i+1] = "\n" + new_llm_section + "\n"
                                elif section_name == "rewriting":
                                    sections[i+1] = "\n" + new_rewriting_section + "\n"
                                elif section_name == "generation":
                                    sections[i+1] = "\n" + new_generation_section + "\n"
                            
                            # Reconstruct the file
                            new_content = ""
                            for i in range(0, len(sections)):
                                if i > 0 and i % 2 == 1:
                                    new_content += f"[{sections[i]}]"
                                else:
                                    new_content += sections[i]
                            
                            f.write(new_content)
                        
                    st.success("Settings applied successfully!")
                else:
                    st.error("Config file not found! Please check your installation.")
            except Exception as e:
                st.error(f"Error saving settings: {str(e)}")
        
        # Display equivalent TOML
        with st.expander("Preview Settings as TOML"):
            st.code(f"""[llm]
            provider = "ollama"
            ollama_base_url = "{ollama_base_url}"
            ollama_timeout = 30
            
            [rewriting]
            enabled = true
            llm_type = "ollama"
            ollama_model = "{rewriting_model}"
            temperature = {rewriting_temp}
            max_tokens = 200
            ollama_base_url = "{ollama_base_url}"
            
            [generation]
            llm_type = "ollama"
            ollama_model = "{generation_model}"
            temperature = {generation_temp}
            max_tokens = {max_tokens}
            ollama_base_url = "{ollama_base_url}"
            """, language="toml")
    
    else:  # Full configuration
        # Choose configuration format
        config_format = st.radio(
            "Configuration Format",
            options=["TOML", "JSON"],
            index=0,
        )
        
        if config_format == "TOML":
            # Display TOML configuration
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.toml")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_text = f.read()
            else:
                config_text = "# Configuration not found"
            
            # Display configuration editor
            st.subheader("Edit Configuration")
            edited_config = st.text_area("Configuration (TOML)", value=config_text, height=600)
            
            # Save configuration
            col1, col2 = st.columns(2)
            
            with col1:
                save_path = st.text_input(
                    "Save Path",
                    placeholder="/path/to/config.toml",
                    value=config_path,
                )
            
            with col2:
                save_button = st.button("Save Configuration")
                
                if save_button:
                    if not save_path:
                        st.error("Please enter a save path")
                    else:
                        try:
                            # Create directory if it doesn't exist
                            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                            
                            # Save configuration
                            with open(save_path, "w") as f:
                                f.write(edited_config)
                                
                            st.success(f"Configuration saved to {save_path}")
                        except Exception as e:
                            st.error(f"Error saving configuration: {str(e)}")
        elif config_format == "JSON":
            # Convert to JSON
            config_text = json.dumps(default_config, indent=2)
            
            # Display configuration editor
            st.subheader("Edit Configuration")
            edited_config = st.text_area("Configuration (JSON)", value=config_text, height=600)
            
            try:
                # Try to parse the edited configuration
                json.loads(edited_config)
                st.success("Configuration is valid JSON")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {str(e)}")
            
            # Save configuration
            col1, col2 = st.columns(2)
            
            with col1:
                save_path = st.text_input(
                    "Save Path",
                    placeholder="/path/to/config.json",
                )
            
            with col2:
                save_button = st.button("Save Configuration")
                
                if save_button:
                    if not save_path:
                        st.error("Please enter a save path")
                    else:
                        try:
                            # Create directory if it doesn't exist
                            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                            
                            # Save configuration
                            with open(save_path, "w") as f:
                                f.write(edited_config)
                                
                            st.success(f"Configuration saved to {save_path}")
                        except Exception as e:
                            st.error(f"Error saving configuration: {str(e)}")

def process_documents(
    input_type: str,
    uploaded_files: Optional[List] = None,
    input_path: Optional[str] = None,
    recursive: bool = False,
    output_formats: List[str] = ["markdown", "json"],
    output_path: Optional[str] = None,
    use_temp_dir: bool = True,
    chunking_strategy: str = "hybrid",
    semantic_chunking: bool = True,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    log_level: str = "info",
    config: Optional[Dict] = None,
    is_ocr_workflow: bool = False,
):
    """Process documents with the selected options.
    
    Args:
        input_type: Type of input ("Upload Files" or "Local Path")
        uploaded_files: List of uploaded files (if input_type is "Upload Files")
        input_path: Local path to files (if input_type is "Local Path")
        recursive: Whether to process subdirectories (if input_type is "Local Path")
        output_formats: List of output formats
        output_path: Output directory path
        use_temp_dir: Whether to use a temporary directory for output
        chunking_strategy: Chunking strategy (hybrid, semantic, size)
        semantic_chunking: Whether to enable semantic chunking
        ocr_enabled: Whether to enable OCR
        ocr_language: OCR language
        log_level: Logging level
        config: Optional config dictionary (overrides individual parameters)
        is_ocr_workflow: Whether this is an OCR-specific workflow (displays additional info)
    """
    # Create or use provided configuration
    if config is None:
        config = {
            "processing": {
                "dry_run": False,
                "recursive": recursive,
            },
            "logging": {
                "log_level": log_level,
            },
            "extraction": {
                "ocr_enabled": ocr_enabled,
                "ocr_language": ocr_language,
            },
            "segmentation": {
                "semantic_chunking": semantic_chunking,
                "chunking_strategy": chunking_strategy,
            },
            "export": {
                "formats": output_formats,
            },
        }
    
    # Create processor
    processor = DocumentProcessor(config=config)
    
    # Determine output directory
    if use_temp_dir:
        output_dir = tempfile.mkdtemp(prefix="raggiro_")
    else:
        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
    
    # Store output directory in session state for later use
    st.session_state["last_output_dir"] = output_dir
    
    # Process documents
    if input_type == "Upload Files":
        # Create a temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp(prefix="raggiro_uploads_")
        
        # Save uploaded files to the temporary directory
        file_paths = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            file_paths.append(file_path)
        
        # Process each file
        results = []
        progress_bar = st.progress(0, "Processing documents")
        
        for i, file_path in enumerate(file_paths):
            result = processor.process_file(file_path, output_dir)
            results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(file_paths))
        
        # Generate summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        summary = {
            "total_files": len(results),
            "successful_files": successful,
            "failed_files": failed,
            "success_rate": round(successful / len(results) * 100, 2) if results else 0,
        }
        
        # Display summary
        st.subheader("Processing Summary")
        st.write(f"Processed {summary['total_files']} files")
        st.write(f"Successfully processed: {summary['successful_files']} files ({summary['success_rate']}%)")
        st.write(f"Failed: {summary['failed_files']} files")
        
        # Display processed files
        if successful > 0:
            with st.expander("View processed files"):
                # Create a DataTable with the processed files
                data = []
                
                for result in results:
                    if result["success"]:
                        file_path = result["file_path"]
                        file_name = os.path.basename(file_path)
                        output_files = []
                        
                        for fmt in output_formats:
                            base_name = os.path.splitext(file_name)[0]
                            output_file = os.path.join(output_dir, f"{base_name}.{fmt}")
                            
                            if os.path.exists(output_file):
                                output_files.append(output_file)
                        
                        data.append({
                            "File": file_name,
                            "Output Files": ", ".join([os.path.basename(f) for f in output_files]),
                            "Metadata": json.dumps(result.get("document", {}).get("metadata", {}), indent=2)
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
        
        # Display output directory
        st.subheader("Output Directory")
        st.code(output_dir)
        
        # Clean up temporary directory for uploaded files
        # Note: We don't clean up the output directory if it's a temporary directory,
        # so the user can access the results
        st.write("Note: Uploaded files will be deleted when you close the application")
        
        # Display failed files
        if failed > 0:
            with st.expander("View failed files"):
                for result in results:
                    if not result["success"]:
                        st.error(f"❌ {os.path.basename(result['file_path'])}: {result.get('error', 'Unknown error')}")
    else:
        # Process local directory
        result = processor.process_directory(input_path, output_dir, recursive=recursive)
        
        if not result["success"]:
            st.error(f"Error processing directory: {result.get('error', 'Unknown error')}")
            return
        
        # Display summary
        summary = result["summary"]
        
        st.subheader("Processing Summary")
        st.write(f"Processed {summary['total_files']} files")
        st.write(f"Successfully processed: {summary['successful_files']} files ({summary['success_rate']}%)")
        st.write(f"Failed: {summary['failed_files']} files")
        
        # Display output directory
        st.subheader("Output Directory")
        st.code(output_dir)
        
        # Display processed files
        if summary["successful_files"] > 0:
            # Create a DataFrame to display processed files
            data = []
            ocr_files = []
            corrected_files = []
            chunk_stats = {}
            
            # Collect information for all files
            for r in result["results"]:
                if r["success"]:
                    file_path = r["file_path"]
                    file_name = os.path.basename(file_path)
                    output_files = []
                    
                    # Find output files
                    for fmt in output_formats:
                        base_name = os.path.splitext(file_name)[0]
                        output_file = os.path.join(output_dir, f"{base_name}.{fmt}")
                        
                        if os.path.exists(output_file):
                            output_files.append(output_file)
                    
                    # Basic metadata for all files
                    entry = {
                        "File": file_name,
                        "Output Files": ", ".join([os.path.basename(f) for f in output_files]),
                        "Metadata": json.dumps(r.get("document", {}).get("metadata", {}), indent=2)
                    }
                    data.append(entry)
                    
                    # Collect OCR and spelling correction stats
                    document = r.get("document", {})
                    if document.get("extraction_method") in ["pdf_ocr", "image_ocr"]:
                        ocr_files.append(file_name)
                    
                    if document.get("metadata", {}).get("spelling_corrected"):
                        corrected_files.append(file_name)
                    
                    # Collect chunking stats
                    chunks = document.get("chunks", [])
                    if chunks:
                        chunk_stats[file_name] = {
                            "chunks": len(chunks),
                            "avg_size": sum(len(chunk.get("text", "")) for chunk in chunks) // max(1, len(chunks)),
                            "semantic_chunks": sum(1 for chunk in chunks if chunk.get("semantic", False))
                        }
            
            # Display basic file table
            with st.expander("View processed files"):
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
            
            # Show OCR-specific information if in OCR workflow
            if is_ocr_workflow:
                with st.expander("OCR & Correction Details", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("OCR Extraction")
                        if ocr_files:
                            st.success(f"{len(ocr_files)} files processed with OCR:")
                            for f in ocr_files:
                                st.write(f"- {f}")
                        else:
                            st.info("No files processed with OCR")
                    
                    with col2:
                        st.subheader("Spelling Correction")
                        if corrected_files:
                            st.success(f"{len(corrected_files)} files had spelling correction applied:")
                            for f in corrected_files:
                                st.write(f"- {f}")
                        else:
                            st.info("No files had spelling correction applied")
                
                with st.expander("Chunking Results", expanded=True):
                    st.subheader("Document Segmentation")
                    
                    if chunk_stats:
                        # Create a DataFrame for chunk statistics
                        chunk_data = []
                        for filename, stats in chunk_stats.items():
                            chunk_data.append({
                                "File": filename,
                                "Total Chunks": stats["chunks"],
                                "Avg. Chunk Size (chars)": stats["avg_size"],
                                "Semantic Chunks": stats["semantic_chunks"],
                            })
                        
                        # Display as table
                        chunk_df = pd.DataFrame(chunk_data)
                        st.table(chunk_df)
                        
                        # Show warning if any document has very few chunks
                        few_chunks = [f for f, stats in chunk_stats.items() if stats["chunks"] < 3]
                        if few_chunks:
                            st.warning(f"The following documents have fewer than 3 chunks, which might affect retrieval quality: {', '.join(few_chunks)}")
                    else:
                        st.info("No chunking information available")
        
        # Display failed files
        if summary["failed_files"] > 0:
            with st.expander("View failed files"):
                for r in result["results"]:
                    if not r["success"]:
                        st.error(f"❌ {os.path.basename(r['file_path'])}: {r.get('error', 'Unknown error')}")

def run_rag_tests(
    input_path: str,
    prompt_file_path: str,
    output_path: str,
):
    """Run RAG tests on processed documents.
    
    Args:
        input_path: Path to processed documents
        prompt_file_path: Path to prompts file
        output_path: Output directory for results
    """
    try:
        # Check if the input directory exists
        if not os.path.exists(input_path):
            st.error(f"Input directory not found: {input_path}")
            return
        
        # Verify if we have test_semantic_chunking.py script
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create progress bar for test setup
        setup_progress = st.progress(0, "Setting up test environment")
        st.info("Setting up test environment...")
        setup_progress.progress(10)
        
        # Look for the script in various locations with more thorough search
        script_locations = [
            os.path.join(repo_dir, "examples", "scripts", "test_semantic_chunking.py"),
            os.path.join(repo_dir, "examples", "test_semantic_chunking.py"),
            os.path.join(repo_dir, "test_semantic_chunking.py"),
            os.path.join(repo_dir, "raggiro", "examples", "test_semantic_chunking.py"),
            os.path.join(repo_dir, "raggiro", "testing", "test_semantic_chunking.py"),
        ]
        
        setup_progress.progress(20)
        
        semantic_chunking_script = None
        for location in script_locations:
            if os.path.exists(location):
                semantic_chunking_script = location
                break
                
        # If not found in predefined locations, try a recursive search (limiting depth)
        if not semantic_chunking_script:
            for root, dirs, files in os.walk(repo_dir, topdown=True):
                if len(root.split(os.sep)) - len(repo_dir.split(os.sep)) > 3:  # Limit search depth
                    dirs[:] = []  # Skip deeper directories
                    continue
                    
                if "test_semantic_chunking.py" in files:
                    semantic_chunking_script = os.path.join(root, "test_semantic_chunking.py")
                    break
        
        setup_progress.progress(40)
        
        # Look for promptfoo runner script with more thorough search
        promptfoo_locations = [
            os.path.join(repo_dir, "raggiro", "testing", "promptfoo_runner.py"),
            os.path.join(repo_dir, "testing", "promptfoo_runner.py"),
            os.path.join(repo_dir, "test_prompts", "promptfoo_runner.py"),
            os.path.join(repo_dir, "raggiro", "testing", "__init__.py"),  # It might be importable as a module
        ]
        
        promptfoo_script = None
        for location in promptfoo_locations:
            if os.path.exists(location):
                promptfoo_script = location
                if location.endswith("__init__.py"):
                    # If we found __init__.py, use the module directly
                    promptfoo_script = os.path.join(os.path.dirname(location), "promptfoo_runner.py")
                    if not os.path.exists(promptfoo_script):
                        # Create a simple wrapper script that imports the module
                        with open(os.path.join(os.path.dirname(location), "wrapper.py"), "w") as f:
                            f.write("""
import sys
from pathlib import Path
from raggiro.testing.promptfoo_runner import run_tests

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python wrapper.py <prompt_file> <output_dir> [<index_dir>]")
        sys.exit(1)
    
    prompt_file = sys.argv[1]
    output_dir = sys.argv[2]
    index_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = run_tests(prompt_file, output_dir, index_dir)
    print(f"Test completed: {result['success']}")
    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Tests run: {result.get('tests_run', 0)}")
        print(f"Results saved to: {result.get('output_file', 'Unknown')}")
""")
                        promptfoo_script = os.path.join(os.path.dirname(location), "wrapper.py")
                break
        
        # If not found in predefined locations, try a recursive search for promptfoo_runner.py
        if not promptfoo_script:
            for root, dirs, files in os.walk(repo_dir, topdown=True):
                if len(root.split(os.sep)) - len(repo_dir.split(os.sep)) > 3:  # Limit search depth
                    dirs[:] = []  # Skip deeper directories
                    continue
                    
                if "promptfoo_runner.py" in files:
                    promptfoo_script = os.path.join(root, "promptfoo_runner.py")
                    break
        
        setup_progress.progress(60)
        
        # Calculate paths
        # Use file base name for output directory to avoid conflicts
        input_base_name = os.path.basename(os.path.normpath(input_path))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_path, f"rag_test_{input_base_name}_{timestamp}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store the output directory in session state for easy access in the "View Results" tab
        st.session_state["last_test_dir"] = output_dir
        
        setup_progress.progress(70)
        
        # Display information
        st.subheader("RAG Test Information")
        
        # Create test info table
        test_info = pd.DataFrame([
            {"Parameter": "Input Directory", "Value": input_path},
            {"Parameter": "Test Prompts File", "Value": prompt_file_path},
            {"Parameter": "Output Directory", "Value": output_dir},
            {"Parameter": "Semantic Chunking Script", "Value": semantic_chunking_script or "Not found"},
            {"Parameter": "PromptFoo Script", "Value": promptfoo_script or "Not found"},
        ])
        st.table(test_info)
        
        setup_progress.progress(80)
        
        # Create container for log output with better styling
        log_container = st.empty()
        log_output = []
        
        # Create a function to update the log with timestamps
        def append_log(message, error=False):
            timestamp = time.strftime("%H:%M:%S")
            if error:
                log_output.append(f"[{timestamp}] ❌ {message}")
            else:
                log_output.append(f"[{timestamp}] {message}")
            log_container.code("\n".join(log_output))
        
        setup_progress.progress(90)
        
        # Execute the test
        if not semantic_chunking_script:
            st.error("Test semantic chunking script not found. Please check the installation.")
            append_log("Test semantic chunking script not found. Please check the installation.", error=True)
            return
        
        # Prepare test files
        # Check if we have markdown or json files in the input directory
        test_file_candidates = []
        test_file_candidates.extend(glob.glob(os.path.join(input_path, "*.pdf")))
        test_file_candidates.extend(glob.glob(os.path.join(input_path, "*.json")))
        test_file_candidates.extend(glob.glob(os.path.join(input_path, "*.md")))
        
        if not test_file_candidates:
            st.error("No test files (PDF, JSON, MD) found in the input directory")
            append_log("No test files found in the input directory", error=True)
            return
        
        setup_progress.progress(100)
        
        # Create test progress bar
        test_progress = st.progress(0, "Running tests")
        
        # Run the test semantic chunking script
        append_log("=== Running semantic chunking test ===")
        
        # Use the first file as input for the test
        test_file = test_file_candidates[0]
        st.info(f"Testing with file: {os.path.basename(test_file)}")
        
        # Load configuration to get Ollama settings
        config_path = os.path.join(repo_dir, "config", "config.toml")
        try:
            config = load_config(str(config_path))
            append_log(f"Loaded config from {config_path}")
            
            # Get Ollama settings from config
            ollama_url = config.get('llm', {}).get('ollama_base_url', 'http://ollama:11434')
            rewriting_model = config.get('rewriting', {}).get('ollama_model', 'llama3')
            generation_model = config.get('generation', {}).get('ollama_model', 'mistral')
            
            append_log(f"Using Ollama URL: {ollama_url}")
            append_log(f"Using rewriting model: {rewriting_model}")
            append_log(f"Using generation model: {generation_model}")
        except Exception as e:
            append_log(f"Warning: Error loading config: {str(e)}, using default values", error=True)
            # Default values
            ollama_url = "http://ollama:11434"
            rewriting_model = "llama3"
            generation_model = "mistral"
        
        # Prepare the command with all available options and Ollama config
        command = [
            "python", 
            semantic_chunking_script, 
            "--input", test_file, 
            "--output", output_dir,
            "--index", os.path.join(output_dir, "index"),
            # Add Ollama configuration options (modify if your script accepts these)
            "--ollama-url", ollama_url,
            "--rewriting-model", rewriting_model,
            "--generation-model", generation_model
        ]
        
        # Try to find test queries in the prompt file
        try:
            with open(prompt_file_path, "r") as f:
                if prompt_file_path.endswith(".yaml") or prompt_file_path.endswith(".yml"):
                    prompt_data = yaml.safe_load(f)
                    if "prompts" in prompt_data and isinstance(prompt_data["prompts"], list):
                        # Take first 3 prompts as test queries
                        test_queries = prompt_data["prompts"][:3]
                        if test_queries:
                            command.extend(["--queries"] + test_queries)
                            append_log(f"Using {len(test_queries)} test queries from prompt file")
        except Exception as e:
            append_log(f"Error loading prompt file for queries: {str(e)}", error=True)
        
        append_log(f"Running command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Wait for the process to finish and update the log
        test_progress.progress(25)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                append_log(output.strip())
        
        # Check the return code
        return_code = process.wait()
        
        if return_code != 0:
            # Get error output
            error_output = process.stderr.read()
            append_log(f"Error: Process exited with code {return_code}", error=True)
            append_log(error_output, error=True)
            
            st.error("Test semantic chunking failed")
            test_progress.progress(100)
            # Continue anyway to see if we have any partial results
        else:
            st.success("Semantic chunking test completed successfully")
            test_progress.progress(50)
        
        # Run the promptfoo runner if available
        if promptfoo_script:
            append_log("\n=== Running promptfoo tests ===")
            
            # Check if promptfoo is installed (try both global and npx versions)
            promptfoo_installed = False
            try:
                # Try global installation
                subprocess.run(["promptfoo", "--version"], check=True, capture_output=True)
                promptfoo_installed = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Try npx as a fallback
                try:
                    subprocess.run(["npx", "promptfoo", "--version"], check=True, capture_output=True)
                    promptfoo_installed = True
                    append_log("Using npx to run PromptFoo (local installation detected)")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    promptfoo_installed = False
                    
            if not promptfoo_installed:
                append_log("⚠️ PromptFoo is not installed. PromptFoo is a Node.js application that must be installed via npm.", error=True)
                append_log("⚠️ Test results will be limited without PromptFoo.", error=True)
                
                with st.warning("PromptFoo is not installed. Some test features will be unavailable."):
                    st.markdown("### Install PromptFoo")
                    
                    # Find the installation script
                    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    install_script_path = os.path.join(repo_dir, "scripts", "installation", "install_promptfoo.sh")
                    verify_script_path = os.path.join(repo_dir, "scripts", "installation", "verify_promptfoo.sh")
                    
                    if os.path.exists(install_script_path):
                        st.info("PromptFoo installation script found. Use one of these methods to install:")
                        
                        tab1, tab2 = st.tabs(["Method 1: User Installation", "Method 2: Project Installation"])
                        
                        with tab1:
                            st.markdown("#### User Installation (Recommended)")
                            st.markdown("This installs PromptFoo in your user directory, avoiding permission issues:")
                            st.code(f"chmod +x {install_script_path} && {install_script_path}", language="bash")
                            st.markdown("**After installation**, add this to your ~/.bashrc or ~/.profile:")
                            st.code("export PATH=\"$HOME/.npm-global/bin:$PATH\"", language="bash")
                            st.markdown("Then reload your terminal or run:")
                            st.code("source ~/.bashrc", language="bash")
                        
                        with tab2:
                            st.markdown("#### Project Installation (Alternative)")
                            st.markdown("This installs PromptFoo locally in the project directory:")
                            st.code("npm install promptfoo --save", language="bash")
                            st.markdown("Then use `npx promptfoo` instead of just `promptfoo`:")
                            st.code("npx promptfoo --version", language="bash")
                        
                        if os.path.exists(verify_script_path):
                            st.markdown("#### Troubleshooting")
                            st.markdown("If you encounter issues, run the verification script:")
                            st.code(f"chmod +x {verify_script_path} && {verify_script_path}", language="bash")
                    else:
                        st.info("PromptFoo is a Node.js application used for advanced RAG testing. Install it with:")
                        st.code("npm install -g promptfoo", language="bash")
                # Continue with the test but skip the promptfoo part
                test_progress.progress(100)
                return
                
            # Use the directory with the test index for better retrieval
            index_dir = os.path.join(output_dir, "index")
            if os.path.exists(index_dir):
                command = ["python", promptfoo_script, prompt_file_path, output_dir, index_dir]
                append_log(f"Using index directory: {index_dir}")
            else:
                command = ["python", promptfoo_script, prompt_file_path, output_dir]
                append_log("No index directory found, using default index")
                
            # Display a note about which PromptFoo method is being used
            try:
                which_promptfoo = subprocess.run(["which", "promptfoo"], check=True, capture_output=True, text=True)
                append_log(f"Using global PromptFoo from: {which_promptfoo.stdout.strip()}")
            except subprocess.CalledProcessError:
                append_log("Using local PromptFoo with npx")
            
            append_log(f"Running command: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            
            # Wait for the process to finish and update the log
            test_progress.progress(75)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    append_log(output.strip())
            
            # Check the return code
            return_code = process.wait()
            
            if return_code != 0:
                # Get error output
                error_output = process.stderr.read()
                append_log(f"Error: Process exited with code {return_code}", error=True)
                append_log(error_output, error=True)
                
                st.error("PromptFoo test failed")
                # Continue anyway because we might have partial results
            else:
                st.success("PromptFoo tests completed successfully")
            
        test_progress.progress(100)
        
        # Find JSON and Markdown result files
        result_files = glob.glob(os.path.join(output_dir, "*.json"))
        result_files.extend(glob.glob(os.path.join(output_dir, "*.md")))
        
        # Display results
        st.subheader("Test Results")
        
        if not result_files:
            st.warning("No result files found")
        else:
            st.success(f"Test completed with {len(result_files)} output files")
            
            # Categorize result files
            result_categories = {
                "PromptFoo Results": [],
                "RAG Query Results": [],
                "Other Results": []
            }
            
            for result_file in result_files:
                file_name = os.path.basename(result_file)
                try:
                    if result_file.endswith(".json"):
                        with open(result_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        if "prompts" in data or "summary" in data:
                            result_categories["PromptFoo Results"].append(result_file)
                        elif isinstance(data, list) and all("query" in item for item in data):
                            result_categories["RAG Query Results"].append(result_file)
                        else:
                            result_categories["Other Results"].append(result_file)
                    else:
                        result_categories["Other Results"].append(result_file)
                except Exception:
                    result_categories["Other Results"].append(result_file)
            
            # Display results by category with tabs
            result_tabs = st.tabs(["PromptFoo Results", "RAG Query Results", "All Files"])
            
            with result_tabs[0]:  # PromptFoo Results
                if result_categories["PromptFoo Results"]:
                    for result_file in result_categories["PromptFoo Results"]:
                        file_name = os.path.basename(result_file)
                        with open(result_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        st.subheader(f"Test Results: {file_name}")
                        display_promptfoo_results(data)
                else:
                    # Check if promptfoo is installed
                    try:
                        subprocess.run(["promptfoo", "--version"], check=True, capture_output=True)
                        st.info("No PromptFoo result files found.")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        with st.error("PromptFoo is not installed. PromptFoo is a Node.js application that must be installed via npm."):
                            # Find the installation script
                            repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                            install_script_path = os.path.join(repo_dir, "scripts", "installation", "install_promptfoo.sh")
                            
                            if os.path.exists(install_script_path):
                                st.markdown("**Installation Script Available:**")
                                st.code(f"chmod +x {install_script_path} && {install_script_path}", language="bash")
                            else:
                                st.code("npm install -g promptfoo", language="bash")
                            
                            st.info("After installation, restart the application and try again.")
            
            with result_tabs[1]:  # RAG Query Results
                if result_categories["RAG Query Results"]:
                    for result_file in result_categories["RAG Query Results"]:
                        file_name = os.path.basename(result_file)
                        with open(result_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        st.subheader(f"RAG Results: {file_name}")
                        # Add a file_id to make keys unique across different result files
                        file_id = hash(file_name) % 10000
                        display_rag_results(data, file_prefix=f"file_{file_id}")
                else:
                    st.info("No RAG query result files found")
            
            with result_tabs[2]:  # All Files
                st.subheader(f"All Output Files ({len(result_files)})")
                for category, files in result_categories.items():
                    if files:
                        with st.expander(f"{category} ({len(files)})"):
                            for result_file in files:
                                file_name = os.path.basename(result_file)
                                file_size = os.path.getsize(result_file) / 1024  # KB
                                st.markdown(f"- **{file_name}** ({file_size:.1f} KB)")
                                if result_file.endswith(".md"):
                                    try:
                                        with open(result_file, "r", encoding="utf-8") as f:
                                            content = f.read()
                                        st.markdown(content)
                                    except Exception as e:
                                        st.error(f"Error reading file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error running tests: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def display_promptfoo_results(data):
    """Display formatted PromptFoo test results with enhanced visualizations.
    
    Args:
        data: PromptFoo test results data
    """
    # Check if we have a summary
    if "summary" in data:
        summary = data["summary"]
        
        st.subheader("Test Summary")
        
        # Create metrics with more visual feedback
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Add emoji based on pass rate
            pass_rate = summary.get('pass_rate', 0)
            emoji = "🎉" if pass_rate > 90 else "👍" if pass_rate > 70 else "🤔" if pass_rate > 50 else "⚠️"
            st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{emoji}")
        
        with col2:
            st.metric("Tests Passed", summary.get("pass", 0))
        
        with col3:
            st.metric("Tests Failed", summary.get("fail", 0))
        
        with col4:
            st.metric("Total Tests", summary.get("total", 0))
        
        # Add a visual pass/fail chart if we have enough data
        if summary.get("pass", 0) > 0 or summary.get("fail", 0) > 0:
            st.subheader("Success Distribution")
            
            # Create a simple bar chart
            pass_count = summary.get("pass", 0)
            fail_count = summary.get("fail", 0)
            total = pass_count + fail_count
            
            if total > 0:
                # Create a dataframe for the chart
                chart_data = pd.DataFrame({
                    "Status": ["Pass", "Fail"],
                    "Count": [pass_count, fail_count],
                    "Percentage": [pass_count/total*100, fail_count/total*100]
                })
                
                # Display as a bar chart
                st.bar_chart(
                    chart_data.set_index("Status")["Count"],
                    color=["#28a745", "#dc3545"] if pass_count > fail_count else ["#dc3545", "#28a745"]
                )
    
    # Check if we have evaluations data
    if "evaluations" in data:
        st.subheader("Evaluation Metrics")
        
        try:
            evals = data["evaluations"]
            
            if evals and isinstance(evals, list):
                # Extract metrics from evaluations
                metrics = {}
                
                for eval_item in evals:
                    if "metric" in eval_item and "value" in eval_item:
                        metric_name = eval_item["metric"]
                        metric_value = eval_item["value"]
                        metrics[metric_name] = metric_value
                
                if metrics:
                    # Display metrics in a nice table
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics.keys()),
                        "Value": list(metrics.values())
                    })
                    st.table(metrics_df)
                    
                    # Try to visualize some common metrics
                    if "latency" in metrics:
                        st.subheader("Latency (ms)")
                        st.bar_chart({"Latency": [metrics["latency"]]})
                    
                    if "token_count" in metrics:
                        st.subheader("Token Usage")
                        st.bar_chart({"Tokens": [metrics["token_count"]]})
        except Exception as e:
            st.warning(f"Could not parse evaluation metrics: {str(e)}")
    
    # Check if we have prompts
    if "prompts" in data:
        st.subheader("Prompt Results")
        
        # Calculate statistics across all prompts
        all_scores = []
        all_responses = []
        prompt_stats = {"passed": 0, "failed": 0}
        
        for prompt_data in data["prompts"]:
            results = prompt_data.get("results", [])
            for result in results:
                if "score" in result:
                    all_scores.append(result["score"])
                if "response" in result:
                    all_responses.append(result["response"])
                if result.get("pass", False):
                    prompt_stats["passed"] += 1
                else:
                    prompt_stats["failed"] += 1
        
        # Overall statistics
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores) if all_scores else 0
            min_score = min(all_scores) if all_scores else 0
            
            st.info(f"Average score: {avg_score:.2f} (Min: {min_score:.2f}, Max: {max_score:.2f})")
        
        # Response length statistics
        if all_responses:
            response_lengths = [len(r) for r in all_responses]
            avg_length = sum(response_lengths) / len(response_lengths)
            st.info(f"Average response length: {avg_length:.0f} characters")
        
        # Display individual prompt results
        for i, prompt_data in enumerate(data["prompts"]):
            prompt = prompt_data["prompt"]
            results = prompt_data.get("results", [])
            
            # Calculate pass rate for this prompt
            prompt_pass_count = sum(1 for r in results if r.get("pass", False))
            prompt_pass_rate = prompt_pass_count / len(results) * 100 if results else 0
            
            # Add an emoji indicator
            emoji = "✅" if prompt_pass_rate == 100 else "⚠️" if prompt_pass_rate > 50 else "❌"
            
            with st.expander(f"{emoji} Prompt {i+1}: {prompt[:50]}{'...' if len(prompt) > 50 else ''} ({prompt_pass_count}/{len(results)} passed)"):
                st.write(f"**Prompt:** {prompt}")
                
                # Display all results for this prompt in a more visual way
                for j, result in enumerate(results):
                    # Determine if the test passed
                    passed = result.get("pass", False)
                    score = result.get("score", 0)
                    
                    # Create a styled container based on pass/fail
                    result_container = st.container()
                    with result_container:
                        status_text = "PASS" if passed else "FAIL"
                        status_emoji = "✅" if passed else "❌"
                        
                        # Display status with proper styling
                        st.markdown(f"### {status_emoji} Test {j+1}: {status_text} (Score: {score:.2f})")
                        
                        # Display metadata about the test
                        if "provider" in result:
                            st.markdown(f"**Provider:** {result['provider']}")
                            
                        # Display any assertions that were checked
                        if "assertions" in result and result["assertions"]:
                            assertions = result["assertions"]
                            st.markdown("**Assertions:**")
                            
                            for assertion in assertions:
                                assertion_type = assertion.get("type", "unknown")
                                assertion_pass = assertion.get("pass", False)
                                assertion_status = "✅" if assertion_pass else "❌"
                                
                                st.markdown(f"- {assertion_status} {assertion_type}")
                        
                        # Display the response in a scrollable text area
                        st.markdown("**Response:**")
                        response = result.get("response", "")
                        st.text_area("", response, height=150, disabled=True, 
                                    label_visibility="collapsed", key=f"promptfoo_response_{i}_{j}")
                        
                        # Display any additional metrics or data
                        if "latency" in result:
                            st.info(f"Latency: {result['latency']} ms")
                        
                        # Add a divider between results
                        st.divider()

def display_rag_results(data, file_prefix="default"):
    """Display formatted RAG query results with enhanced visualizations.
    
    Args:
        data: RAG query results data
        file_prefix: Prefix to use for keys to ensure uniqueness across files
    """
    # Calculate overall statistics for all queries
    if isinstance(data, list) and data:
        # Count successful and failed queries
        success_count = sum(1 for item in data if item.get("success", True) and not item.get("error"))
        error_count = len(data) - success_count
        
        # Display summary metrics
        st.subheader("RAG Query Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emoji = "🎉" if success_count == len(data) else "⚠️" if success_count > 0 else "❌"
            st.metric("Success Rate", f"{success_count/len(data)*100:.1f}%", delta=emoji)
        
        with col2:
            st.metric("Successful Queries", success_count)
        
        with col3:
            st.metric("Failed Queries", error_count)
        
        # Calculate response length statistics
        response_lengths = [len(item.get("response", "")) for item in data if "response" in item]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            max_length = max(response_lengths)
            min_length = min(response_lengths)
            
            st.info(f"Average response length: {avg_length:.0f} characters (Min: {min_length}, Max: {max_length})")
        
        # Calculate chunk usage statistics
        chunks_used = [item.get("chunks_used", 0) for item in data if "chunks_used" in item]
        if chunks_used:
            avg_chunks = sum(chunks_used) / len(chunks_used)
            max_chunks = max(chunks_used)
            min_chunks = min(chunks_used)
            
            st.info(f"Average chunks used: {avg_chunks:.1f} (Min: {min_chunks}, Max: {max_chunks})")
        
        # Check if query rewriting was used
        rewritten_count = sum(1 for item in data if "rewritten_query" in item)
        if rewritten_count > 0:
            st.info(f"Query rewriting used in {rewritten_count}/{len(data)} queries ({rewritten_count/len(data)*100:.1f}%)")
    
    # Display detailed results for each query
    st.subheader("Query Results")
    
    # If data is a list, display each item
    if isinstance(data, list):
        # Group by success/error status
        successful_queries = [item for item in data if item.get("success", True) and not item.get("error")]
        failed_queries = [item for item in data if not item.get("success", True) or item.get("error")]
        
        # Create tabs for successful and failed queries
        if successful_queries or failed_queries:
            query_tabs = st.tabs([
                f"Successful Queries ({len(successful_queries)})", 
                f"Failed Queries ({len(failed_queries)})",
                f"All Queries ({len(data)})"
            ])
            
            # Display successful queries
            with query_tabs[0]:
                if successful_queries:
                    for i, item in enumerate(successful_queries):
                        display_single_rag_result(item, i+1, tab_prefix=f"{file_prefix}_success")
                else:
                    st.info("No successful queries found")
            
            # Display failed queries
            with query_tabs[1]:
                if failed_queries:
                    for i, item in enumerate(failed_queries):
                        display_single_rag_result(item, i+1, is_error=True, tab_prefix=f"{file_prefix}_fail")
                else:
                    st.info("No failed queries found")
            
            # Display all queries
            with query_tabs[2]:
                for i, item in enumerate(data):
                    is_error = not item.get("success", True) or item.get("error")
                    display_single_rag_result(item, i+1, is_error=is_error, tab_prefix=f"{file_prefix}_all")
        else:
            st.warning("No query results to display")
    else:
        # Display a single result
        is_error = not data.get("success", True) or data.get("error")
        display_single_rag_result(data, 1, is_error=is_error, tab_prefix=f"{file_prefix}_single")

def display_single_rag_result(item, index, is_error=False, tab_prefix="main"):
    """Display a single RAG query result.
    
    Args:
        item: Query result data
        index: Query index number
        is_error: Whether this is an error result
        tab_prefix: Prefix to use for keys to avoid conflicts between tabs
    """
    import time
    
    query = item.get("query", "")
    query_prefix = query[:50] + ('...' if len(query) > 50 else '')
    
    # Create a unique identifier for this result to use in keys
    # Add timestamp to ensure uniqueness even if same content is displayed multiple times
    timestamp = int(time.time() * 1000)  # Milliseconds since epoch
    random_suffix = hash(f"{query}_{timestamp}") % 100000
    result_id = f"{tab_prefix}_{index}_{random_suffix}"
    
    status_emoji = "❌" if is_error else "✅"
    
    with st.expander(f"{status_emoji} Query {index}: {query_prefix}"):
        # Format the query and show metadata
        st.markdown(f"### Query {index}")
        st.markdown(f"**Original Query:** {query}")
        
        # Show rewritten query if available
        if "rewritten_query" in item and item["rewritten_query"] != query:
            st.markdown(f"**Rewritten Query:** {item['rewritten_query']}")
            # Show a visual comparison
            st.markdown("**Query Comparison:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.code(query)
            with col2:
                st.markdown("**Rewritten**")
                st.code(item["rewritten_query"])
        
        # Display error if present
        if "error" in item:
            st.error(f"Error: {item['error']}")
        
        # Show retrieval metadata
        retrieval_container = st.container()
        with retrieval_container:
            st.markdown("### Retrieval Stats")
            
            # Create columns for retrieval metrics
            rcol1, rcol2, rcol3 = st.columns(3)
            
            with rcol1:
                chunks = item.get("chunks_used", 0)
                st.metric("Chunks Used", chunks)
            
            with rcol2:
                if "retrieval_time" in item:
                    st.metric("Retrieval Time", f"{item['retrieval_time']:.2f} ms")
            
            with rcol3:
                if "generation_time" in item:
                    st.metric("Generation Time", f"{item['generation_time']:.2f} ms")
        
        # Show source chunks if available
        if "source_chunks" in item and item["source_chunks"]:
            with st.expander("Source Chunks Used"):
                source_chunks = item["source_chunks"]
                
                # Display each chunk with metadata
                for i, chunk in enumerate(source_chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    
                    # Show chunk metadata if available
                    if isinstance(chunk, dict):
                        # Show metadata such as relevance score, doc source, etc.
                        if "score" in chunk:
                            st.markdown(f"Score: {chunk['score']:.4f}")
                        
                        if "source" in chunk:
                            st.markdown(f"Source: {chunk['source']}")
                        
                        # Show the actual text content with unique keys based on result_id
                        if "text" in chunk:
                            st.text_area(f"Content (Chunk {i+1})", chunk["text"], height=100, disabled=True, 
                                         label_visibility="collapsed", key=f"chunk_text_{result_id}_{i}")
                    else:
                        # If it's just a string, show it directly
                        st.text_area(f"Content (Chunk {i+1})", str(chunk), height=100, disabled=True, 
                                     label_visibility="collapsed", key=f"chunk_str_{result_id}_{i}")
                    
                    st.divider()
        
        # Display the response
        if "response" in item:
            st.markdown("### Response")
            
            # Calculate response metrics
            response = item["response"]
            response_length = len(response)
            
            # Show response length
            st.info(f"Response length: {response_length} characters")
            
            # Show the full response - use a unique key based on result_id
            st.text_area("", response, height=200, disabled=True, 
                         label_visibility="collapsed", key=f"response_{result_id}")
        
        # Display additional metadata
        if "metadata" in item:
            with st.expander("Additional Metadata"):
                st.json(item["metadata"])

if __name__ == "__main__":
    # When run directly, just call the app function
    # This ensures it works in any Streamlit version
    run_app()