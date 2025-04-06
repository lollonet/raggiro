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
    tab1, tab2, tab3, tab4 = st.tabs(["Process Documents", "Test RAG", "View Results", "Configuration"])
    
    with tab1:
        process_documents_ui()
    
    with tab2:
        test_rag_ui()
    
    with tab3:
        view_results_ui()
    
    with tab4:
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

def test_rag_ui():
    """UI for testing RAG capabilities."""
    st.header("Test RAG System")
    
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
        # Search for test prompt files in multiple possible locations
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_prompt_locations = [
            os.path.join(repo_dir, "test_prompts"),
            os.path.join(repo_dir, "config", "test_prompts"),
            os.path.join(repo_dir, "config")
        ]
        
        prompt_files = []
        found_dir = None
        
        # Search in each location
        for test_dir in test_prompt_locations:
            if os.path.exists(test_dir):
                # Look for YAML files in this directory
                yaml_files = [
                    f for f in glob.glob(os.path.join(test_dir, "*.yaml")) 
                    if os.path.isfile(f)
                ]
                yaml_files.extend([
                    f for f in glob.glob(os.path.join(test_dir, "*.yml")) 
                    if os.path.isfile(f)
                ])
                
                if yaml_files:
                    prompt_files = [os.path.basename(f) for f in yaml_files]
                    found_dir = test_dir
                    break
        
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
                            "model": "ollama",
                            "temperature": 0.7,
                            "endpoint": "http://192.168.63.204:11434"
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
        progress = st.progress(0)
        
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

def configuration_ui():
    """UI for configuration."""
    st.header("Configuration")
    
    # Load default configuration
    default_config = load_config()
    
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
    else:
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
    """
    # Create configuration for the processor
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
        progress_bar = st.progress(0)
        
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
            with st.expander("View processed files"):
                # Create a DataTable with the processed files
                data = []
                
                for r in result["results"]:
                    if r["success"]:
                        file_path = r["file_path"]
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
                            "Metadata": json.dumps(r.get("document", {}).get("metadata", {}), indent=2)
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df)
        
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
        setup_progress = st.progress(0)
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
        test_progress = st.progress(0)
        
        # Run the test semantic chunking script
        append_log("=== Running semantic chunking test ===")
        
        # Use the first file as input for the test
        test_file = test_file_candidates[0]
        st.info(f"Testing with file: {os.path.basename(test_file)}")
        
        # Prepare the command with all available options
        command = [
            "python", 
            semantic_chunking_script, 
            "--input", test_file, 
            "--output", output_dir,
            "--index", os.path.join(output_dir, "index")
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
            
            # Use the directory with the test index for better retrieval
            index_dir = os.path.join(output_dir, "index")
            if os.path.exists(index_dir):
                command = ["python", promptfoo_script, prompt_file_path, output_dir, index_dir]
                append_log(f"Using index directory: {index_dir}")
            else:
                command = ["python", promptfoo_script, prompt_file_path, output_dir]
                append_log("No index directory found, using default index")
            
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
                    st.info("No PromptFoo result files found")
            
            with result_tabs[1]:  # RAG Query Results
                if result_categories["RAG Query Results"]:
                    for result_file in result_categories["RAG Query Results"]:
                        file_name = os.path.basename(result_file)
                        with open(result_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        st.subheader(f"RAG Results: {file_name}")
                        display_rag_results(data)
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

def display_rag_results(data):
    """Display formatted RAG query results with enhanced visualizations.
    
    Args:
        data: RAG query results data
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
                        display_single_rag_result(item, i+1)
                else:
                    st.info("No successful queries found")
            
            # Display failed queries
            with query_tabs[1]:
                if failed_queries:
                    for i, item in enumerate(failed_queries):
                        display_single_rag_result(item, i+1, is_error=True)
                else:
                    st.info("No failed queries found")
            
            # Display all queries
            with query_tabs[2]:
                for i, item in enumerate(data):
                    is_error = not item.get("success", True) or item.get("error")
                    display_single_rag_result(item, i+1, is_error=is_error)
        else:
            st.warning("No query results to display")
    else:
        # Display a single result
        is_error = not data.get("success", True) or data.get("error")
        display_single_rag_result(data, 1, is_error=is_error)

def display_single_rag_result(item, index, is_error=False):
    """Display a single RAG query result.
    
    Args:
        item: Query result data
        index: Query index number
        is_error: Whether this is an error result
    """
    query = item.get("query", "")
    query_prefix = query[:50] + ('...' if len(query) > 50 else '')
    
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
                        
                        # Show the actual text content
                        if "text" in chunk:
                            st.text_area(f"Content (Chunk {i+1})", chunk["text"], height=100, disabled=True, 
                                         label_visibility="collapsed", key=f"chunk_text_{index}_{i}")
                    else:
                        # If it's just a string, show it directly
                        st.text_area(f"Content (Chunk {i+1})", str(chunk), height=100, disabled=True, 
                                     label_visibility="collapsed", key=f"chunk_str_{index}_{i}")
                    
                    st.divider()
        
        # Display the response
        if "response" in item:
            st.markdown("### Response")
            
            # Calculate response metrics
            response = item["response"]
            response_length = len(response)
            
            # Show response length
            st.info(f"Response length: {response_length} characters")
            
            # Show the full response
            st.text_area("", response, height=200, disabled=True, 
                         label_visibility="collapsed", key=f"response_{index}")
        
        # Display additional metadata
        if "metadata" in item:
            with st.expander("Additional Metadata"):
                st.json(item["metadata"])

if __name__ == "__main__":
    # When run directly, just call the app function
    # This ensures it works in any Streamlit version
    run_app()