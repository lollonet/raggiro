"""Streamlit-based GUI for Raggiro document processing."""

import os
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import streamlit as st
from tqdm import tqdm

from raggiro.processor import DocumentProcessor
from raggiro.utils.config import load_config

def run_app():
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Raggiro - Document Processing for RAG",
        page_icon="📄",
        layout="wide",
    )
    
    st.title("Raggiro - Document Processing for RAG")
    st.write("Process documents for Retrieval-Augmented Generation (RAG) systems")
    
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["Process Documents", "Configuration", "About"])
    
    with tab1:
        process_documents_ui()
    
    with tab2:
        configuration_ui()
    
    with tab3:
        about_ui()

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
        ocr_enabled = st.checkbox("Enable OCR for scanned documents", value=True)
        
        # Log level
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
                ocr_enabled=ocr_enabled,
                log_level=log_level,
            )

def process_documents(
    input_type: str,
    uploaded_files: Optional[List] = None,
    input_path: Optional[str] = None,
    recursive: bool = False,
    output_formats: List[str] = ["markdown", "json"],
    output_path: Optional[str] = None,
    use_temp_dir: bool = True,
    ocr_enabled: bool = True,
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
        ocr_enabled: Whether to enable OCR
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
        
        # Display output directory
        st.subheader("Output Directory")
        st.code(output_dir)
        
        # Clean up temporary directory for uploaded files
        # Note: We don't clean up the output directory if it's a temporary directory,
        # so the user can access the results
        st.write("Note: Uploaded files will be deleted when you close the application")
        
        # Display detailed results
        if results:
            st.subheader("Detailed Results")
            
            for result in results:
                if result["success"]:
                    st.success(f"✅ {os.path.basename(result['file_path'])}")
                else:
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
        
        # Display detailed results
        if result["results"]:
            st.subheader("Detailed Results")
            
            for r in result["results"]:
                if r["success"]:
                    st.success(f"✅ {os.path.basename(r['file_path'])}")
                else:
                    st.error(f"❌ {os.path.basename(r['file_path'])}: {r.get('error', 'Unknown error')}")

def configuration_ui():
    """UI for configuration."""
    st.header("Configuration")
    
    # Load default configuration
    default_config = load_config()
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

def about_ui():
    """UI for about information."""
    st.header("About Raggiro")
    
    st.write("""
    Raggiro is an advanced document processing pipeline for RAG (Retrieval-Augmented Generation) applications.
    It processes documents from various formats, extracts text, cleans and segments it, and exports it in
    formats suitable for RAG systems.
    """)
    
    st.subheader("Features")
    
    st.markdown("""
    - **Comprehensive document support**: PDF (native and scanned), DOCX, TXT, HTML, RTF, XLSX, images with text
    - **Advanced preprocessing**: Extraction, cleaning, normalization and logical segmentation
    - **Metadata extraction**: Title, author, date, language, document type, etc.
    - **Structured output**: Markdown and JSON formats with all metadata
    - **Modular architecture**: CLI and minimal GUI interface
    - **Fully offline**: Works without external API dependencies
    - **RAG integration**: Ready for local semantic search pipelines
    """)
    
    st.subheader("Usage")
    
    st.write("""
    Raggiro can be used as a command-line tool, as a Python library, or through this GUI.
    See the README.md file for more information.
    """)

if __name__ == "__main__":
    run_app()