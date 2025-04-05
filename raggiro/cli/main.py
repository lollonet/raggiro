"""Command-line interface for Raggiro document processing."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
from tqdm import tqdm

from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.cleaner import Cleaner
from raggiro.core.segmenter import Segmenter
from raggiro.core.metadata import MetadataExtractor
from raggiro.core.exporter import Exporter
from raggiro.core.logger import DocumentLogger
from raggiro.utils.config import load_config

@click.group()
@click.version_option()
def main():
    """Raggiro - Advanced document processing pipeline for RAG applications."""
    pass

@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for processed files."
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to configuration file."
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Process directories recursively."
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Run without writing any files."
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging level."
)
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "json", "txt"], case_sensitive=False),
    multiple=True,
    default=["markdown", "json"],
    help="Output formats (can be specified multiple times)."
)
@click.option(
    "--ocr/--no-ocr",
    default=True,
    help="Enable OCR for scanned documents and images."
)
def process(
    input_path: str,
    output: Optional[str] = None,
    config: Optional[str] = None,
    recursive: bool = True,
    dry_run: bool = False,
    log_level: str = "info",
    format: Tuple[str] = ("markdown", "json"),
    ocr: bool = True,
):
    """Process documents from INPUT_PATH.
    
    INPUT_PATH can be a file or directory.
    """
    # Load configuration
    cfg = load_config(config)
    
    # Override config with command-line options
    if cfg is None:
        cfg = {}
    
    cfg["processing"] = cfg.get("processing", {})
    cfg["processing"]["dry_run"] = dry_run
    
    cfg["logging"] = cfg.get("logging", {})
    cfg["logging"]["log_level"] = log_level
    
    cfg["export"] = cfg.get("export", {})
    cfg["export"]["formats"] = list(format)
    
    cfg["extraction"] = cfg.get("extraction", {})
    cfg["extraction"]["ocr_enabled"] = ocr
    
    # Set up output directory
    if output:
        output_dir = Path(output)
    else:
        output_dir = Path("output")
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging directory
    log_dir = output_dir / "logs"
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    file_handler = FileHandler(cfg)
    extractor = Extractor(cfg)
    cleaner = Cleaner(cfg)
    segmenter = Segmenter(cfg)
    metadata_extractor = MetadataExtractor(cfg)
    exporter = Exporter(cfg)
    logger = DocumentLogger(cfg, log_dir)
    
    logger.log_info(f"Starting processing of {input_path}")
    
    # Get files to process
    input_path_obj = Path(input_path)
    files = file_handler.get_files(input_path_obj, recursive=recursive)
    
    if not files:
        logger.log_error(f"No processable files found in {input_path}")
        sys.exit(1)
    
    logger.log_info(f"Found {len(files)} files to process")
    
    # Process files
    for file_path in tqdm(files, desc="Processing files", unit="file"):
        try:
            logger.log_info(f"Processing {file_path}")
            
            # Get file metadata
            file_metadata = file_handler.get_file_metadata(file_path)
            file_type_info = file_handler.detect_file_type(file_path)
            
            # Extract text
            document = extractor.extract(file_path, file_type_info)
            
            if not document["success"]:
                logger.log_error(f"Extraction failed for {file_path}: {document.get('error', 'Unknown error')}")
                logger.log_file_processing(document, "failure", document.get("error", "Extraction failed"))
                continue
            
            # Clean text
            document = cleaner.clean_document(document)
            
            # Segment text
            document = segmenter.segment(document)
            
            # Extract metadata
            document["metadata"] = metadata_extractor.extract(document, file_metadata)
            
            # Export to output formats
            if not dry_run:
                result = exporter.export(document, output_dir)
                
                if not result["success"]:
                    logger.log_error(f"Export failed for {file_path}: {result.get('error', 'Unknown error')}")
                else:
                    logger.log_info(f"Exported {file_path} to: {', '.join(result['formats'].values())}")
            
            # Log success
            logger.log_file_processing(document, "success")
            
        except Exception as e:
            logger.log_error(f"Error processing {file_path}: {str(e)}")
            document = {
                "metadata": {
                    "file": file_handler.get_file_metadata(file_path)
                },
                "error": str(e)
            }
            logger.log_file_processing(document, "failure", str(e))
    
    # Export processing summary
    if not dry_run:
        logger.export_processing_summary()
    
    logger.log_info("Processing complete")

@main.command()
@click.option(
    "--tui", 
    is_flag=True,
    help="Use text-based UI instead of web-based Streamlit UI."
)
def gui(tui: bool = False):
    """Launch the GUI interface."""
    # Import here to avoid dependencies if only using CLI
    try:
        if tui:
            from raggiro.gui.textual_app import run_app
            run_app()
        else:
            from raggiro.gui.streamlit_app import run_app
            run_app()
    except ImportError:
        click.echo("Error: GUI dependencies not installed. Install with: pip install raggiro[gui]")
        sys.exit(1)

@main.command()
@click.option(
    "--prompt-set",
    type=click.Path(exists=True),
    required=True,
    help="Path to promptfoo prompt set configuration."
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for test results."
)
def test_rag(prompt_set: str, output: Optional[str] = None):
    """Test RAG pipeline using promptfoo."""
    # Import here to avoid dependencies if not testing
    try:
        from raggiro.testing.promptfoo_runner import run_tests
        
        if output:
            output_dir = Path(output)
        else:
            output_dir = Path("test_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_tests(prompt_set, output_dir)
    except ImportError:
        click.echo("Error: Testing dependencies not installed. Install with: pip install raggiro[test]")
        sys.exit(1)

if __name__ == "__main__":
    main()