"""Textual-based TUI for Raggiro document processing."""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Input, Label, Select, Checkbox, DirectoryTree, Log
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive

from raggiro.processor import DocumentProcessor
from raggiro.utils.config import load_config

class RaggiroApp(App):
    """Textual app for Raggiro document processing."""
    
    CSS = """
    #app-grid {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr;
        height: 100%;
    }
    
    #input-panel {
        height: 100%;
        overflow-y: auto;
    }
    
    #output-panel {
        height: 100%;
    }
    
    .panel-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
        background: $primary;
        color: $text;
    }
    
    #log {
        height: 1fr;
        min-height: 10;
    }
    
    #input-path {
        width: 100%;
    }
    
    #output-path {
        width: 100%;
    }
    
    .form-row {
        height: auto;
        margin: 1 0;
    }
    """
    
    TITLE = "Raggiro - Document Processing for RAG"
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Container(id="app-grid"):
            # Input panel
            with Container(id="input-panel"):
                yield Label("Input Settings", classes="panel-title")
                
                with Vertical():
                    with Horizontal(classes="form-row"):
                        yield Label("Input Path:")
                        yield Input(placeholder="/path/to/documents", id="input-path")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("Output Path:")
                        yield Input(placeholder="/path/to/output", id="output-path")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("Process Recursively:")
                        yield Checkbox(value=True, id="recursive")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("Enable OCR:")
                        yield Checkbox(value=True, id="ocr")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("Output Format:")
                        yield Select(
                            [(format, format) for format in ["markdown", "json", "txt"]],
                            value="markdown",
                            id="formats",
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Label("Log Level:")
                        yield Select(
                            [(level, level) for level in ["debug", "info", "warning", "error"]],
                            value="info",
                            id="log-level",
                        )
                    
                    with Horizontal(classes="form-row"):
                        yield Button("Process Documents", id="process-button", variant="primary")
                
                # Directory tree for browsing
                yield Label("File Browser", classes="panel-title")
                yield DirectoryTree("/", id="directory-tree")
            
            # Output panel
            with Container(id="output-panel"):
                yield Label("Processing Output", classes="panel-title")
                yield Log(id="log")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.query_one("#directory-tree").path = os.getcwd()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button is pressed."""
        button_id = event.button.id
        
        if button_id == "process-button":
            self.process_documents()
    
    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Called when a file is selected in the directory tree."""
        # Update the input path
        self.query_one("#input-path").value = str(event.path)
    
    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Called when a directory is selected in the directory tree."""
        # Update the input path
        self.query_one("#input-path").value = str(event.path)
    
    def process_documents(self) -> None:
        """Process documents with the selected options."""
        # Get input values
        input_path = self.query_one("#input-path").value
        output_path = self.query_one("#output-path").value
        recursive = self.query_one("#recursive").value
        ocr_enabled = self.query_one("#ocr").value
        formats = self.query_one("#formats").value
        log_level = self.query_one("#log-level").value
        
        # Validate input
        if not input_path:
            self.log_error("Please enter an input path")
            return
        
        if not output_path:
            self.log_error("Please enter an output path")
            return
        
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
                "formats": formats.split(",") if "," in formats else [formats],
            },
        }
        
        # Log configuration
        self.log_info(f"Processing with configuration: {config}")
        
        # Create processor
        processor = DocumentProcessor(config=config)
        
        # Process documents
        self.log_info(f"Processing documents from {input_path} to {output_path}")
        
        # Check if input is a file or directory
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Process a single file
            self.log_info(f"Processing file: {input_path}")
            
            result = processor.process_file(input_path, output_path)
            
            if result["success"]:
                self.log_success(f"Successfully processed {input_path}")
            else:
                self.log_error(f"Failed to process {input_path}: {result.get('error', 'Unknown error')}")
        elif input_path_obj.is_dir():
            # Process a directory
            self.log_info(f"Processing directory: {input_path}")
            
            result = processor.process_directory(input_path, output_path, recursive=recursive)
            
            if result["success"]:
                summary = result["summary"]
                self.log_success(f"Processed {summary['total_files']} files")
                self.log_info(f"Successfully processed: {summary['successful_files']} files ({summary['success_rate']}%)")
                self.log_info(f"Failed: {summary['failed_files']} files")
                
                # Log individual file results
                for r in result["results"]:
                    if r["success"]:
                        self.log_success(f"✓ {os.path.basename(r['file_path'])}")
                    else:
                        self.log_error(f"✗ {os.path.basename(r['file_path'])}: {r.get('error', 'Unknown error')}")
            else:
                self.log_error(f"Failed to process directory: {result.get('error', 'Unknown error')}")
        else:
            self.log_error(f"Input path does not exist: {input_path}")
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        log = self.query_one("#log")
        log.write_line(f"[blue]INFO:[/] {message}")
    
    def log_success(self, message: str) -> None:
        """Log a success message."""
        log = self.query_one("#log")
        log.write_line(f"[green]SUCCESS:[/] {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        log = self.query_one("#log")
        log.write_line(f"[red]ERROR:[/] {message}")

def run_app():
    """Run the Textual app."""
    app = RaggiroApp()
    app.run()

if __name__ == "__main__":
    run_app()