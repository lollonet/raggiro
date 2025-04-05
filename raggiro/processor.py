"""Main document processor module that orchestrates the document processing pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.cleaner import Cleaner
from raggiro.core.segmenter import Segmenter
from raggiro.core.metadata import MetadataExtractor
from raggiro.core.exporter import Exporter
from raggiro.core.logger import DocumentLogger
from raggiro.utils.config import load_config

class DocumentProcessor:
    """Main document processor class that orchestrates the document processing pipeline."""
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """Initialize the document processor.
        
        Args:
            config: Configuration dictionary (overrides config_path)
            config_path: Path to configuration file
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
        
        # Initialize components
        self.file_handler = FileHandler(self.config)
        self.extractor = Extractor(self.config)
        self.cleaner = Cleaner(self.config)
        self.segmenter = Segmenter(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.exporter = Exporter(self.config)
        self.logger = None  # Initialize later when we know the output directory
    
    def process_file(self, file_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """Process a single file.
        
        Args:
            file_path: Path to the file
            output_dir: Output directory
            
        Returns:
            Processing result
        """
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        # Ensure the file exists
        if not file_path.exists():
            return {
                "file_path": str(file_path),
                "success": False,
                "error": "File does not exist",
            }
        
        # Ensure the file is a file (not a directory)
        if not file_path.is_file():
            return {
                "file_path": str(file_path),
                "success": False,
                "error": "Not a file",
            }
        
        # Initialize logger if needed
        if self.logger is None:
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.logger = DocumentLogger(self.config, log_dir)
        
        # Process the file
        try:
            # Get file metadata
            file_metadata = self.file_handler.get_file_metadata(file_path)
            file_type_info = self.file_handler.detect_file_type(file_path)
            
            # Extract text
            document = self.extractor.extract(file_path, file_type_info)
            
            if not document["success"]:
                if self.logger:
                    self.logger.log_file_processing(document, "failure", document.get("error", "Extraction failed"))
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "error": document.get("error", "Extraction failed"),
                }
            
            # Clean text
            document = self.cleaner.clean_document(document)
            
            # Segment text
            document = self.segmenter.segment(document)
            
            # Extract metadata
            document["metadata"] = self.metadata_extractor.extract(document, file_metadata)
            
            # Export to output formats
            dry_run = self.config.get("processing", {}).get("dry_run", False)
            if not dry_run:
                result = self.exporter.export(document, output_dir)
                
                if not result["success"]:
                    if self.logger:
                        self.logger.log_file_processing(document, "failure", result.get("error", "Export failed"))
                    return {
                        "file_path": str(file_path),
                        "success": False,
                        "error": result.get("error", "Export failed"),
                    }
            
            # Log success
            if self.logger:
                self.logger.log_file_processing(document, "success")
            
            return {
                "file_path": str(file_path),
                "success": True,
                "document": document,
            }
            
        except Exception as e:
            if self.logger:
                document = {
                    "metadata": {
                        "file": self.file_handler.get_file_metadata(file_path)
                    },
                    "error": str(e)
                }
                self.logger.log_file_processing(document, "failure", str(e))
            
            return {
                "file_path": str(file_path),
                "success": False,
                "error": str(e),
            }
    
    def process_directory(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path], 
        recursive: bool = True,
    ) -> Dict:
        """Process all files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Whether to process subdirectories
            
        Returns:
            Processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Ensure the directory exists
        if not input_dir.exists():
            return {
                "input_dir": str(input_dir),
                "success": False,
                "error": "Directory does not exist",
            }
        
        # Ensure the input is a directory
        if not input_dir.is_dir():
            return {
                "input_dir": str(input_dir),
                "success": False,
                "error": "Not a directory",
            }
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = DocumentLogger(self.config, log_dir)
        
        # Get files to process
        files = self.file_handler.get_files(input_dir, recursive=recursive)
        
        if not files:
            return {
                "input_dir": str(input_dir),
                "success": False,
                "error": "No processable files found",
            }
        
        # Process each file
        results = []
        
        for file_path in files:
            result = self.process_file(file_path, output_dir)
            results.append(result)
        
        # Generate processing summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        summary = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "total_files": len(results),
            "successful_files": successful,
            "failed_files": failed,
            "success_rate": round(successful / len(results) * 100, 2) if results else 0,
        }
        
        # Export processing summary
        if self.logger:
            self.logger.export_processing_summary()
        
        return {
            "input_dir": str(input_dir),
            "success": True,
            "summary": summary,
            "results": results,
        }