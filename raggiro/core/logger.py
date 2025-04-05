"""Module for logging document processing events and results."""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

class DocumentLogger:
    """Logs document processing events and results."""
    
    def __init__(self, config: Optional[Dict] = None, log_dir: Optional[Union[str, Path]] = None):
        """Initialize the document logger.
        
        Args:
            config: Configuration dictionary
            log_dir: Directory to store logs
        """
        self.config = config or {}
        
        # Configure logging settings
        logging_config = self.config.get("logging", {})
        self.log_level = logging_config.get("log_level", "info").upper()
        self.log_to_file = logging_config.get("log_to_file", True)
        self.log_format = logging_config.get("log_format", "%(asctime)s - %(levelname)s - %(message)s")
        self.log_date_format = logging_config.get("log_date_format", "%Y-%m-%d %H:%M:%S")
        
        # Set up log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(logging_config.get("log_dir", "logs"))
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger("raggiro")
        self.logger.setLevel(getattr(logging, self.log_level))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(self.log_format, self.log_date_format))
        self.logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if self.log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"raggiro_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(self.log_format, self.log_date_format))
            self.logger.addHandler(file_handler)
        
        # Log files processed
        self.processed_files = []
        
        # Create CSV log file for processed files
        self.csv_log_file = self.log_dir / f"processed_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", 
                "filename", 
                "file_path", 
                "file_type", 
                "file_size", 
                "status", 
                "extraction_method",
                "word_count",
                "page_count",
                "error"
            ])
    
    def log_file_processing(self, document: Dict, status: str, error: Optional[str] = None) -> None:
        """Log file processing status and result.
        
        Args:
            document: Processed document dictionary
            status: Processing status (success/failure)
            error: Error message if any
        """
        metadata = document.get("metadata", {})
        file_metadata = metadata.get("file", {})
        
        # Log to general logger
        if status == "success":
            self.logger.info(f"Processed {file_metadata.get('path', 'unknown')} successfully")
        else:
            self.logger.error(f"Failed to process {file_metadata.get('path', 'unknown')}: {error}")
        
        # Add to processed files log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": file_metadata.get("filename", ""),
            "file_path": file_metadata.get("path", ""),
            "file_type": file_metadata.get("extension", ""),
            "file_size": file_metadata.get("size_bytes", 0),
            "status": status,
            "extraction_method": document.get("extraction_method", ""),
            "word_count": metadata.get("word_count", 0),
            "page_count": metadata.get("page_count", 0),
            "error": error or "",
        }
        
        self.processed_files.append(log_entry)
        
        # Write to CSV log
        with open(self.csv_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry["timestamp"],
                log_entry["filename"],
                log_entry["file_path"],
                log_entry["file_type"],
                log_entry["file_size"],
                log_entry["status"],
                log_entry["extraction_method"],
                log_entry["word_count"],
                log_entry["page_count"],
                log_entry["error"],
            ])
    
    def export_processing_summary(self) -> None:
        """Export a summary of all processed files."""
        if not self.processed_files:
            self.logger.warning("No files were processed")
            return
            
        # Calculate summary statistics
        total_files = len(self.processed_files)
        successful_files = sum(1 for entry in self.processed_files if entry["status"] == "success")
        failed_files = total_files - successful_files
        
        file_types = {}
        extraction_methods = {}
        errors = {}
        
        for entry in self.processed_files:
            # Count file types
            file_type = entry.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Count extraction methods
            if entry["status"] == "success":
                method = entry.get("extraction_method", "unknown")
                extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Count error types
            if entry["status"] != "success" and entry.get("error"):
                error = entry["error"]
                # Truncate error message for summary
                error = error[:50] + "..." if len(error) > 50 else error
                errors[error] = errors.get(error, 0) + 1
        
        # Create summary
        summary = {
            "start_time": self.processed_files[0]["timestamp"] if self.processed_files else "",
            "end_time": self.processed_files[-1]["timestamp"] if self.processed_files else "",
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": round(successful_files / total_files * 100, 2) if total_files > 0 else 0,
            "file_types": file_types,
            "extraction_methods": extraction_methods,
            "errors": errors,
        }
        
        # Export summary as JSON
        summary_file = self.log_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # Log summary
        self.logger.info(f"Processing summary: {successful_files}/{total_files} files processed successfully ({summary['success_rate']}%)")
        self.logger.info(f"Summary saved to {summary_file}")
    
    def log_info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message.
        
        Args:
            message: Message to log
        """
        self.logger.debug(message)