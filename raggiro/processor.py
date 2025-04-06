"""Main document processor module that orchestrates the document processing pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.cleaner import Cleaner
from raggiro.core.spelling import SpellingCorrector
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
        self.spelling_corrector = SpellingCorrector(self.config)
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
        import time
        
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        # Total processing phases
        TOTAL_PHASES = 5  # Validation, Extraction, Cleaning, Segmentation, Export
        
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
        start_time = time.time()
        processing_times = {}
        
        try:
            # PHASE 1: File Validation and Metadata Extraction
            phase_start = time.time()
            self.logger.log_info(f"Starting processing of {file_path}")
            
            # Get file metadata
            file_metadata = self.file_handler.get_file_metadata(file_path)
            file_type_info = self.file_handler.detect_file_type(file_path)
            
            # Create initial document structure
            document = {
                "metadata": {"file": file_metadata},
                "file_type_info": file_type_info
            }
            
            phase_time = int((time.time() - phase_start) * 1000)
            processing_times["validation"] = phase_time
            
            if self.logger:
                self.logger.log_file_processing(
                    document, "success", 
                    component="file_handler", 
                    phase="File validation", 
                    phase_number=1, 
                    total_phases=TOTAL_PHASES,
                    processing_time_ms=phase_time
                )
            
            # PHASE 2: Text Extraction
            phase_start = time.time()
            document = self.extractor.extract(file_path, file_type_info)
            phase_time = int((time.time() - phase_start) * 1000)
            processing_times["extraction"] = phase_time
            
            if not document["success"]:
                if self.logger:
                    self.logger.log_file_processing(
                        document, "failure", 
                        component="extractor", 
                        phase="Text extraction", 
                        phase_number=2, 
                        total_phases=TOTAL_PHASES,
                        processing_time_ms=phase_time,
                        error=document.get("error", "Extraction failed")
                    )
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "error": document.get("error", "Extraction failed"),
                    "processing_times": processing_times
                }
            
            if self.logger:
                self.logger.log_file_processing(
                    document, "success", 
                    component="extractor", 
                    phase="Text extraction", 
                    phase_number=2, 
                    total_phases=TOTAL_PHASES,
                    processing_time_ms=phase_time
                )
            
            # PHASE 3: Text Cleaning
            phase_start = time.time()
            document = self.cleaner.clean_document(document)
            phase_time = int((time.time() - phase_start) * 1000)
            processing_times["cleaning"] = phase_time
            
            if self.logger:
                self.logger.log_file_processing(
                    document, "success", 
                    component="cleaner", 
                    phase="Text cleaning", 
                    phase_number=3, 
                    total_phases=TOTAL_PHASES,
                    processing_time_ms=phase_time
                )
                
            # PHASE 3.5: Spelling Correction (especially for OCR text)
            # Print debug info about extraction method and config
            extraction_method = document.get("extraction_method", "unknown")
            always_correct = self.config.get("spelling", {}).get("always_correct", False)
            print(f"Document extraction method: {extraction_method}")
            print(f"Always correct setting: {always_correct}")
            
            if extraction_method in ["pdf_ocr", "image_ocr"] or always_correct:
                print(f"Applying spelling correction to document...")
                spelling_start = time.time()
                document = self.spelling_corrector.correct_document(document)
                spelling_time = int((time.time() - spelling_start) * 1000)
                processing_times["spelling_correction"] = spelling_time
                
                if self.logger:
                    self.logger.log_file_processing(
                        document, "success", 
                        component="spelling_corrector", 
                        phase="Spelling correction", 
                        phase_number=3.5, 
                        total_phases=TOTAL_PHASES,
                        processing_time_ms=spelling_time
                    )
                print(f"Spelling correction completed in {spelling_time}ms")
            
            # PHASE 4: Text Segmentation
            phase_start = time.time()
            document = self.segmenter.segment(document)
            
            # Update with metadata
            document["metadata"] = self.metadata_extractor.extract(document, file_metadata)
            
            phase_time = int((time.time() - phase_start) * 1000)
            processing_times["segmentation"] = phase_time
            
            segment_stats = f"{len(document.get('segments', []))} segments, {len(document.get('chunks', []))} chunks"
            
            if self.logger:
                self.logger.log_file_processing(
                    document, "success", 
                    component="segmenter", 
                    phase=f"Text segmentation ({segment_stats})", 
                    phase_number=4, 
                    total_phases=TOTAL_PHASES,
                    processing_time_ms=phase_time
                )
            
            # PHASE 5: Export
            phase_start = time.time()
            dry_run = self.config.get("processing", {}).get("dry_run", False)
            
            if not dry_run:
                result = self.exporter.export(document, output_dir)
                
                phase_time = int((time.time() - phase_start) * 1000)
                processing_times["export"] = phase_time
                
                if not result["success"]:
                    if self.logger:
                        self.logger.log_file_processing(
                            document, "failure", 
                            component="exporter", 
                            phase="Export", 
                            phase_number=5, 
                            total_phases=TOTAL_PHASES,
                            processing_time_ms=phase_time,
                            error=result.get("error", "Export failed")
                        )
                    return {
                        "file_path": str(file_path),
                        "success": False,
                        "error": result.get("error", "Export failed"),
                        "processing_times": processing_times
                    }
                
                if self.logger:
                    export_formats = ", ".join(result.get("formats", {}).values())
                    self.logger.log_file_processing(
                        document, "success", 
                        component="exporter", 
                        phase=f"Export ({export_formats})", 
                        phase_number=5, 
                        total_phases=TOTAL_PHASES,
                        processing_time_ms=phase_time
                    )
            else:
                processing_times["export"] = 0
                if self.logger:
                    self.logger.log_file_processing(
                        document, "success", 
                        component="exporter", 
                        phase="Export (dry-run)", 
                        phase_number=5, 
                        total_phases=TOTAL_PHASES,
                        processing_time_ms=0
                    )
            
            # Final success log
            total_time = int((time.time() - start_time) * 1000)
            processing_times["total"] = total_time
            
            # Log complete process success
            if self.logger:
                word_count = document.get("metadata", {}).get("word_count", 0)
                self.logger.log_file_processing(
                    document, "success", 
                    component="processor", 
                    phase=f"Complete ({word_count} words)", 
                    processing_time_ms=total_time
                )
                
                # Log detailed metadata information
                self.logger.log_metadata(document)
            
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