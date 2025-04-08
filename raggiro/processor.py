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
from raggiro.models.classifier import DocumentClassifier
from raggiro.pipelines import get_pipeline_for_category
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
        self.classifier = DocumentClassifier(self.config)
        self.extractor = Extractor(self.config)
        self.cleaner = Cleaner(self.config)
        self.spelling_corrector = SpellingCorrector(self.config)
        self.segmenter = Segmenter(self.config)
        self.metadata_extractor = MetadataExtractor(self.config)
        self.exporter = Exporter(self.config)
        self.logger = None  # Initialize later when we know the output directory
        
        # Configure document classification
        self.use_document_classification = self.config.get("classifier", {}).get("enabled", False)
        self.pipeline_config = self.config.get("pipeline", {})
        self.use_specialized_pipelines = self.pipeline_config.get("use_specialized_pipelines", False)
    
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
        TOTAL_PHASES = 6  # Validation, Classification, Extraction, Cleaning, Segmentation, Export
        
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
            
        # Step 0: Check if we should use specialized pipeline
        if self.use_document_classification and self.use_specialized_pipelines:
            # Initialize basic validation
            file_metadata = self.file_handler.get_file_metadata(file_path)
            file_type_info = self.file_handler.detect_file_type(file_path)
            
            # Perform initial classification based on file metadata
            classification_result = self.classifier.classify_from_metadata(file_metadata, file_type_info)
            
            if classification_result["success"] and classification_result["category"] != "unknown":
                document_category = classification_result["category"]
                # Try to get specialized pipeline
                specialized_pipeline = get_pipeline_for_category(document_category, self.config)
                
                if specialized_pipeline:
                    self.logger.log_info(f"Using specialized pipeline for {document_category} document: {file_path}")
                    # Use specialized pipeline for this document type
                    return specialized_pipeline.process(file_path, output_dir)
        
        # Process the file with standard pipeline
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
                
            # PHASE 1.5: Document Classification (if enabled)
            document_category = None
            if self.use_document_classification:
                phase_start = time.time()
                
                # Perform initial classification based on file metadata
                classification_result = self.classifier.classify_from_metadata(file_metadata, file_type_info)
                
                if classification_result["success"]:
                    document_category = classification_result["category"]
                    document["classification"] = classification_result
                    
                    phase_time = int((time.time() - phase_start) * 1000)
                    processing_times["classification"] = phase_time
                    
                    if self.logger:
                        self.logger.log_file_processing(
                            document, "success", 
                            component="classifier", 
                            phase=f"Document classification: {document_category}", 
                            phase_number=1.5, 
                            total_phases=TOTAL_PHASES,
                            processing_time_ms=phase_time
                        )
                else:
                    # Classification failed, log but continue processing
                    if self.logger:
                        self.logger.log_file_processing(
                            document, "warning", 
                            component="classifier", 
                            phase="Document classification failed", 
                            phase_number=1.5, 
                            total_phases=TOTAL_PHASES,
                            processing_time_ms=int((time.time() - phase_start) * 1000),
                            error=classification_result.get("error", "Unknown classification error")
                        )
            
            # PHASE 2: Text Extraction
            phase_start = time.time()
            
            # Use specialized extraction based on document category if available
            if self.use_specialized_pipelines and document_category:
                extraction_config = self.config.get("extraction", {}).get(document_category, {})
                if extraction_config:
                    # Merge specialized extraction config with general config
                    specialized_config = self.config.copy()
                    specialized_config["extraction"] = {**self.config.get("extraction", {}), **extraction_config}
                    specialized_extractor = Extractor(specialized_config)
                    document = specialized_extractor.extract(file_path, file_type_info)
                else:
                    # Use default extractor if no specialized config exists
                    document = self.extractor.extract(file_path, file_type_info)
            else:
                # Use default extractor
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
            spelling_config = self.config.get("spelling", {})
            always_correct = spelling_config.get("always_correct", False)
            spelling_enabled = spelling_config.get("enabled", True)
            backend = spelling_config.get("backend", "symspellpy")
            
            print(f"Document extraction method: {extraction_method}")
            print(f"Spelling config - enabled: {spelling_enabled}, always_correct: {always_correct}, backend: {backend}")
            
            # Force spelling correction regardless of settings for testing
            always_apply_correction = True
            if extraction_method in ["pdf_ocr", "image_ocr"] or always_correct or always_apply_correction:
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
            
            # Refine document classification based on content if available
            if self.use_document_classification and "text" in document:
                content_classification = self.classifier.classify(document["text"])
                
                # Update classification if content-based is successful
                if content_classification["success"]:
                    # If we already have a classification, use a weighted approach
                    if "classification" in document and document["classification"]["success"]:
                        metadata_class = document["classification"]["category"]
                        metadata_conf = document["classification"]["confidence"]
                        content_class = content_classification["category"]
                        content_conf = content_classification["confidence"]
                        
                        # Keep the higher confidence classification or combine them
                        if content_class == metadata_class:
                            # Same category, average confidence
                            document["classification"]["confidence"] = (metadata_conf + content_conf) / 2
                            document["classification"]["method"] = "combined"
                        elif content_conf > metadata_conf:
                            # Content-based has higher confidence
                            document["classification"] = content_classification
                        # Otherwise keep the metadata classification
                    else:
                        # No existing classification, use the content-based one
                        document["classification"] = content_classification
                        
                # Update document category for specialized processing
                if "classification" in document and document["classification"]["success"]:
                    document_category = document["classification"]["category"]
            
            # Use specialized segmentation based on document category if available
            if self.use_specialized_pipelines and document_category and document_category != "unknown":
                segmentation_config = self.config.get("segmentation", {}).get(document_category, {})
                if segmentation_config:
                    # Merge specialized segmentation config with general config
                    specialized_config = self.config.copy()
                    specialized_config["segmentation"] = {**self.config.get("segmentation", {}), **segmentation_config}
                    specialized_segmenter = Segmenter(specialized_config)
                    document = specialized_segmenter.segment(document)
                else:
                    # Use default segmenter if no specialized config exists
                    document = self.segmenter.segment(document)
            else:
                # Use default segmenter
                document = self.segmenter.segment(document)
            
            # Update with metadata
            document["metadata"] = self.metadata_extractor.extract(document, file_metadata)
            
            # Add classification information to metadata
            if "classification" in document and document["classification"]["success"]:
                document["metadata"]["document_category"] = document["classification"]["category"]
                document["metadata"]["classification_confidence"] = document["classification"]["confidence"]
                document["metadata"]["classification_method"] = document["classification"]["method"]
            
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