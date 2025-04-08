"""Specialized pipeline for technical documents."""

from typing import Dict, List, Optional, Union
from pathlib import Path

from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.cleaner import Cleaner
from raggiro.core.spelling import SpellingCorrector
from raggiro.core.segmenter import Segmenter
from raggiro.core.metadata import MetadataExtractor
from raggiro.core.exporter import Exporter
from raggiro.core.logger import DocumentLogger

class TechnicalPipeline:
    """Specialized pipeline for processing technical documents."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the technical document pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Override config with technical-specific settings
        technical_config = self._create_technical_config(self.config)
        
        # Initialize components with specialized config
        self.file_handler = FileHandler(technical_config)
        self.extractor = Extractor(technical_config)
        self.cleaner = Cleaner(technical_config)
        self.spelling_corrector = SpellingCorrector(technical_config)
        self.segmenter = Segmenter(technical_config)
        self.metadata_extractor = MetadataExtractor(technical_config)
        self.exporter = Exporter(technical_config)
        self.logger = None
    
    def _create_technical_config(self, base_config: Dict) -> Dict:
        """Create specialized config for technical documents.
        
        Args:
            base_config: Base configuration dictionary
            
        Returns:
            Specialized configuration dictionary
        """
        # Create a copy of the base config
        config = base_config.copy()
        
        # Apply specialized settings for technical documents
        # These overrides can come from config or be hardcoded defaults
        
        # Extraction settings
        extraction_config = config.get("extraction", {}).copy()
        extraction_config.update({
            "extract_tables": True,
            "extract_images": True,
            "extract_captions": True,
            "extract_equations": True,
            "extract_code_blocks": True,
        })
        config["extraction"] = extraction_config
        
        # Segmentation settings
        segmentation_config = config.get("segmentation", {}).copy()
        segmentation_config.update({
            "detect_code_blocks": True,
            "detect_equations": True,
            "detect_diagrams": True,
            "preserve_tables": True,
            "keep_tables_together": True,
            "max_chunk_size": 2000,  # Larger chunks for technical content
        })
        config["segmentation"] = segmentation_config
        
        # Cleaning settings - less aggressive for technical documents
        cleaning_config = config.get("cleaning", {}).copy()
        cleaning_config.update({
            "preserve_code_blocks": True,
            "preserve_equations": True,
            "preserve_diagrams": True,
            "aggressive_cleaning": False,
        })
        config["cleaning"] = cleaning_config
        
        return config
        
    def process(self, file_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """Process a technical document.
        
        Args:
            file_path: Path to the file
            output_dir: Output directory
            
        Returns:
            Processing result with technical metadata
        """
        import time
        
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        # Initialize logger if needed
        if self.logger is None:
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.logger = DocumentLogger(self.config, log_dir)
        
        # Process the file
        start_time = time.time()
        
        try:
            # Get file metadata
            file_metadata = self.file_handler.get_file_metadata(file_path)
            file_type_info = self.file_handler.detect_file_type(file_path)
            
            # Create initial document structure
            document = {
                "metadata": {"file": file_metadata},
                "file_type_info": file_type_info,
                "document_category": "technical",
                "pipeline_type": "technical"
            }
            
            # Extract text with technical-specific options
            document = self.extractor.extract(file_path, file_type_info)
            if not document["success"]:
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "error": document.get("error", "Extraction failed"),
                }
            
            # Technical-specific cleaning
            document = self.cleaner.clean_document(document)
            
            # Apply spelling correction if needed
            document = self.spelling_corrector.correct_document(document)
            
            # Segment the document with technical-specific settings
            document = self.segmenter.segment(document)
            
            # Extract technical metadata
            document["metadata"] = self.metadata_extractor.extract(document, file_metadata)
            
            # Add technical-specific metadata
            document["metadata"]["document_category"] = "technical"
            document["metadata"]["pipeline_type"] = "technical"
            
            # Add technical features detection
            document["technical_features"] = self._detect_technical_features(document)
            
            # Export the document
            result = self.exporter.export(document, output_dir)
            
            if not result["success"]:
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "error": result.get("error", "Export failed"),
                }
            
            # Add pipeline info to result
            result["pipeline_type"] = "technical"
            result["document_category"] = "technical"
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.log_file_processing(
                    {"metadata": {"file": file_metadata}} if 'file_metadata' in locals() else {},
                    "failure",
                    error=str(e)
                )
            
            return {
                "file_path": str(file_path),
                "success": False,
                "error": str(e),
            }
    
    def _detect_technical_features(self, document: Dict) -> Dict:
        """Detect technical features in the document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Dictionary with technical features
        """
        # Initialize features dictionary
        features = {
            "has_code_blocks": False,
            "has_equations": False,
            "has_diagrams": False,
            "has_tables": False,
            "has_technical_terms": False,
            "complexity_level": "unknown",
        }
        
        # Check for code blocks
        if "segments" in document:
            for segment in document["segments"]:
                if segment.get("type") == "code_block":
                    features["has_code_blocks"] = True
                if segment.get("type") == "equation":
                    features["has_equations"] = True
                if segment.get("type") == "diagram":
                    features["has_diagrams"] = True
                if segment.get("type") == "table":
                    features["has_tables"] = True
        
        # Check for technical terms using a simple keyword-based approach
        if "text" in document:
            technical_terms = [
                "algorithm", "function", "parameter", "component", "interface",
                "specification", "configuration", "implementation", "module", "version",
                "biblioteca", "algoritmo", "funzione", "parametro", "componente",  # Italian terms
                "interfaccia", "specifica", "configurazione", "implementazione", "modulo"
            ]
            
            text_lower = document["text"].lower()
            features["technical_term_count"] = 0
            
            for term in technical_terms:
                if term in text_lower:
                    features["has_technical_terms"] = True
                    # Count occurrences
                    features["technical_term_count"] += text_lower.count(term)
            
            # Estimate complexity level based on term density
            if features["technical_term_count"] > 0:
                word_count = len(document["text"].split())
                term_density = features["technical_term_count"] / word_count
                
                if term_density > 0.05:
                    features["complexity_level"] = "high"
                elif term_density > 0.02:
                    features["complexity_level"] = "medium"
                else:
                    features["complexity_level"] = "low"
        
        return features