#!/usr/bin/env python3
"""
Script to test Table of Contents detection with multiple languages.
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("toc_test")

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
from raggiro.core.file_handler import FileHandler
from raggiro.core.extractor import Extractor
from raggiro.core.segmenter import Segmenter

def test_toc_detection(pdf_path, language=None):
    """Test TOC detection on a specific PDF file."""
    pdf_name = Path(pdf_path).name
    logger.info(f"Testing TOC detection for: {pdf_name}")
    
    # Create minimal configuration
    config = {
        "extraction": {
            "ocr_enabled": False,  # Disable OCR since Tesseract is not available
            "ocr_language": language or "eng",
            "force_ocr": False
        },
        "segmentation": {
            "use_spacy": False,
            "max_chunk_size": 1000,
            "chunk_overlap": 200,
            "toc_detection": {
                "enabled": True,
                "min_entries": 3,
                "aggressiveness": 3,  # Use aggressive detection
                "max_entries": 150
            }
        }
    }
    
    # Initialize components
    file_handler = FileHandler(config)
    extractor = Extractor(config)
    segmenter = Segmenter(config)
    
    # Get document type information
    file_type_info = file_handler.detect_file_type(pdf_path)
    logger.info(f"Detected file type: {file_type_info.get('document_type')}")
    
    # Extract text from document
    try:
        document = extractor.extract(pdf_path, file_type_info)
        if not document["success"]:
            logger.error(f"Text extraction failed: {document.get('error')}")
            return None
            
        logger.info(f"Extraction successful using method: {document.get('extraction_method')}")
        
        # Segment the document and check for TOC
        segmented_doc = segmenter.segment(document)
        
        # Check if TOC was detected
        if "table_of_contents" in segmented_doc:
            toc_info = segmented_doc["table_of_contents"]
            toc_title = toc_info.get("title", "Unknown")
            toc_lang = toc_info.get("language", "unknown")
            entry_count = toc_info.get("entry_count", 0)
            
            logger.info(f"TOC detected: '{toc_title}' in language '{toc_lang}' with {entry_count} entries")
            
            # Print first 5 entries (if available)
            entries = toc_info.get("entries", [])
            if entries:
                logger.info("Sample TOC entries:")
                for i, entry in enumerate(entries[:5]):
                    title = entry.get("title", entry.get("text", "Unknown"))
                    page = entry.get("page", "?")
                    level = entry.get("level", 0)
                    logger.info(f"  {i+1}. {'  ' * level}'{title}' -> page {page}")
                
            return toc_info
        else:
            logger.warning("No table of contents detected in the document")
            return None
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

def main():
    """Test TOC detection on sample documents."""
    # Define the PDF documents to test
    pdf_directory = Path("/home/ubuntu/raggiro/tmp")
    
    # Map of files and their expected languages
    test_files = {
        "2020-Scrum-Guide-Italian.pdf": "it",
        "Canción de peregrino.pdf": "es",
        "Capitolato Tecnico e Allegati 1.pdf": "it",
        "Hornresp manual (1).pdf": "en",
        "Humanizar_it.pdf": "it",
        "Kenny_Werner_Effortless_Mastery_Liberati.pdf": "en",
        "PSN_Allegato Tecnico_v2.0.pdf": "it",
        "WEF_Future_of_Jobs_Report_2025.pdf": "en"
    }
    
    results = []
    
    # Process each test file
    for filename, lang in test_files.items():
        pdf_path = pdf_directory / filename
        if not pdf_path.exists():
            logger.warning(f"File not found: {filename}")
            continue
            
        logger.info(f"\n{'=' * 40}\nProcessing: {filename} (Expected language: {lang})\n{'=' * 40}")
        toc_result = test_toc_detection(pdf_path, lang)
        
        # Store results
        status = "✓ Detected" if toc_result else "✗ Not detected"
        detected_lang = toc_result.get("language", "N/A") if toc_result else "N/A"
        entry_count = toc_result.get("entry_count", 0) if toc_result else 0
        
        results.append({
            "filename": filename,
            "expected_language": lang,
            "status": status,
            "detected_language": detected_lang,
            "entry_count": entry_count
        })
    
    # Print summary
    logger.info("\n\n" + "=" * 80)
    logger.info("TOC DETECTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Filename':<40} {'Status':<15} {'Expected Lang':<15} {'Detected Lang':<15} {'Entries'}")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(f"{result['filename']:<40} {result['status']:<15} {result['expected_language']:<15} {result['detected_language']:<15} {result['entry_count']}")
    
    # Calculate success rate
    total = len(results)
    detected = sum(1 for r in results if "✓" in r["status"])
    success_rate = (detected / total * 100) if total > 0 else 0
    
    logger.info("-" * 80)
    logger.info(f"Success rate: {detected}/{total} ({success_rate:.1f}%)")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()