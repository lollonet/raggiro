"""Module for exporting processed documents in various formats."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# For PDF creation
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

class Exporter:
    """Exports processed documents in various formats."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the exporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure export settings
        export_config = self.config.get("export", {})
        self.formats = export_config.get("formats", ["markdown", "json"])
        self.include_metadata = export_config.get("include_metadata", True)
        self.pretty_json = export_config.get("pretty_json", True)
        self.json_indent = 2 if self.pretty_json else None
        
        # Configure PDF output settings
        pdf_config = self.config.get("pdf_output", {})
        self.save_corrected_pdf = pdf_config.get("save_corrected_pdf", True)
        self.generate_comparison = pdf_config.get("generate_comparison", True)
    
    def export(self, document: Dict, output_dir: Union[str, Path]) -> Dict:
        """Export a processed document to the specified formats.
        
        Args:
            document: Processed document dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary with export status and paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "filename": document.get("metadata", {}).get("filename", "unknown"),
            "formats": {},
            "success": False,
            "error": None,
        }
        
        try:
            # Create a base filename
            base_filename = self._create_filename(document)
            
            # Export in each format
            for format_name in self.formats:
                if format_name == "markdown":
                    output_path = output_dir / f"{base_filename}.md"
                    self._export_markdown(document, output_path)
                    result["formats"]["markdown"] = str(output_path)
                    
                elif format_name == "json":
                    output_path = output_dir / f"{base_filename}.json"
                    self._export_json(document, output_path)
                    result["formats"]["json"] = str(output_path)
                    
                elif format_name == "txt":
                    output_path = output_dir / f"{base_filename}.txt"
                    self._export_text(document, output_path)
                    result["formats"]["txt"] = str(output_path)
                    
                elif format_name == "pdf" and HAS_FITZ:
                    # Check if this is an OCR document with spelling correction
                    has_spell_correction = document.get("metadata", {}).get("spelling_corrected", False)
                    is_ocr_document = document.get("extraction_method", "") in ["pdf_ocr", "image_ocr"]
                    
                    # If this is a corrected document, generate corrected PDF if configured
                    if (is_ocr_document or has_spell_correction) and self.save_corrected_pdf:
                        try:
                            corrected_pdf_path = output_dir / f"{base_filename}_corrected.pdf"
                            self._export_corrected_pdf(document, corrected_pdf_path)
                            result["formats"]["corrected_pdf"] = str(corrected_pdf_path)
                        except Exception as e:
                            print(f"Warning: Failed to create corrected PDF: {str(e)}")
                    
                    # Generate side-by-side comparison if configured
                    if (is_ocr_document or has_spell_correction) and self.generate_comparison:
                        try:
                            comparison_pdf_path = output_dir / f"{base_filename}_comparison.pdf"
                            self._export_comparison_pdf(document, comparison_pdf_path)
                            result["formats"]["comparison_pdf"] = str(comparison_pdf_path)
                        except Exception as e:
                            print(f"Warning: Failed to create comparison PDF: {str(e)}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _create_filename(self, document: Dict) -> str:
        """Create a clean filename for the exported document.
        
        Args:
            document: Processed document dictionary
            
        Returns:
            Base filename (without extension)
        """
        metadata = document.get("metadata", {})
        
        # Try to use title if available
        if "title" in metadata and metadata["title"]:
            # Clean the title for use as a filename
            title = metadata["title"]
            filename = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
            filename = filename.strip()
            
            # If title is too long, truncate it
            if len(filename) > 100:
                filename = filename[:100]
        else:
            # Fallback to original filename
            original_filename = metadata.get("filename", "document")
            # Remove extension if present
            filename = os.path.splitext(original_filename)[0]
        
        # Add date if available
        if "date" in metadata and metadata["date"]:
            date_str = metadata["date"].replace("-", "")
            filename = f"{date_str}_{filename}"
        
        # Make sure the filename is not empty and doesn't start with a dot
        if not filename or filename.startswith("."):
            filename = f"document_{datetime.now().strftime('%Y%m%d')}"
            
        # Replace spaces with underscores and ensure no consecutive underscores
        filename = filename.replace(" ", "_")
        while "__" in filename:
            filename = filename.replace("__", "_")
            
        return filename
    
    def _export_markdown(self, document: Dict, output_path: Path) -> None:
        """Export document as Markdown.
        
        Args:
            document: Processed document dictionary
            output_path: Output file path
        """
        metadata = document.get("metadata", {})
        content = []
        
        # Add YAML frontmatter with metadata
        if self.include_metadata:
            content.append("---")
            
            if "title" in metadata:
                content.append(f"title: {metadata['title']}")
                
            if "author" in metadata:
                content.append(f"author: {metadata['author']}")
                
            if "date" in metadata:
                content.append(f"date: {metadata['date']}")
                
            if "language" in metadata:
                content.append(f"language: {metadata['language']}")
                
            if "topics" in metadata and metadata["topics"]:
                topics_str = ", ".join(metadata["topics"])
                content.append(f"topics: {topics_str}")
                
            if "file" in metadata:
                content.append(f"source: {metadata.get('file', {}).get('path', '')}")
                
            content.append("---")
            content.append("")
        
        # Add title
        if "title" in metadata and metadata["title"]:
            content.append(f"# {metadata['title']}")
            content.append("")
        
        # Add metadata section
        if self.include_metadata:
            content.append("## Metadata")
            content.append("")
            
            if "author" in metadata and metadata["author"]:
                content.append(f"**Author:** {metadata['author']}")
                
            if "date" in metadata and metadata["date"]:
                content.append(f"**Date:** {metadata['date']}")
                
            if "language" in metadata and metadata["language"]:
                content.append(f"**Language:** {metadata['language']}")
                
            if "topics" in metadata and metadata["topics"]:
                content.append(f"**Topics:** {', '.join(metadata['topics'])}")
                
            if "word_count" in metadata:
                content.append(f"**Word count:** {metadata['word_count']}")
                
            if "page_count" in metadata:
                content.append(f"**Page count:** {metadata['page_count']}")
                
            content.append("")
            content.append("---")
            content.append("")
        
        # Add content sections
        if "chunks" in document and document["chunks"]:
            for chunk in document["chunks"]:
                # Find headers in the chunk
                headers = [seg for seg in chunk.get("segments", []) if seg.get("type") == "header"]
                
                if headers:
                    # Use the headers to structure the content
                    for header in headers:
                        level = header.get("level", 2)
                        header_markdown = "#" * min(level + 1, 6)  # Ensure header level is valid
                        content.append(f"{header_markdown} {header.get('text', 'Section')}")
                        content.append("")
                else:
                    # If no headers, use a generic section title
                    content.append(f"## Content (Chunk {chunk.get('id', '')})")
                    content.append("")
                
                # Add the chunk text
                content.append(chunk.get("text", ""))
                content.append("")
                content.append("---")
                content.append("")
        else:
            # If no chunks, add the full text
            content.append("## Content")
            content.append("")
            content.append(document.get("text", ""))
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
    
    def _export_json(self, document: Dict, output_path: Path) -> None:
        """Export document as JSON.
        
        Args:
            document: Processed document dictionary
            output_path: Output file path
        """
        # Create a clean JSON export with basic fields
        export_data = {
            "metadata": document.get("metadata", {}),
            "text": document.get("text", ""),
            "extraction_method": document.get("extraction_method", ""),
        }
        
        # Ensure original text is available for comparisons
        # Use a consistent naming convention: original_text instead of raw_text
        if "original_text" in document:
            export_data["original_text"] = document["original_text"]
        elif "raw_text" in document:
            export_data["original_text"] = document["raw_text"]
            # Also keep raw_text for backward compatibility
            export_data["raw_text"] = document["raw_text"]
            
        # Log what we're storing
        keys_with_text = [k for k in document.keys() if "text" in k]
        print(f"Document text fields: {keys_with_text}")
        print(f"Exporting text fields: {[k for k in export_data.keys() if 'text' in k]}")
        
        # Add page-level original and raw text if available
        if "pages" in document:
            export_data["pages"] = []
            for page_idx, page in enumerate(document["pages"]):
                # Basic page data
                page_data = {
                    "page_num": page.get("page_num", page_idx + 1),
                    "text": page.get("text", "")
                }
                
                # Ensure both original_text and raw_text are available
                # This ensures maximum compatibility
                if "original_text" in page:
                    page_data["original_text"] = page["original_text"]
                if "raw_text" in page:
                    # Keep raw_text but ensure original_text also exists
                    page_data["raw_text"] = page["raw_text"]
                    if "original_text" not in page_data:
                        page_data["original_text"] = page["raw_text"]
                
                # Include other useful page metadata
                for key in ["has_text", "char_count", "processing_time"]:
                    if key in page:
                        page_data[key] = page[key]
                        
                export_data["pages"].append(page_data)
                
            # Log what fields we have in the first page
            if export_data["pages"]:
                print(f"First page data fields: {list(export_data['pages'][0].keys())}")
                
        # Add original pages if they exist
        if "original_pages" in document:
            export_data["original_pages"] = document["original_pages"]
        
        # Add segments and chunks if available
        if "segments" in document:
            export_data["segments"] = document["segments"]
            
        if "chunks" in document:
            export_data["chunks"] = document["chunks"]
        
        # Write to file with pretty printing
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=self.json_indent, ensure_ascii=False)
    
    def _export_text(self, document: Dict, output_path: Path) -> None:
        """Export document as plain text.
        
        Args:
            document: Processed document dictionary
            output_path: Output file path
        """
        content = []
        
        # Add a simple header
        metadata = document.get("metadata", {})
        if "title" in metadata and metadata["title"]:
            content.append(metadata["title"].upper())
            content.append("=" * len(metadata["title"]))
            content.append("")
        
        # Add basic metadata
        if self.include_metadata:
            if "author" in metadata and metadata["author"]:
                content.append(f"Author: {metadata['author']}")
                
            if "date" in metadata and metadata["date"]:
                content.append(f"Date: {metadata['date']}")
                
            if "topics" in metadata and metadata["topics"]:
                content.append(f"Topics: {', '.join(metadata['topics'])}")
                
            content.append("")
            content.append("-" * 80)
            content.append("")
        
        # Add the text content
        content.append(document.get("text", ""))
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
            
    def _export_corrected_pdf(self, document: Dict, output_path: Path) -> None:
        """Export a PDF with corrected text.
        
        Args:
            document: Processed document dictionary
            output_path: Output file path
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required to create PDF files")
            
        # List of built-in PDF fonts we can use safely
        BUILTIN_FONTS = ["helv", "times", "courier", "symbol", "zapf"]
        # Default to Helvetica for all text
        DEFAULT_FONT = "helv"
        
        # Set text size limits to avoid overflows
        MAX_TEXT_SIZE = 100000  # Maximum character count for a single textbox
        MAX_PAGE_LENGTH = 5000  # Maximum length of text per page
        
        # Create a new PDF document
        doc = fitz.open()
        
        # Get document metadata
        metadata = document.get("metadata", {})
        
        # Set document metadata
        doc.set_metadata({
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": "Corrected document from OCR",
            "creator": "Raggiro OCR Correction",
            "producer": "Raggiro",
            "creationDate": fitz.get_pdf_now(),
        })
        
        # Process each page separately if available
        pages = document.get("pages", [])
        if pages:
            for i, page_data in enumerate(pages):
                page_text = page_data.get("text", "")
                if page_text:
                    # Create a new page in the PDF
                    page = doc.new_page(width=595, height=842)  # A4 size
                    
                    # Add a header to indicate this is a corrected version with char count
                    char_count = page_data.get("char_count", len(page_text))
                    header_text = f"Corrected Text - Page {i+1}/{len(pages)} ({char_count} chars)"
                    # Use 'helv' (Helvetica) font which is built-in to PDF
                    page.insert_text((50, 50), header_text, fontsize=12, fontname="helv", color=(0, 0, 0.8))
                    
                    # Add a divider line
                    page.draw_line((50, 70), (545, 70), color=(0, 0, 0.5), width=1)
                    
                    # Add the text content
                    rect = fitz.Rect(50, 80, 545, 792)  # Leave margins
                    try:
                        # Truncate text if it's too long to avoid memory issues
                        if len(page_text) > MAX_TEXT_SIZE:
                            truncated_text = page_text[:MAX_TEXT_SIZE] + "\n\n[... Text truncated due to size limits. See JSON for full content ...]"
                            print(f"Warning: Truncating page {i+1} text from {len(page_text)} to {len(truncated_text)} characters")
                            page_text = truncated_text
                            
                        page.insert_textbox(rect, page_text, fontsize=10, fontname="helv",
                                          align=0, color=(0, 0, 0))
                    except Exception as e:
                        # If textbox fails, try simpler text insertion with fallback
                        print(f"Warning: Failed to insert text box, using simpler method: {str(e)}")
                        page.insert_text((50, 100), "Error rendering text. See JSON output for content.", 
                                       fontsize=10, fontname="helv", color=(0.8, 0, 0))
        else:
            # If no individual pages, use the full text
            full_text = document.get("text", "")
            if full_text:
                # Create a new page in the PDF
                page = doc.new_page(width=595, height=842)  # A4 size
                
                # Add a header to indicate this is a corrected version
                header_text = "Corrected Text"
                page.insert_text((50, 50), header_text, fontsize=12, fontname="helv", color=(0, 0, 0.8))
                
                # Add a divider line
                page.draw_line((50, 70), (545, 70), color=(0, 0, 0.5), width=1)
                
                # Add the text content
                rect = fitz.Rect(50, 80, 545, 792)  # Leave margins
                try:
                    # Split into multiple pages if text is very long
                    if len(full_text) > MAX_PAGE_LENGTH:
                        # Create a multi-page document
                        current_text = full_text
                        page_number = 1
                        
                        while current_text:
                            # Take a chunk of text for this page
                            page_chunk = current_text[:MAX_PAGE_LENGTH]
                            current_text = current_text[MAX_PAGE_LENGTH:]
                            
                            if page_number > 1:
                                # Create a new page for the next chunk
                                page = doc.new_page(width=595, height=842)  # A4 size
                                header_text = f"Corrected Text - Page {page_number}"
                                page.insert_text((50, 50), header_text, fontsize=12, fontname="helv", color=(0, 0, 0.8))
                                page.draw_line((50, 70), (545, 70), color=(0, 0, 0.5), width=1)
                                
                            # Insert this chunk
                            page.insert_textbox(rect, page_chunk, fontsize=10, fontname="helv",
                                              align=0, color=(0, 0, 0))
                            
                            page_number += 1
                    else:
                        # Short enough for a single page
                        page.insert_textbox(rect, full_text, fontsize=10, fontname="helv",
                                          align=0, color=(0, 0, 0))
                except Exception as e:
                    # If textbox fails, try simpler text insertion with fallback
                    print(f"Warning: Failed to insert text box, using simpler method: {str(e)}")
                    page.insert_text((50, 100), "Error rendering text. See JSON output for content.", 
                                   fontsize=10, fontname="helv", color=(0.8, 0, 0))
        
        # Save the PDF
        doc.save(output_path)
        doc.close()
        
    def _export_comparison_pdf(self, document: Dict, output_path: Path) -> None:
        """Export a PDF with side-by-side comparison of original and corrected text.
        
        Args:
            document: Processed document dictionary
            output_path: Output file path
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required to create PDF files")
            
        # Set text size limits to avoid overflows
        MAX_TEXT_SIZE = 50000  # Maximum character count for a single textbox in comparison
        MAX_PAGE_LENGTH = 3000  # Maximum length of text per page in comparison
        
        # Create a new PDF document
        doc = fitz.open()
        
        # Get document metadata
        metadata = document.get("metadata", {})
        
        # Set document metadata
        doc.set_metadata({
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": "Text comparison document",
            "creator": "Raggiro OCR Correction",
            "producer": "Raggiro",
            "creationDate": fitz.get_pdf_now(),
        })
        
        # Extract original and corrected text
        # Check if we have explicit original_text in the document
        has_original = "original_text" in document or "original_pages" in document
        
        # Process each page separately if available
        pages = document.get("pages", [])
        original_pages = document.get("original_pages", [])
        
        # If we don't have explicit original text, check if the extraction method is OCR
        is_ocr = document.get("extraction_method", "") in ["pdf_ocr", "image_ocr"]
        
        if pages:
            # Create a page for each text page
            for i, page_data in enumerate(pages):
                corrected_text = page_data.get("text", "")
                
                # Get original text if available - use multiple fallback approaches
                original_text = ""
                
                # Debug avanzato per diagnosticare problemi con i testi
                print("-" * 80)
                print(f"DIAGNOSTICA PAGINA {i+1}:")
                print(f"Chiavi disponibili: {list(page_data.keys())}")
                
                # Verifica di tutte le possibili fonti di testo originale
                if has_original and i < len(original_pages):
                    original_keys = list(original_pages[i].keys())
                    print(f"Chiavi in original_pages[{i}]: {original_keys}")
                    original_page_text = original_pages[i].get("text", "")
                    original_page_raw = original_pages[i].get("raw_text", "")
                    
                    # Aggiungi statistiche sui contenuti
                    print(f"Lunghezza original_pages[{i}]['text']: {len(original_page_text)}")
                    print(f"Lunghezza original_pages[{i}]['raw_text']: {len(original_page_raw)}")
                
                # Verifica contenuto in page_data
                if "raw_text" in page_data:
                    print(f"Lunghezza page_data['raw_text']: {len(page_data['raw_text'])}")
                if "original_text" in page_data:
                    print(f"Lunghezza page_data['original_text']: {len(page_data['original_text'])}")
                if "text" in page_data:
                    print(f"Lunghezza page_data['text']: {len(page_data['text'])}")
                print("-" * 80)
                
                # Approccio prioritizzato per ottenere il testo originale - più affidabile e con log dettagliati
                
                # Priorità 1: Usa original_text da page_data (aggiunto nelle modifiche recenti)
                if "original_text" in page_data and page_data.get("original_text", "").strip():
                    print(f"✓ USANDO original_text da page_data (priorità 1) per pagina {i+1}")
                    original_text = page_data.get("original_text", "")
                
                # Priorità 2: Usa raw_text da page_data (opzione più comune)
                elif "raw_text" in page_data and page_data.get("raw_text", "").strip():
                    print(f"✓ USANDO raw_text da page_data (priorità 2) per pagina {i+1}")
                    original_text = page_data.get("raw_text", "")
                
                # Priorità 3: Usa text da original_pages
                elif has_original and i < len(original_pages) and original_pages[i].get("text", "").strip():
                    print(f"✓ USANDO text da original_pages (priorità 3) per pagina {i+1}")
                    original_text = original_pages[i].get("text", "")
                
                # Priorità 4: Usa raw_text da original_pages
                elif has_original and i < len(original_pages) and original_pages[i].get("raw_text", "").strip():
                    print(f"✓ USANDO raw_text da original_pages (priorità 4) per pagina {i+1}")
                    original_text = original_pages[i].get("raw_text", "")
                
                # Log which approach worked
                if original_text.strip():
                    print(f"Successfully retrieved original text for page {i+1} ({len(original_text)} chars)")
                else:
                    print(f"WARNING: Could not find original text for page {i+1}")
                
                # Create a new page for side-by-side comparison
                page = doc.new_page(width=842, height=595)  # A4 landscape
                
                # Add a header with character counts
                orig_len = len(original_text)
                corr_len = len(corrected_text)
                char_diff = corr_len - orig_len
                diff_sign = "+" if char_diff >= 0 else ""
                
                header_text = f"Text Comparison - Page {i+1}/{len(pages)} (Original: {orig_len} chars, Corrected: {corr_len} chars, Diff: {diff_sign}{char_diff})"
                page.insert_text((50, 40), header_text, fontsize=12, fontname="helv", color=(0, 0, 0.8))
                
                # Add section titles - use standard helvetica (helv) which is built into PDFs
                page.insert_text((150, 70), "Original Text", fontsize=11, fontname="helv", color=(0, 0, 0))
                page.insert_text((550, 70), "Corrected Text", fontsize=11, fontname="helv", color=(0, 0, 0))
                
                # Add divider lines
                page.draw_line((50, 80), (792, 80), color=(0, 0, 0.5), width=1)  # Horizontal
                page.draw_line((421, 80), (421, 545), color=(0, 0, 0.5), width=1)  # Vertical divider
                
                # Add text content in two columns
                rect_orig = fitz.Rect(50, 90, 411, 545)  # Left column
                rect_corr = fitz.Rect(431, 90, 792, 545)  # Right column
                
                # Insert original text with error handling and size management
                try:
                    # Verifica che il testo originale non sia vuoto
                    if not original_text.strip():
                        original_text = "ATTENZIONE: Testo originale non disponibile.\nPotrebbe esserci un problema con l'estrazione del testo originale."
                        print(f"Warning: Original text is empty for page {i+1}")
                    
                    # Truncate text if it's too long to avoid memory issues
                    elif len(original_text) > MAX_TEXT_SIZE:
                        truncated_text = original_text[:MAX_TEXT_SIZE] + "\n\n[... Text truncated due to size limits ...]"
                        print(f"Warning: Truncating original text from {len(original_text)} to {len(truncated_text)} characters")
                        original_text = truncated_text
                    
                    # Log di debug per verificare il contenuto
                    print(f"Original text sample for page {i+1}: {original_text[:100]}...")
                        
                    page.insert_textbox(rect_orig, original_text, fontsize=10, fontname="helv",
                                       align=0, color=(0, 0, 0))
                except Exception as e:
                    # If textbox fails, try simpler text insertion
                    print(f"Warning: Failed to insert original text box: {str(e)}")
                    page.insert_text((rect_orig.x0, rect_orig.y0 + 20), 
                                   "Error rendering text. See JSON output for content.", 
                                   fontsize=10, fontname="helv", color=(0.8, 0, 0))
                
                # Insert corrected text with error handling and size management
                try:
                    # Verifica che il testo corretto non sia vuoto
                    if not corrected_text.strip():
                        corrected_text = "ATTENZIONE: Testo corretto non disponibile."
                        print(f"Warning: Corrected text is empty for page {i+1}")
                    
                    # Truncate text if it's too long to avoid memory issues
                    elif len(corrected_text) > MAX_TEXT_SIZE:
                        truncated_text = corrected_text[:MAX_TEXT_SIZE] + "\n\n[... Text truncated due to size limits ...]"
                        print(f"Warning: Truncating corrected text from {len(corrected_text)} to {len(truncated_text)} characters")
                        corrected_text = truncated_text
                        
                    # Log di debug per verificare il contenuto
                    print(f"Corrected text sample for page {i+1}: {corrected_text[:100]}...")
                        
                    page.insert_textbox(rect_corr, corrected_text, fontsize=10, fontname="helv",
                                       align=0, color=(0, 0, 0))
                except Exception as e:
                    # If textbox fails, try simpler text insertion
                    print(f"Warning: Failed to insert corrected text box: {str(e)}")
                    page.insert_text((rect_corr.x0, rect_corr.y0 + 20), 
                                   "Error rendering text. See JSON output for content.", 
                                   fontsize=10, fontname="helv", color=(0.8, 0, 0))
        else:
            # If no individual pages, use the full text
            corrected_text = document.get("text", "")
            
            # Multiple approaches to get the original text for full document
            original_text = ""
            
            # Debug logging to understand what fields are available
            print(f"Full document keys: {[k for k in document.keys() if k not in ['pages', 'original_pages']]}")
            
            # Approach 1: Use original_text field
            if "original_text" in document:
                print("Using original_text approach for full document")
                original_text = document.get("original_text", "")
            
            # Approach 2: Use raw_text field
            if not original_text.strip() and "raw_text" in document:
                print("Using raw_text approach for full document")
                original_text = document.get("raw_text", "")
                
            # Log success or failure
            if original_text.strip():
                print(f"Successfully retrieved original text for full document ({len(original_text)} chars)")
            else:
                print(f"WARNING: Could not find original text for full document")
            
            # Create a new page for side-by-side comparison
            page = doc.new_page(width=842, height=595)  # A4 landscape
            
            # Add a header with character counts
            orig_len = len(original_text)
            corr_len = len(corrected_text)
            char_diff = corr_len - orig_len
            diff_sign = "+" if char_diff >= 0 else ""
            
            header_text = f"Text Comparison (Original: {orig_len} chars, Corrected: {corr_len} chars, Diff: {diff_sign}{char_diff})"
            page.insert_text((50, 40), header_text, fontsize=12, fontname="helv", color=(0, 0, 0.8))
            
            # Add section titles - use standard helvetica (helv) which is built into PDFs
            page.insert_text((150, 70), "Original Text", fontsize=11, fontname="helv", color=(0, 0, 0))
            page.insert_text((550, 70), "Corrected Text", fontsize=11, fontname="helv", color=(0, 0, 0))
            
            # Add divider lines
            page.draw_line((50, 80), (792, 80), color=(0, 0, 0.5), width=1)  # Horizontal
            page.draw_line((421, 80), (421, 545), color=(0, 0, 0.5), width=1)  # Vertical divider
            
            # Add text content in two columns
            rect_orig = fitz.Rect(50, 90, 411, 545)  # Left column
            rect_corr = fitz.Rect(431, 90, 792, 545)  # Right column
            
            # Insert original text with error handling and size management
            try:
                # Truncate text if it's too long to avoid memory issues
                if len(original_text) > MAX_TEXT_SIZE:
                    truncated_text = original_text[:MAX_TEXT_SIZE] + "\n\n[... Text truncated due to size limits ...]"
                    print(f"Warning: Truncating original text from {len(original_text)} to {len(truncated_text)} characters")
                    original_text = truncated_text
                    
                page.insert_textbox(rect_orig, original_text, fontsize=10, fontname="helv",
                                   align=0, color=(0, 0, 0))
            except Exception as e:
                # If textbox fails, try simpler text insertion
                print(f"Warning: Failed to insert original text box: {str(e)}")
                page.insert_text((rect_orig.x0, rect_orig.y0 + 20), 
                               "Error rendering text. See JSON output for content.", 
                               fontsize=10, fontname="helv", color=(0.8, 0, 0))
            
            # Insert corrected text with error handling and size management
            try:
                # Truncate text if it's too long to avoid memory issues
                if len(corrected_text) > MAX_TEXT_SIZE:
                    truncated_text = corrected_text[:MAX_TEXT_SIZE] + "\n\n[... Text truncated due to size limits ...]"
                    print(f"Warning: Truncating corrected text from {len(corrected_text)} to {len(truncated_text)} characters")
                    corrected_text = truncated_text
                    
                page.insert_textbox(rect_corr, corrected_text, fontsize=10, fontname="helv",
                                   align=0, color=(0, 0, 0))
            except Exception as e:
                # If textbox fails, try simpler text insertion
                print(f"Warning: Failed to insert corrected text box: {str(e)}")
                page.insert_text((rect_corr.x0, rect_corr.y0 + 20), 
                               "Error rendering text. See JSON output for content.", 
                               fontsize=10, fontname="helv", color=(0.8, 0, 0))
        
        # Save the PDF
        doc.save(output_path)
        doc.close()