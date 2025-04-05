"""Module for text extraction from various document formats."""

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# PDF extraction
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text

# Document extraction
import docx
import openpyxl
import pandas as pd
from bs4 import BeautifulSoup

# OCR for scanned documents and images
import pytesseract
from PIL import Image

# Character encoding detection
import chardet

class Extractor:
    """Extracts text content from various document formats."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # OCR config
        self.ocr_enabled = self.config.get("extraction", {}).get("ocr_enabled", True)
        self.ocr_language = self.config.get("extraction", {}).get("ocr_language", "eng")
        
        # Configure pytesseract path if provided
        tesseract_path = self.config.get("extraction", {}).get("tesseract_path")
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract(self, file_path: Union[str, Path], file_type_info: Dict) -> Dict:
        """Extract text from a file based on its type.
        
        Args:
            file_path: Path to the file
            file_type_info: File type information from file_handler
            
        Returns:
            Dictionary with extracted text and metadata
        """
        file_path = Path(file_path)
        document_type = file_type_info.get("document_type", "unknown")
        mime_type = file_type_info.get("mime_type", "")
        
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": None,
            "has_text_layer": False,
            "success": False,
            "error": None,
        }
        
        try:
            if document_type == "pdf":
                return self._extract_pdf(file_path)
            elif document_type == "word":
                return self._extract_docx(file_path)
            elif document_type == "spreadsheet":
                return self._extract_excel(file_path)
            elif document_type == "text":
                if "html" in mime_type:
                    return self._extract_html(file_path)
                elif "rtf" in mime_type:
                    return self._extract_rtf(file_path)
                else:
                    return self._extract_text(file_path)
            elif document_type == "image" and self.ocr_enabled:
                return self._extract_image_ocr(file_path)
            else:
                result["error"] = f"Unsupported document type: {document_type}"
                return result
                
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _extract_pdf(self, file_path: Path) -> Dict:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "pdf",
            "has_text_layer": False,
            "success": False,
            "error": None,
        }
        
        try:
            # First try PyMuPDF (faster and better metadata)
            doc = fitz.open(file_path)
            
            # Extract metadata
            result["metadata"] = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "total_pages": len(doc),
            }
            
            # Extract text from each page
            full_text = []
            pages = []
            has_text = False
            
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    has_text = True
                    
                full_text.append(text)
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "has_text": bool(text.strip()),
                })
            
            result["has_text_layer"] = has_text
            result["text"] = "\n\n".join(full_text)
            result["pages"] = pages
            
            # If no text layer is detected and OCR is enabled, use OCR
            if not has_text and self.ocr_enabled:
                ocr_result = self._extract_pdf_with_ocr(file_path)
                if ocr_result["success"]:
                    result = ocr_result
                    result["extraction_method"] = "pdf_ocr"
            
            # If still no text, try pdfminer as a fallback
            if not result["text"].strip() and not result["error"]:
                try:
                    text = pdfminer_extract_text(str(file_path))
                    if text.strip():
                        result["text"] = text
                        result["extraction_method"] = "pdf_pdfminer"
                        result["has_text_layer"] = True
                except Exception as e:
                    # Just log the error but keep the PyMuPDF result
                    pass
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_pdf_with_ocr(self, file_path: Path) -> Dict:
        """Extract text from a PDF file using OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "pdf_ocr",
            "has_text_layer": False,
            "success": False,
            "error": None,
        }
        
        try:
            doc = fitz.open(file_path)
            
            # Basic metadata
            result["metadata"] = {
                "total_pages": len(doc),
            }
            
            # Extract text from each page using OCR
            full_text = []
            pages = []
            
            for i, page in enumerate(doc):
                # Convert page to an image
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                # Perform OCR
                text = pytesseract.image_to_string(img, lang=self.ocr_language)
                
                full_text.append(text)
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "has_text": bool(text.strip()),
                })
            
            result["text"] = "\n\n".join(full_text)
            result["pages"] = pages
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_docx(self, file_path: Path) -> Dict:
        """Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "docx",
            "has_text_layer": True,
            "success": False,
            "error": None,
        }
        
        try:
            doc = docx.Document(file_path)
            
            # Extract metadata
            core_properties = doc.core_properties
            result["metadata"] = {
                "title": core_properties.title or "",
                "author": core_properties.author or "",
                "subject": core_properties.subject or "",
                "keywords": core_properties.keywords or "",
                "created": core_properties.created.isoformat() if core_properties.created else "",
                "modified": core_properties.modified.isoformat() if core_properties.modified else "",
                "last_modified_by": core_properties.last_modified_by or "",
                "category": core_properties.category or "",
                "paragraphs": len(doc.paragraphs),
            }
            
            # Extract text from paragraphs
            text_content = [p.text for p in doc.paragraphs]
            
            # Handle tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    text_content.append(" | ".join(row_text))
            
            result["text"] = "\n".join(text_content)
            result["success"] = True
            
            # DOCX doesn't have the concept of pages in the same way as PDF
            result["pages"] = [{
                "page_num": 1,
                "text": result["text"],
                "has_text": True,
            }]
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_excel(self, file_path: Path) -> Dict:
        """Extract text from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "excel",
            "has_text_layer": True,
            "success": False,
            "error": None,
        }
        
        try:
            # Try with pandas first (handles more formats, including .xls)
            try:
                # Read all sheets
                sheets = pd.read_excel(file_path, sheet_name=None)
                
                text_parts = []
                sheet_texts = []
                
                for sheet_name, df in sheets.items():
                    sheet_text = f"Sheet: {sheet_name}\n"
                    sheet_text += df.to_string(index=False)
                    text_parts.append(sheet_text)
                    sheet_texts.append({
                        "sheet_name": sheet_name,
                        "text": sheet_text,
                    })
                
                result["text"] = "\n\n".join(text_parts)
                result["pages"] = sheet_texts
                result["metadata"]["sheets"] = list(sheets.keys())
                result["metadata"]["total_sheets"] = len(sheets)
                
            except Exception:
                # Fallback to openpyxl (XLSX only)
                workbook = openpyxl.load_workbook(file_path, data_only=True)
                
                text_parts = []
                sheet_texts = []
                
                for sheet in workbook:
                    rows = []
                    for row in sheet.iter_rows(values_only=True):
                        rows.append(" | ".join(str(cell) if cell is not None else "" for cell in row))
                    
                    sheet_text = f"Sheet: {sheet.title}\n"
                    sheet_text += "\n".join(rows)
                    
                    text_parts.append(sheet_text)
                    sheet_texts.append({
                        "sheet_name": sheet.title,
                        "text": sheet_text,
                    })
                
                result["text"] = "\n\n".join(text_parts)
                result["pages"] = sheet_texts
                result["metadata"]["sheets"] = [sheet.title for sheet in workbook]
                result["metadata"]["total_sheets"] = len(workbook.sheetnames)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_html(self, file_path: Path) -> Dict:
        """Extract text from an HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "html",
            "has_text_layer": True,
            "success": False,
            "error": None,
        }
        
        try:
            with open(file_path, "rb") as file:
                content = file.read()
                encoding = chardet.detect(content)["encoding"] or "utf-8"
                html_content = content.decode(encoding, errors="replace")
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract metadata
            title = soup.title.text if soup.title else ""
            meta_tags = {}
            
            for meta in soup.find_all("meta"):
                name = meta.get("name", meta.get("property", ""))
                if name:
                    meta_tags[name] = meta.get("content", "")
            
            result["metadata"] = {
                "title": title,
                "meta_tags": meta_tags,
            }
            
            # Remove scripts and styles for cleaner text
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Remove excessive newlines
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)
            
            result["text"] = text
            result["pages"] = [{
                "page_num": 1,
                "text": text,
                "has_text": True,
            }]
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_text(self, file_path: Path) -> Dict:
        """Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with extracted text
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "text",
            "has_text_layer": True,
            "success": False,
            "error": None,
        }
        
        try:
            with open(file_path, "rb") as file:
                content = file.read()
                
                # Detect encoding
                detected = chardet.detect(content)
                encoding = detected["encoding"] or "utf-8"
                confidence = detected["confidence"]
                
                text = content.decode(encoding, errors="replace")
            
            result["text"] = text
            result["metadata"] = {
                "encoding": encoding,
                "encoding_confidence": confidence,
            }
            result["pages"] = [{
                "page_num": 1,
                "text": text,
                "has_text": True,
            }]
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_rtf(self, file_path: Path) -> Dict:
        """Extract text from an RTF file.
        
        Args:
            file_path: Path to the RTF file
            
        Returns:
            Dictionary with extracted text
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "rtf",
            "has_text_layer": True,
            "success": False,
            "error": None,
        }
        
        try:
            # Since RTF can be complex, we'll try a few methods
            
            # First try to read as plain text (works for simple RTF)
            with open(file_path, "rb") as file:
                content = file.read()
                
                # Try to decode as is
                try:
                    text = content.decode("utf-8", errors="replace")
                    
                    # Remove RTF control sequences
                    cleaned_text = ""
                    i = 0
                    in_control = False
                    
                    while i < len(text):
                        if text[i] == "\\" and not in_control:
                            in_control = True
                            i += 1
                        elif in_control and text[i].isalpha():
                            # Skip the control word
                            while i < len(text) and text[i].isalpha():
                                i += 1
                            in_control = False
                        elif in_control and text[i] == "'":
                            # Skip hex character
                            i += 3
                            in_control = False
                        elif in_control:
                            # Single character control
                            i += 1
                            in_control = False
                        else:
                            cleaned_text += text[i]
                            i += 1
                    
                    text = cleaned_text
                    
                except Exception:
                    # If the above doesn't work, try pip install striprtf
                    # For now, use a simple fallback
                    text = str(content)
                    text = text.replace("\\par", "\n")
                    text = text.replace("\\tab", "\t")
            
            result["text"] = text
            result["pages"] = [{
                "page_num": 1,
                "text": text,
                "has_text": True,
            }]
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def _extract_image_ocr(self, file_path: Path) -> Dict:
        """Extract text from an image using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with extracted text
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "extraction_method": "image_ocr",
            "has_text_layer": False,
            "success": False,
            "error": None,
        }
        
        if not self.ocr_enabled:
            result["error"] = "OCR is disabled in configuration"
            return result
        
        try:
            # Open the image
            image = Image.open(file_path)
            
            # Get image metadata
            result["metadata"] = {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            
            result["text"] = text
            result["pages"] = [{
                "page_num": 1,
                "text": text,
                "has_text": bool(text.strip()),
            }]
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            
        return result