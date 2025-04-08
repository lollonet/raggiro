"""Module for file detection, validation, and I/O operations."""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import magic  # For MIME type detection

# Supported file types
SUPPORTED_EXTENSIONS = {
    # Document formats
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".rtf": "application/rtf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    
    # Image formats (for OCR)
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}

class FileHandler:
    """Handles file operations for the document processing pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the file handler.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.processed_files = set()
    
    def get_files(self, input_path: Union[str, Path], recursive: bool = True) -> List[Path]:
        """Get all processable files from a directory or a single file.
        
        Args:
            input_path: Path to a file or directory
            recursive: Whether to search directories recursively
            
        Returns:
            List of file paths that can be processed
        """
        input_path = Path(input_path)
        files = []
        
        if input_path.is_file():
            if self._is_supported_file(input_path):
                files.append(input_path)
        elif input_path.is_dir():
            glob_pattern = "**/*" if recursive else "*"
            for file_path in input_path.glob(glob_pattern):
                if file_path.is_file() and self._is_supported_file(file_path):
                    files.append(file_path)
        
        return files
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if the file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Whether the file is supported
        """
        # Check by extension first (faster)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        
        try:
            # Use python-magic to verify actual file type
            mime = magic.from_file(str(file_path), mime=True)
            expected_mime = SUPPORTED_EXTENSIONS[file_path.suffix.lower()]
            
            # Some flexibility in MIME type matching
            if mime == expected_mime or expected_mime in mime or mime in expected_mime:
                return True
                
            return False
        except Exception:
            return False  # If we can't determine the file type, assume it's not supported
    
    def compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """Compute SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash as a hexadecimal string
        """
        file_path = Path(file_path)
        h = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
                
        return h.hexdigest()
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict:
        """Get basic file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of file metadata
        """
        file_path = Path(file_path)
        stats = file_path.stat()
        
        return {
            "filename": file_path.name,
            "path": str(file_path.absolute()),
            "extension": file_path.suffix.lower(),
            "size_bytes": stats.st_size,
            "created_at": stats.st_ctime,
            "modified_at": stats.st_mtime,
            "hash": self.compute_file_hash(file_path),
        }
    
    def ensure_output_dir(self, output_dir: Union[str, Path]) -> Path:
        """Ensure the output directory exists.
        
        Args:
            output_dir: Path to the output directory
            
        Returns:
            Path object for the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def detect_file_type(self, file_path: Union[str, Path]) -> Dict:
        """Detect detailed file type information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with detailed file type information
        """
        file_path = Path(file_path)
        
        mime = magic.from_file(str(file_path), mime=True)
        description = magic.from_file(str(file_path))
        
        result = {
            "mime_type": mime,
            "description": description,
            "extension": file_path.suffix.lower(),
        }
        
        # Determine document type
        if mime == "application/pdf":
            result["document_type"] = "pdf"
        elif "word" in mime or file_path.suffix.lower() in [".docx", ".doc"]:
            result["document_type"] = "word"
        elif "excel" in mime or file_path.suffix.lower() in [".xlsx", ".xls"]:
            result["document_type"] = "spreadsheet"
        elif mime in ["text/plain", "text/html", "application/rtf"]:
            result["document_type"] = "text"
        elif mime.startswith("image/"):
            result["document_type"] = "image"
        else:
            result["document_type"] = "unknown"
            
        # Detect format hints from filename and extension
        filename_lower = file_path.name.lower()
        
        # Format hints help with document classification
        result["format_hints"] = []
        
        # Document format hints based on filename
        format_keywords = {
            "technical": ["manual", "guide", "documentation", "specification", "technical", "manuale", "guida"],
            "legal": ["contract", "agreement", "legal", "law", "regulation", "policy", "contratto", "legale"],
            "academic": ["paper", "thesis", "dissertation", "research", "journal", "tesi", "ricerca"],
            "business": ["report", "presentation", "financial", "business", "corporate", "rapporto", "presentazione"],
            "structured": ["form", "invoice", "cv", "resume", "application", "modulo", "fattura", "curriculum"],
            "narrative": ["article", "story", "book", "novel", "blog", "articolo", "libro", "storia"]
        }
        
        # Check filename for format hints
        for format_type, keywords in format_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    result["format_hints"].append(format_type)
                    break
                    
        # Remove duplicates while preserving order
        result["format_hints"] = list(dict.fromkeys(result["format_hints"]))
            
        return result