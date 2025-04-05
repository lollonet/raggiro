"""Module for exporting processed documents in various formats."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        # Create a clean JSON export
        export_data = {
            "metadata": document.get("metadata", {}),
            "text": document.get("text", ""),
        }
        
        # Add segments and chunks if available
        if "segments" in document:
            export_data["segments"] = document["segments"]
            
        if "chunks" in document:
            export_data["chunks"] = document["chunks"]
        
        # Write to file
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