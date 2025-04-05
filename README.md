# Raggiro

Advanced document processing pipeline for RAG (Retrieval-Augmented Generation) applications

![Raggiro](https://img.shields.io/badge/Raggiro-v0.1.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Overview

Raggiro is a comprehensive document processing framework designed for building local, offline RAG systems. It provides end-to-end capabilities from document ingestion to response generation, with a focus on modular, configurable components that can be used together or independently.

## Features

- **Comprehensive document support**: PDF (native and scanned), DOCX, TXT, HTML, RTF, XLSX, images with text
- **Advanced preprocessing**: Extraction, cleaning, normalization and logical segmentation
- **Metadata extraction**: Title, author, date, language, document type, category detection
- **Structured output**: Markdown and JSON formats with all metadata
- **Modular architecture**: CLI, Python API, and GUI interfaces (Streamlit/Textual)
- **Fully offline operation**: Works without external API dependencies
- **Complete RAG pipeline**: Integrated vector indexing, retrieval, and response generation

## Installation

### Basic Installation

```bash
# From PyPI (not yet available)
pip install raggiro

# From the repository
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -e .
```

### Install with Optional Dependencies

```bash
# For development tools
pip install -e ".[dev]"

# For vector database support
pip install -e ".[qdrant]"

# For testing capabilities
pip install -e ".[test]"

# For all optional dependencies
pip install -e ".[dev,qdrant,test]"
```

### System Requirements

Some features require additional system dependencies:

- **OCR**: Requires Tesseract OCR to be installed
  ```bash
  # Ubuntu/Debian
  sudo apt install tesseract-ocr
  
  # macOS
  brew install tesseract
  ```

- **PDF Processing**: Some PDF operations may require Poppler
  ```bash
  # Ubuntu/Debian
  sudo apt install poppler-utils
  
  # macOS
  brew install poppler
  ```

## User Guide

### Command Line Interface

The CLI provides the primary interface for document processing:

```bash
# Process a single file
raggiro process document.pdf --output output_dir

# Process a directory of documents
raggiro process documents/ --output output_dir

# Enable OCR for processing scanned documents
raggiro process document.pdf --output output_dir --ocr

# Specify output formats
raggiro process document.pdf --output output_dir --format markdown json

# Set logging level
raggiro process documents/ --output output_dir --log-level debug
```

### GUI Interface

Raggiro includes both Streamlit and Textual-based GUI interfaces:

```bash
# Launch Streamlit GUI (web-based)
raggiro gui

# Launch Textual GUI (terminal-based)
raggiro gui --tui
```

### Python API

You can use Raggiro programmatically in your Python code:

```python
from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

# Load configuration
config = load_config()

# Process a document
processor = DocumentProcessor(config)
result = processor.process_file("document.pdf", "output_dir")

# Index processed documents
indexer = VectorIndexer(config)
indexer.index_directory("output_dir", "index_dir")

# Create a RAG pipeline and query it
pipeline = RagPipeline(config)
pipeline.retriever.load_index("index_dir")
response = pipeline.query("What is the main topic of this document?")
print(response["response"])
```

### Complete Processing Pipeline Example

Here's a complete example demonstrating the entire Raggiro workflow:

```python
import os
from pathlib import Path
from raggiro.processor import DocumentProcessor
from raggiro.rag.indexer import VectorIndexer
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

# Directories
input_dir = "documents"
output_dir = "processed"
index_dir = "index"

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# Load configuration 
config = load_config()

# 1. Process documents
print("Processing documents...")
processor = DocumentProcessor(config)
process_result = processor.process_directory(input_dir, output_dir, recursive=True)

if process_result["success"]:
    print(f"Processed {process_result['summary']['total_files']} documents")
    print(f"Success rate: {process_result['summary']['success_rate']}%")
    
    # 2. Index the processed documents
    print("\nIndexing documents...")
    indexer = VectorIndexer(config)
    index_result = indexer.index_directory(output_dir)
    
    if index_result["success"]:
        print(f"Indexed {index_result['summary']['total_chunks_indexed']} chunks")
        
        # Save the index
        indexer.save_index(index_dir)
        print(f"Index saved to {index_dir}")
        
        # 3. Set up RAG pipeline for queries
        print("\nSetting up RAG pipeline...")
        pipeline = RagPipeline(config)
        pipeline.retriever.load_index(index_dir)
        
        # 4. Query the pipeline
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            print("\nProcessing query...")
            result = pipeline.query(query)
            
            if result["success"]:
                print("\nResponse:")
                print(result["response"])
                print(f"\nUsed {result['chunks_used']} chunks for this response")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Indexing failed: {index_result.get('error', 'Unknown error')}")
else:
    print(f"Processing failed: {process_result.get('error', 'Unknown error')}")
```

## Configuration

Raggiro uses TOML configuration files for customization. You can create a config file at `~/.raggiro/config.toml` or specify a custom path with `--config`.

### Example Configuration

```toml
# Processing settings
[processing]
dry_run = false
recursive = true

# Logging settings
[logging]
log_level = "info"
log_to_file = true

# Extraction settings
[extraction]
ocr_enabled = true
ocr_language = "eng"

# Cleaning settings
[cleaning]
remove_headers_footers = true
normalize_whitespace = true
remove_special_chars = true

# Segmentation settings
[segmentation]
use_spacy = true
spacy_model = "en_core_web_sm"
max_chunk_size = 1000
chunk_overlap = 200

# Export settings
[export]
formats = ["markdown", "json"]
include_metadata = true

# Indexing settings
[indexing]
embedding_model = "all-MiniLM-L6-v2"
vector_db = "faiss"

# Query rewriting settings
[rewriting]
enabled = true
llm_type = "ollama"
model_name = "llama3"

# Response generation settings
[generation]
llm_type = "ollama"
model_name = "mistral"
temperature = 0.7
```

## Architecture

Raggiro follows a modular architecture with clearly separated components:

```
raggiro/
├── cli/          # Command-line interface
├── gui/          # GUI interfaces (Streamlit/Textual)
├── core/         # Core processing modules
│   ├── file_handler.py  # File detection and validation
│   ├── extractor.py     # Text extraction from various formats
│   ├── cleaner.py       # Text cleaning and normalization
│   ├── segmenter.py     # Logical segmentation of text
│   ├── metadata.py      # Metadata extraction
│   ├── exporter.py      # Output generation (Markdown, JSON)
│   └── logger.py        # Logging and reporting
├── models/       # Classification models
├── rag/          # RAG pipeline components
│   ├── indexer.py      # Vector indexing
│   ├── retriever.py    # Vector retrieval
│   ├── rewriter.py     # Query rewriting
│   ├── generator.py    # Response generation
│   └── pipeline.py     # Complete RAG pipeline
├── utils/        # Utility functions
├── testing/      # Testing utilities
└── examples/     # Example code and documents
```

## RAG Integration

Raggiro provides a complete RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: Extracts, cleans, and segments documents into logical chunks
2. **Vector Indexing**: Indexes chunks using FAISS or Qdrant for vector search
3. **Query Processing**: Supports query rewriting for improved retrieval
4. **Local LLM Integration**: Works with Ollama, LLaMA, and other local models
5. **Response Generation**: Creates high-quality responses with proper source citations

### RAG Pipeline Components

The RAG pipeline includes:

- **VectorIndexer**: Creates and manages vector indices
- **VectorRetriever**: Retrieves relevant chunks from the index
- **QueryRewriter**: Enhances queries for better retrieval
- **ResponseGenerator**: Generates responses from retrieved chunks
- **RagPipeline**: Orchestrates the entire RAG workflow

### Example: Using the RAG Pipeline

```python
from raggiro.rag.pipeline import RagPipeline
from raggiro.utils.config import load_config

# Load configuration
config = load_config()

# Initialize the RAG pipeline
pipeline = RagPipeline(config)

# Load a previously created index
pipeline.retriever.load_index("index_directory")

# Query the pipeline
result = pipeline.query("What are the key benefits of this product?")

# Display the response
print(result["response"])

# Get information about the query process
if "rewritten_query" in result:
    print(f"Original query: {result['original_query']}")
    print(f"Rewritten query: {result['rewritten_query']}")

print(f"Chunks used: {result['chunks_used']}")
```

## Testing and Evaluation

Raggiro includes tools for testing and evaluating your RAG system:

```bash
# Run promptfoo evaluations
raggiro test-rag --prompt-set evaluation/prompts.yaml --output test_results
```

You can also use the testing utilities programmatically:

```python
from raggiro.testing.promptfoo_runner import run_tests

# Run tests with custom prompt set and output directory
results = run_tests("path/to/prompts.yaml", "path/to/output")

# Check test results
print(f"Tests run: {results['tests_run']}")
print(f"Success rate: {results['success_rate']}%")
```

## API Reference

For detailed API documentation, see the [API Reference](https://github.com/lollonet/raggiro/wiki/API-Reference) in the wiki.

## Contributing

Contributions are welcome! Please check out our [Contribution Guidelines](https://github.com/lollonet/raggiro/wiki/Contributing) for details on how to submit pull requests, report issues, and suggest improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with inspiration from modern RAG architectures and document processing pipelines
- Utilizes many excellent open-source libraries for document parsing and text processing