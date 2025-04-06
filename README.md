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
- **Semantic chunking**: Intelligent content division based on meaning rather than just size
- **Metadata extraction**: Title, author, date, language, document type, category detection
- **Structured output**: Markdown and JSON formats with all metadata
- **Modular architecture**: CLI, Python API, and GUI interfaces (Streamlit/Textual)
- **Fully offline operation**: Works without external API dependencies
- **Complete RAG pipeline**: Integrated vector indexing, retrieval, and response generation
- **Testing utilities**: Tools for benchmarking and comparing chunking strategies

## Installation

### Basic Installation

```bash
# From PyPI (not yet available)
pip install raggiro

# From the repository with pip
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -e .

# From the repository with requirements.txt
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Download language model
```

### Install with Optional Dependencies

```bash
# For development tools
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt

# For vector database support
pip install -e ".[qdrant]"

# For testing capabilities (including PromptFoo)
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

### Command Line Interface Reference

Raggiro provides a comprehensive CLI with several commands for document processing, GUI access, and RAG testing:

#### Main Command

```bash
# Show help and available commands
raggiro --help

# Check version
raggiro --version
```

#### Document Processing Command

The `process` command is the core functionality for document processing:

```bash
# Show help for process command
raggiro process --help

# Process a single file
raggiro process document.pdf --output output_dir

# Process a directory of documents
raggiro process documents/ --output output_dir

# Process recursively (default) or non-recursively
raggiro process documents/ --output output_dir --recursive
raggiro process documents/ --output output_dir --no-recursive

# Enable OCR for scanned documents (default: enabled)
raggiro process document.pdf --output output_dir --ocr
raggiro process document.pdf --output output_dir --no-ocr

# Specify output formats (default: markdown and json)
raggiro process document.pdf --output output_dir --format markdown --format json

# Run in dry-run mode (no files written)
raggiro process document.pdf --output output_dir --dry-run

# Set logging level (default: info)
raggiro process documents/ --output output_dir --log-level debug

# Use a custom configuration file
raggiro process documents/ --output output_dir --config my_config.toml
```

#### GUI Interfaces

Raggiro includes both web-based (Streamlit) and terminal-based (Textual) GUI interfaces:

```bash
# Show help for GUI command
raggiro gui --help

# Launch Streamlit GUI (web-based)
raggiro gui
# Or run directly with Streamlit for better browser integration
streamlit run $(which raggiro)

# Launch Textual GUI (terminal-based)
raggiro gui --tui
```

#### RAG Testing Commands

The `test-rag` command allows automated testing of your RAG pipeline, using settings from your TOML configuration file:

```bash
# Show help for test-rag command
raggiro test-rag --help

# Run tests with a promptfoo configuration
raggiro test-rag --prompt-set tests/prompts.yaml

# Specify output directory for test results
raggiro test-rag --prompt-set tests/prompts.yaml --output test_results

# Override Ollama URL from the command line (overrides TOML config)
raggiro test-rag --prompt-set tests/prompts.yaml --ollama-url http://localhost:11434

# Specify different models for rewriting and generation (overrides TOML config)
raggiro test-rag --prompt-set tests/prompts.yaml --rewriting-model llama3 --generation-model mistral
```

### GUI Interface

Raggiro's GUI interfaces provide interactive document processing without having to use command-line parameters:

1. **Streamlit Interface (Web-based)**
   - **Document Processing**: Upload and process documents through a browser
   - **RAG Testing**: Run and evaluate tests on your processed documents
   - **Result Visualization**: View test results with metrics and charts
   - **Configuration Management**: Edit and manage configuration options visually
   - **Ollama Integration**: Dynamically fetches available models from Ollama server
   
   The Streamlit interface includes four main tabs:
   - **Process Documents**: Upload and process documents with customizable options
   - **Test RAG**: Run tests on processed documents with predefined or custom prompts and dynamic Ollama model selection
   - **View Results**: Analyze test results with visualizations and metrics
   - **Configuration**: Edit configuration settings for the entire pipeline, including LLM settings with real-time validation

2. **Textual Interface (Terminal-based)**
   - Text-based UI for environments without a web browser
   - Keyboard-driven interface
   - Useful for remote server environments
   - Low resource consumption

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

### Configuration Structure

Raggiro follows a clean separation of concerns in its configuration:

1. **TOML Configuration File** (`config/config.toml`): Contains all LLM settings, including:
   - Ollama server URLs
   - Model names for different components (rewriting, generation)
   - Provider-specific settings
   - Performance parameters (temperature, max_tokens)
   
2. **Test Prompt Files** (`test_prompts/*.yaml`): Contain only test-specific settings:
   - Test prompts for different documents
   - Assertions and evaluation criteria
   - Chunking strategy configuration
   
This separation allows you to maintain centralized LLM configuration while having document-specific test prompts.

### Example Configuration

```toml
# Processing settings
[processing]
dry_run = false
recursive = true

# Extraction settings
[extraction]
ocr_enabled = true
ocr_language = "eng"

# Segmentation settings
[segmentation]
use_spacy = true
spacy_model = "en_core_web_sm"
max_chunk_size = 500  # Size in characters for each chunk
chunk_overlap = 100   # Overlap between chunks
semantic_chunking = true  # Enable semantic-based chunking
chunking_strategy = "hybrid"  # Options: "size", "semantic", "hybrid"
semantic_similarity_threshold = 0.65  # Threshold for semantic similarity

# Export settings
[export]
formats = ["markdown", "json"]
include_metadata = true

# LLM settings (shared across components)
[llm]
provider = "ollama"  # "ollama", "llamacpp", "openai", "replicate"
ollama_base_url = "http://ollama:11434"  # Ollama API URL
ollama_timeout = 30  # Timeout in seconds
llamacpp_path = ""  # Path to llama.cpp executable
api_key = ""  # For OpenAI or other external providers
openai_model = "gpt-3.5-turbo"  # Default model for OpenAI

# Vector database settings
[vector_db]
type = "faiss"  # "faiss", "qdrant", "milvus"
qdrant_url = "http://qdrant:6333"
qdrant_collection = "raggiro"

# Embedding settings
[embedding]
model = "all-MiniLM-L6-v2"  # Model name for embeddings
dimensions = 384  # Embedding dimensions
device = "cpu"  # "cpu" or "cuda" for GPU acceleration

# Query rewriting settings
[rewriting]
enabled = true
llm_type = ${llm.provider}  # Inherit from llm section
temperature = 0.1
max_tokens = 200

# Model names by provider type
ollama_model = "llama3"  # Model name for Ollama
llamacpp_model = "llama3"  # Model name for LLaMA.cpp
openai_model = ${llm.openai_model}  # Inherit from llm section

# Provider-specific settings (inherited from llm section)
ollama_base_url = ${llm.ollama_base_url}
llamacpp_path = ${llm.llamacpp_path}
api_key = ${llm.api_key}  # For OpenAI
api_url = ${llm.api_url}  # For OpenAI

# Response generation settings
[generation]
llm_type = ${llm.provider}  # Inherit from llm section
temperature = 0.1  # Lower temperature for more deterministic outputs
max_tokens = 1000

# Model names by provider type
ollama_model = "mistral"  # Model name for Ollama
llamacpp_model = "mistral"  # Model name for LLaMA.cpp
openai_model = ${llm.openai_model}  # Inherit from llm section

# Provider-specific settings (inherited from llm section)
ollama_base_url = ${llm.ollama_base_url}
llamacpp_path = ${llm.llamacpp_path}
api_key = ${llm.api_key}  # For OpenAI
api_url = ${llm.api_url}  # For OpenAI
```

### LLM Configuration

Raggiro supports multiple LLM providers for query rewriting and response generation. All LLM configuration has been centralized in the TOML config file:

1. **Ollama** (default): Local LLM server
   ```toml
   # Shared LLM settings
   [llm]
   provider = "ollama"
   ollama_base_url = "http://localhost:11434"  # Change to your Ollama server URL
   ollama_timeout = 30  # Timeout in seconds
   
   # Query rewriting settings
   [rewriting]
   llm_type = ${llm.provider}  # Inherit from llm section
   ollama_model = "llama3"  # Model for query rewriting
   ollama_base_url = ${llm.ollama_base_url}  # Inherit from llm section
   
   # Response generation settings
   [generation]
   llm_type = ${llm.provider}  # Inherit from llm section
   ollama_model = "mistral"  # Model for response generation
   ollama_base_url = ${llm.ollama_base_url}  # Inherit from llm section
   ```

2. **llama.cpp**: Direct integration with llama.cpp binary
   ```toml
   # Shared LLM settings
   [llm]
   provider = "llamacpp"
   llamacpp_path = "/path/to/llama"  # Path to llama.cpp executable
   
   # Query rewriting settings
   [rewriting]
   llm_type = ${llm.provider}  # Inherit from llm section
   llamacpp_model = "llama3"  # Model for query rewriting
   
   # Response generation settings
   [generation]
   llm_type = ${llm.provider}  # Inherit from llm section
   llamacpp_model = "mistral"  # Model for response generation
   ```

3. **OpenAI**: Cloud-based model access
   ```toml
   # Shared LLM settings
   [llm]
   provider = "openai"
   api_key = "your-openai-api-key"
   openai_model = "gpt-3.5-turbo"  # Default model
   api_url = "https://api.openai.com/v1"  # Optional custom endpoint (for Azure OpenAI, etc.)
   
   # Query rewriting settings
   [rewriting]
   llm_type = ${llm.provider}  # Inherit from llm section
   openai_model = ${llm.openai_model}  # Inherit from llm section, or override with a specific model
   
   # Response generation settings
   [generation]
   llm_type = ${llm.provider}  # Inherit from llm section
   openai_model = "gpt-4"  # Override with a more powerful model for generation
   ```

You can mix and match different providers for query rewriting and response generation, for example, using a lighter model for query rewriting and a more powerful model for response generation. The TOML configuration uses variable interpolation to inherit values from the shared `[llm]` section, simplifying maintenance.

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
    └── scripts/  # Testing and evaluation scripts
```

## RAG Integration

Raggiro provides a complete RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: Extracts, cleans, and segments documents into logical chunks
2. **Semantic Chunking**: Intelligently divides text based on semantic meaning and context
3. **Vector Indexing**: Indexes chunks using FAISS or Qdrant for vector search
4. **Query Processing**: Supports query rewriting for improved retrieval
5. **Local LLM Integration**: Works with Ollama, LLaMA, and other local models
6. **Response Generation**: Creates high-quality responses with proper source citations

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

See the `raggiro/examples/` directory for more comprehensive examples, including:
- Basic usage examples in `examples/basic_usage.py`
- OpenAI integration in `examples/openai_integration.py`
- Testing scripts in `examples/scripts/`

## Processing Logs and Statistics

Raggiro generates detailed logs and statistics during document processing, providing valuable insights into the pipeline performance:

### Document Processing Logs

During document processing, Raggiro generates several log files:

```
output_dir/
├── logs/
│   ├── raggiro_20250405_123045.log        # Main processing log
│   ├── processed_files_20250405_123045.csv # CSV of all processed files
│   └── processing_summary_20250405_123045.json # Processing statistics
├── document1.md                          # Processed output files
├── document1.json
└── ...
```

The log directory is automatically created as a subdirectory of your specified output directory. By default, Raggiro sets logging configuration in `config.toml`:

```toml
# Logging settings
[logging]
log_level = "info"  # Options: debug, info, warning, error
log_to_file = true  # Whether to create log files
log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"
```

#### Sample Processing Log

```
2025-04-05 12:30:45 - INFO - Starting processing of /data/documents/
2025-04-05 12:30:45 - INFO - Found 24 files to process
2025-04-05 12:30:47 - INFO - FILE_HANDLER [Phase 1/5]: File validation - Processed /data/documents/report1.pdf successfully (45ms)
2025-04-05 12:30:48 - INFO - EXTRACTOR [Phase 2/5]: Text extraction - Processed /data/documents/report1.pdf successfully (1243ms)
2025-04-05 12:30:49 - INFO - CLEANER [Phase 3/5]: Text cleaning - Processed /data/documents/report1.pdf successfully (327ms)
2025-04-05 12:30:50 - INFO - SEGMENTER [Phase 4/5]: Text segmentation (23 segments, 8 chunks) - Processed /data/documents/report1.pdf successfully (412ms)
2025-04-05 12:30:51 - INFO - EXPORTER [Phase 5/5]: Export (markdown, json) - Processed /data/documents/report1.pdf successfully (198ms)
2025-04-05 12:30:51 - INFO - PROCESSOR: Complete (1520 words) - Processed /data/documents/report1.pdf successfully (2225ms)
2025-04-05 12:30:51 - INFO - Document Metadata:
2025-04-05 12:30:51 - INFO -   Title: Quarterly Financial Report - Q1 2025
2025-04-05 12:30:51 - INFO -   Author: Financial Analysis Team
2025-04-05 12:30:51 - INFO -   Date: 2025-03-15
2025-04-05 12:30:51 - INFO -   Language: en
2025-04-05 12:30:51 - INFO -   Topics: financial, report
2025-04-05 12:30:51 - INFO -   Segment Types: {'paragraph': 15, 'header': 6, 'section': 2}
2025-04-05 12:31:03 - INFO - Processing /data/documents/presentation.pptx
2025-04-05 12:31:03 - ERROR - EXTRACTOR: Text extraction - Failed to process /data/documents/presentation.pptx: Unsupported file type: .pptx
...
2025-04-05 12:35:28 - INFO - Processing summary: 22/24 files processed successfully (91.67%)
2025-04-05 12:35:28 - INFO - Summary saved to /output/logs/processing_summary_20250405_123528.json
2025-04-05 12:35:28 - INFO - Processing complete
```

#### Processing Statistics with Metadata Analysis

Raggiro generates a JSON summary with detailed statistics about the processing run, including metadata extraction quality:

```json
{
  "start_time": "2025-04-05T12:30:45.123456",
  "end_time": "2025-04-05T12:35:28.789012",
  "total_files": 24,
  "successful_files": 22,
  "failed_files": 2,
  "success_rate": 91.67,
  "file_types": {
    ".pdf": 15,
    ".docx": 5,
    ".txt": 2,
    ".pptx": 1,
    ".xlsx": 1
  },
  "extraction_methods": {
    "pdf": 10,
    "pdf_ocr": 5,
    "docx": 5,
    "text": 2
  },
  "languages": {
    "en": 18,
    "es": 2,
    "fr": 2
  },
  "topics": {
    "financial": 8,
    "technical": 6,
    "legal": 4,
    "report": 4
  },
  "metadata_completeness": {
    "title": 95.45,
    "author": 81.82,
    "date": 86.36,
    "language": 100.00,
    "topics": 77.27
  },
  "errors": {
    "Unsupported file type: .pptx": 1,
    "Failed to extract text: Document is password protected": 1
  }
}
```

### RAG Pipeline Metrics

When using the RAG pipeline, you can collect metrics on query performance:

```python
# Query the RAG pipeline and collect metrics
result = pipeline.query("What are the key benefits?", collect_metrics=True)

# Access metrics
print(f"Query processing time: {result['metrics']['query_time_ms']}ms")
print(f"Chunks retrieved: {result['metrics']['chunks_retrieved']}")
print(f"Top chunk similarity: {result['metrics']['top_similarity']:.2f}")
```

Example metrics output:
```
Query: "What are the key benefits?"
Rewritten query: "What are the key benefits and advantages described in this document?"
Query processing time: 325ms
Chunks retrieved: 3
Top chunk similarity: 0.87
Response generation time: 1254ms
Total processing time: 1579ms
```

## Testing and Evaluation

Raggiro includes comprehensive tools for testing and evaluating your RAG system, accessible through both command-line interfaces and the GUI:

### Testing Dependencies

Raggiro uses [PromptFoo](https://www.promptfoo.dev/) for advanced RAG evaluation and testing. This allows you to compare different chunking strategies, evaluate response quality, and generate metrics on your RAG system's performance.

#### Installing PromptFoo

PromptFoo is a Node.js application that needs to be installed via npm. To use the full testing capabilities:

```bash
# Ensure npm is installed first
npm install -g promptfoo
```

You can also use the provided installation script:

```bash
# Run the installation script
chmod +x install_promptfoo.sh
./install_promptfoo.sh
```

If you get a "command not found" error after installation, you may need to add the npm global bin directory to your PATH:

```bash
# Add this to your .bashrc or .zshrc
export PATH="$(npm config get prefix)/bin:$PATH"
```

If PromptFoo is not installed, the Streamlit interface will show an error message with installation instructions. Basic testing will still work, but advanced evaluation features will be limited.

### Running Evaluation Tests

#### Command Line Testing

```bash
# Run promptfoo evaluations
raggiro test-rag --prompt-set test_prompts/kenny_werner.yaml --output test_results

# Override Ollama settings when running tests
raggiro test-rag --prompt-set test_prompts/scrum_guide.yaml --output test_results --ollama-url http://localhost:11434

# Test semantic chunking with a specific document
python -m raggiro.examples.scripts.test_semantic_chunking --input /path/to/your/document.pdf --output test_output

# Compare different chunking strategies
python -m raggiro.examples.scripts.test_rag_comparison --input /path/to/your/document.pdf --strategies size semantic hybrid
```

#### GUI Testing with Streamlit

The Streamlit GUI provides a user-friendly interface for RAG testing:

1. **Test RAG Tab**:
   - Select processed documents to test
   - Choose from predefined test prompts or create custom ones
   - Execute tests with real-time progress tracking
   - View results immediately after test completion
   - Automatic PromptFoo installation detection with helpful messages
   - Full test execution with logs and progress indicators

2. **View Results Tab**:
   - Browse test history across different test runs
   - Compare results from different chunking strategies
   - View detailed metrics and statistics
   - See visualizations of test performance

3. **Test Capabilities**:
   - Semantic chunking evaluation
   - Query rewriting effectiveness
   - Response quality assessment
   - Retrieval accuracy metrics
   - Performance statistics

The GUI makes it easy to conduct and interpret tests without writing a single line of code, supporting both technical and non-technical users in evaluating RAG system performance.

### Testing Semantic Chunking

The semantic chunking feature can be tested and analyzed using the included test scripts. All scripts now use the central TOML configuration but also accept command-line overrides:

```bash
# Basic test with detailed chunk analysis
python -m raggiro.examples.scripts.test_semantic_chunking --input document.pdf --output test_output

# Test with custom queries and Ollama settings
python -m raggiro.examples.scripts.test_semantic_chunking --input document.pdf --queries "What is the main topic?" "Summarize key points" --ollama-url http://localhost:11434 --rewriting-model llama3 --generation-model mistral

# Specify number of chunks to retrieve for each query
python -m raggiro.examples.scripts.test_semantic_chunking --input document.pdf --top-k 5
```

### Comparing Chunking Strategies

You can compare different chunking strategies to find the most effective approach for your documents:

```bash
# Compare all available strategies (using TOML config for LLM settings)
python -m raggiro.examples.scripts.test_rag_comparison --input document.pdf 

# Compare only specific strategies with custom Ollama settings
python -m raggiro.examples.scripts.test_rag_comparison --input document.pdf --strategies size hybrid --ollama-url http://localhost:11434 --rewriting-model llama3 --generation-model mistral

# Test with specific queries and output directory
python -m raggiro.examples.scripts.test_rag_comparison --input document.pdf --queries "What is the main topic?" --output my_test_results
```

### Example Promptfoo Configurations

Raggiro includes several test prompt configurations for different use cases. All LLM configuration (Ollama URLs, model names, etc.) has been moved to the central TOML file. Test YAML files now only specify chunking strategies and test-specific settings.

#### Default English Configuration

The default test configuration in `test_prompts/` directory targets general document analysis:

```yaml
prompts:
  - "What is the main topic of this document?"
  - "Who is the author of this document?"
  - "Summarize the key points of this document."
  # More prompts...

variants:
  - name: "semantic_chunking"
    description: "Test with semantic chunking"
    config:
      chunking_strategy: "semantic"
      
  - name: "size_chunking"
    description: "Test with size-based chunking"
    config:
      chunking_strategy: "size"

tests:
  - description: "Basic information extraction"
    assert:
      - type: "contains-any"
        value: ["author", "document", "information"]
      
  - description: "Response quality"
    assert:
      - type: "contains-json"
        value: |
          {
            "min_length": 50
          }
      - type: "language-match"
        value: "en"
```

#### Italian Document Configuration

For Italian documents, use files in the `test_prompts/` directory:

```yaml
prompts:
  - "Qual è l'argomento principale di questo documento?"
  - "Chi è l'autore del documento?"
  - "Riassumi i punti chiave di questo documento."
  # More prompts...

variants:
  - name: "semantic_chunking"
    description: "Test con chunking semantico"
    config:
      chunking_strategy: "semantic"

tests:
  - description: "Estrazione informazioni di base"
    assert:
      - type: "language-match"
        value: "it"
```

#### Domain-Specific Configuration

For specialized content, Raggiro includes domain-specific test files like `test_prompts/kenny_werner.yaml` for music theory documents:

```yaml
prompts:
  - "Qual è il concetto principale di 'Effortless Mastery' secondo Kenny Werner?"
  - "Come descrive Kenny Werner il rapporto tra musicisti e la loro arte?"
  # More specialized prompts...

variants:
  - name: "semantic_chunking"
    description: "Test con chunking semantico"
    config:
      chunking_strategy: "semantic"
```

### Creating Custom Test Configurations

You can create custom test configurations for your specific documents:

```bash
# Test with custom configuration
python -m raggiro.testing.promptfoo_runner path/to/custom_prompts.yaml test_output

# Compare chunking strategies with domain-specific queries
python -m raggiro.examples.scripts.test_rag_comparison --input document.pdf --queries "Domain specific question 1" "Domain specific question 2"
```

### Sample Evaluation Results

```json
{
  "summary": {
    "pass": 18,
    "fail": 2,
    "total": 20,
    "pass_rate": 90.00
  },
  "prompts": [
    {
      "prompt": "What is the main topic of this document?",
      "results": [
        {
          "pass": true,
          "score": 0.92,
          "response": "The main topic of this document is artificial intelligence trends and their impact on business in 2023. The document specifically focuses on large language models, multimodal AI systems, and AI applications in healthcare. [Source: Summary section, paragraph 1]"
        }
      ]
    },
    {
      "prompt": "Who is the author of this document?",
      "results": [
        {
          "pass": true,
          "score": 0.89,
          "response": "The authors of this document are Dr. Maria Rodriguez, Dr. James Chen, and Alex Nowak from the Advanced Technologies Research Institute. [Source: Metadata section]"
        }
      ]
    }
  ]
}
```

### Programmatic Testing

You can also use the testing utilities programmatically:

```python
from raggiro.testing.promptfoo_runner import run_tests

# Run tests with custom prompt set and output directory
results = run_tests(
    prompt_file="test_prompts/kenny_werner.yaml", 
    output_dir="test_output",
    index_dir="index_directory"  # Optional path to vector index
)

# Check if PromptFoo is installed and tests succeeded
if not results["success"]:
    error = results.get("error", "Unknown error")
    if "promptfoo not installed" in error:
        print("PromptFoo is not installed. Install with: pip install promptfoo")
    else:
        print(f"Test failed: {error}")
else:
    print(f"Tests run: {results['tests_run']}")
    print(f"Results saved to: {results.get('output_file')}")
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