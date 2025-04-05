# Raggiro

Advanced document processing pipeline for RAG applications

## Features

- **Comprehensive document support**: PDF (native and scanned), DOCX, TXT, HTML, RTF, XLSX, images with text
- **Advanced preprocessing**: Extraction, cleaning, normalization and logical segmentation
- **Metadata extraction**: Title, author, date, language, document type, etc.
- **Structured output**: Markdown and JSON formats with all metadata
- **Modular architecture**: CLI and minimal GUI interface
- **Fully offline**: Works without external API dependencies
- **RAG integration**: Ready for local semantic search pipelines

## Installation

```bash
# From the repository
git clone https://github.com/yourusername/raggiro.git
cd raggiro
pip install -e .

# Install optional development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Process a single file
raggiro process document.pdf --output output_dir

# Process a directory
raggiro process documents/ --output output_dir

# Enable detailed logging
raggiro process documents/ --output output_dir --log-level debug
```

### GUI Interface

```bash
raggiro gui
```

## Configuration

Create a configuration file at `~/.raggiro/config.toml` or specify a custom path with `--config`.

```toml
[processing]
dry_run = false
log_level = "info"
output_format = ["markdown", "json"]

[extraction]
ocr_enabled = true
ocr_language = "eng"

[segmentation]
use_spacy = true
spacy_model = "en_core_web_sm"
```

## Architecture

```
raggiro/
├── cli/          # Command-line interface
├── gui/          # GUI interface (Streamlit/Textual)
├── core/         # Core processing modules
│   ├── file_handler.py
│   ├── extractor.py
│   ├── cleaner.py
│   ├── segmenter.py
│   ├── metadata.py
│   ├── exporter.py
│   ├── logger.py
├── models/       # Classification models
├── utils/        # Utility functions
├── config/       # Configuration management
├── tests/        # Test suite
├── examples/     # Example inputs/outputs
```

## RAG Integration

Raggiro is designed to work seamlessly with local RAG (Retrieval-Augmented Generation) pipelines:

1. **Document Processing**: Extracts, cleans, and segments documents into logical chunks
2. **Semantic Indexing**: Output can be directly indexed in FAISS, Qdrant, or Milvus
3. **Query Rewriting**: Supports preprocessing user queries for better retrieval
4. **Local LLM Integration**: Works with Ollama, LLaMA, Mistral, etc.
5. **Response Generation**: Structured output enables source citation and metadata inclusion

## Testing

Tests are conducted using pytest and promptfoo for automated evaluation of RAG output quality.

```bash
# Run unit tests
pytest

# Run promptfoo evaluations
raggiro test-rag --prompt-set evaluation/prompts.yaml
```

## License

MIT
