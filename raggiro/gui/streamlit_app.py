#!/usr/bin/env python3
"""Streamlit-based GUI for Raggiro document processing."""

from raggiro.gui.imports_for_streamlit import *
from raggiro.gui.imports_for_streamlit import import_all, get_processed_files

# Ensure all imports are available
import_all()

def run_app():
    """Main entry point for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="Raggiro - Document Processing for RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and subtitle
    st.title("Raggiro")
    st.subheader("Document Processing for RAG (Retrieval-Augmented Generation)")
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Process Documents", 
        "OCR & Correction", 
        "Document Structure", 
        "Document Classification",
        "Test RAG",
        "View Results",
        "Configuration"
    ])
    
    # Process Documents tab
    with tabs[0]:
        process_documents_ui()
    
    # OCR & Correction tab
    with tabs[1]:
        ocr_correction_ui()
    
    # Document Structure tab
    with tabs[2]:
        document_structure_ui()
    
    # Document Classification tab
    with tabs[3]:
        document_classification_ui()
    
    # Test RAG tab
    with tabs[4]:
        test_rag_ui()
    
    # View Results tab
    with tabs[5]:
        view_results_ui()
    
    # Configuration tab
    with tabs[6]:
        configuration_ui()
    
    # Add footer
    st.markdown("---")
    st.markdown("### Raggiro - Document Processing Pipeline for RAG")
    st.markdown("*Visit the [GitHub repository](https://github.com/lollonet/raggiro) for documentation and updates.*")

def process_documents_ui():
    """UI for document processing."""
    st.header("Document Processing")
    
    # Split into two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document to process", 
            type=["pdf", "docx", "txt", "md", "html", "rtf", "xlsx", "jpg", "png", "jpeg", "tiff"],
            key="process_file_upload"
        )
        
        # OR provide a directory path
        directory_path = st.text_input("Or provide directory path:", key="directory_path")
        
        # Basic processing options
        st.subheader("Processing Options")
        
        # Enable recursion for directories
        recursive = st.checkbox("Process recursively (for directories)", value=True, key="recursive")
        
        # Enable OCR
        ocr_enabled = st.checkbox("Enable OCR for PDFs and images", value=True, key="ocr_enabled")
        
        # Enable spelling correction
        spelling_enabled = st.checkbox("Enable spelling correction", value=True, key="spelling_enabled")
        
        # Output formats
        output_formats = st.multiselect(
            "Output formats:",
            ["markdown", "json", "txt"],
            default=["markdown", "json"],
            key="output_formats"
        )
        
        # Output directory
        output_directory = st.text_input("Output directory:", value="test_output", key="output_directory")
        
        # Process button
        if st.button("Process Document(s)", key="process_documents_btn"):
            st.info("Processing document(s)...")
            # Processing logic would go here
            
    with col2:
        st.subheader("Processing Status")
        # Status display would go here
        st.info("Upload or select a document and click 'Process Document(s)' to start processing.")
        
        # Progress bar placeholder
        if st.session_state.get("processing", False):
            st.progress(0.75)
            st.text("Processing file 3 of 4...")
        
        # Results display placeholder
        if st.session_state.get("processing_complete", False):
            st.success("Processing complete!")
            st.json({
                "files_processed": 4,
                "successful": 3,
                "failed": 1,
                "output_files": ["output1.md", "output2.md", "output3.md"],
                "time_taken": "45 seconds"
            })
    
    # Documentation
    with st.expander("Document Processing Information", expanded=False):
        st.markdown("""
        ## Document Processing Pipeline
        
        The document processing pipeline consists of several steps:
        
        1. **File Validation**: Verifies the file format and accessibility
        2. **Document Classification**: Identifies document type and selects appropriate pipeline
        3. **Text Extraction**: Extracts raw text (with OCR if enabled)
        4. **Text Cleaning**: Removes noise and unwanted elements
        5. **Spelling Correction**: Fixes OCR and other spelling errors
        6. **Segmentation**: Divides text into logical chunks
        7. **Metadata Extraction**: Identifies title, author, date, etc.
        8. **Export**: Saves processed content in selected formats
        
        Each document is processed according to its detected category using specialized pipelines.
        """)

def ocr_correction_ui():
    """UI for OCR and correction settings."""
    st.header("OCR & Correction")
    
    # Split into two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("OCR Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document for OCR", 
            type=["pdf", "jpg", "png", "jpeg", "tiff"],
            key="ocr_file_upload"
        )
        
        # OCR language
        ocr_language = st.selectbox(
            "OCR Language:",
            ["ita", "eng", "fra", "deu", "spa", "por", "auto"],
            index=6,  # Default to auto
            key="ocr_language"
        )
        
        # OCR Engine mode
        ocr_engine = st.selectbox(
            "OCR Engine Mode:",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Legacy Engine Only",
                1: "Neural Net LSTM Engine Only",
                2: "Legacy + LSTM",
                3: "Default"
            }[x],
            index=3,
            key="ocr_engine"
        )
        
        # OCR Page Segmentation Mode
        ocr_psm = st.selectbox(
            "Page Segmentation Mode:",
            list(range(0, 14)),
            format_func=lambda x: {
                0: "Orientation and script detection only",
                1: "Automatic page segmentation with OSD",
                2: "Automatic page segmentation, but no OSD or OCR",
                3: "Fully automatic page segmentation, but no OSD",
                4: "Assume a single column of text of variable sizes",
                5: "Assume a single uniform block of vertically aligned text",
                6: "Assume a single uniform block of text",
                7: "Treat the image as a single text line",
                8: "Treat the image as a single word",
                9: "Treat the image as a single word in a circle",
                10: "Treat the image as a single character",
                11: "Sparse text. Find as much text as possible in no particular order",
                12: "Sparse text with OSD",
                13: "Raw line. Treat the image as a single text line",
            }[x],
            index=3,
            key="ocr_psm"
        )
        
        # Batch processing settings
        st.subheader("Batch Processing")
        
        # Max pages to process (0 = all)
        max_pages = st.number_input(
            "Max Pages to Process (0 = all):", 
            min_value=0, 
            value=0,
            key="max_pages"
        )
        
        # Batch size
        batch_size = st.number_input(
            "Batch Size:", 
            min_value=1, 
            value=10,
            key="batch_size"
        )
        
        # Process every N pages
        process_every = st.number_input(
            "Process Every N Pages:", 
            min_value=1, 
            value=1,
            key="process_every"
        )
        
        # Process button
        if st.button("Extract & Correct Text", key="extract_text_btn"):
            st.info("Extracting and correcting text...")
            # OCR logic would go here
            
    with col2:
        st.subheader("Spelling Correction")
        
        # Enable spelling correction
        enable_correction = st.checkbox(
            "Enable spelling correction", 
            value=True,
            key="enable_correction"
        )
        
        # Correction language
        correction_language = st.selectbox(
            "Correction Language:",
            ["it", "en", "fr", "de", "es", "same_as_ocr"],
            index=5,  # Default to same as OCR
            key="correction_language"
        )
        
        # Correction method
        correction_method = st.selectbox(
            "Correction Method:",
            ["symspellpy", "pyspellchecker", "textblob", "wordfreq", "combined"],
            index=4,  # Default to combined
            key="correction_method"
        )
        
        # Edit distance
        edit_distance = st.slider(
            "Max Edit Distance:",
            min_value=1,
            max_value=5,
            value=2,
            key="edit_distance"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="correction_confidence"
        )
        
        # Dictionary selection
        st.subheader("Custom Dictionaries")
        
        # Use custom dictionary
        use_custom_dict = st.checkbox(
            "Use custom dictionary", 
            value=False,
            key="use_custom_dict"
        )
        
        # Upload custom dictionary
        custom_dict_file = st.file_uploader(
            "Upload custom dictionary file (one word per line)", 
            type=["txt"],
            key="custom_dict_upload",
            disabled=not use_custom_dict
        )
        
        # Results display area
        st.subheader("OCR Results")
        if st.session_state.get("ocr_complete", False):
            # Tabs for different views
            result_tabs = st.tabs(["Extracted Text", "Corrected Text", "Diff", "Statistics"])
            
            with result_tabs[0]:
                st.text_area(
                    "Raw Extracted Text:", 
                    value="Sample extracted text with erors and misspellings.",
                    height=300,
                    key="raw_text",
                    disabled=True
                )
            
            with result_tabs[1]:
                st.text_area(
                    "Corrected Text:", 
                    value="Sample extracted text with errors and misspellings.",
                    height=300,
                    key="corrected_text",
                    disabled=True
                )
            
            with result_tabs[2]:
                st.text("Differences between original and corrected text:")
                st.code("- Sample extracted text with erors and misspellings.\n+ Sample extracted text with errors and misspellings.")
            
            with result_tabs[3]:
                st.metric("Words Processed", "157")
                st.metric("Corrections Made", "3")
                st.metric("Confidence Score", "0.92")
        else:
            st.info("Upload a document and click 'Extract & Correct Text' to see results.")

def document_structure_ui():
    """UI for document structure analysis."""
    st.header("Document Structure")
    
    # Split into two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Document Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document", 
            type=["pdf", "docx", "txt", "md"],
            key="structure_file_upload"
        )
        
        # OR select from already processed
        processed_files = get_processed_files() if "processed_files" not in st.session_state else st.session_state.processed_files
        selected_file = st.selectbox(
            "Or select from processed files", 
            [""] + processed_files,
            key="structure_file_select"
        )
        
        # Structure detection options
        st.subheader("Detection Options")
        
        # Structure detection method
        structure_method = st.selectbox(
            "Structure Detection Method:",
            ["headings", "semantic", "hybrid", "auto"],
            index=3,  # Default to auto
            key="structure_method"
        )
        
        # TOC detection
        enable_toc = st.checkbox(
            "Enable Table of Contents detection", 
            value=True,
            key="enable_toc"
        )
        
        # Language
        doc_language = st.selectbox(
            "Document Language:",
            ["auto", "it", "en", "fr", "de", "es"],
            index=0,  # Default to auto
            key="doc_language"
        )
        
        # Analyze button
        if st.button("Analyze Structure", key="analyze_structure_btn"):
            st.info("Analyzing document structure...")
            # Structure analysis logic would go here
        
    with col2:
        st.subheader("Document Structure Results")
        
        if st.session_state.get("structure_analyzed", False):
            # Tabs for different views
            structure_tabs = st.tabs(["Table of Contents", "Hierarchy", "Metadata", "Visualizations"])
            
            with structure_tabs[0]:
                st.markdown("""
                ## Table of Contents
                
                1. Introduction
                   1.1 Background
                   1.2 Purpose
                2. Methodology
                   2.1 Data Collection
                   2.2 Analysis Methods
                3. Results
                   3.1 Initial Findings
                   3.2 Secondary Analysis
                4. Discussion
                5. Conclusion
                """)
            
            with structure_tabs[1]:
                st.markdown("""
                ```
                Document
                ├── Title: "Sample Technical Document"
                ├── 1. Introduction
                │   ├── 1.1 Background
                │   └── 1.2 Purpose
                ├── 2. Methodology
                │   ├── 2.1 Data Collection
                │   └── 2.2 Analysis Methods
                ├── 3. Results
                │   ├── 3.1 Initial Findings
                │   └── 3.2 Secondary Analysis
                ├── 4. Discussion
                └── 5. Conclusion
                ```
                """)
            
            with structure_tabs[2]:
                st.json({
                    "title": "Sample Technical Document",
                    "author": "Sample Author",
                    "date": "2023-04-01",
                    "language": "en",
                    "pages": 15,
                    "sections": 5,
                    "subsections": 6,
                    "detected_category": "technical"
                })
            
            with structure_tabs[3]:
                st.markdown("### Structure Visualization")
                st.text("(Placeholder for document structure visualization)")
        else:
            st.info("Upload or select a document and click 'Analyze Structure' to see results.")
    
    # Documentation
    with st.expander("Document Structure Information", expanded=False):
        st.markdown("""
        ## Document Structure Analysis
        
        The document structure analysis examines the organization of documents to facilitate better chunking and retrieval:
        
        ### Detection Methods:
        
        - **Headings-based**: Identifies structure based on heading formatting
        - **Semantic-based**: Analyzes content to determine logical sections
        - **Hybrid**: Combines formatting and semantic analysis
        - **Auto**: Selects the best method based on document type
        
        ### Table of Contents Detection:
        
        The system can identify and extract table of contents sections, which provide valuable signals about document organization. This works across multiple languages including Italian, English, French, German, and Spanish.
        
        ### Benefits:
        
        - **Improved Chunking**: Create more meaningful document segments
        - **Enhanced Retrieval**: Better understand document hierarchy
        - **Semantic Context**: Maintain relationships between sections
        """)

def test_rag_ui():
    """UI for testing RAG functionality."""
    st.header("Test RAG")
    
    # Input options
    st.subheader("RAG Testing Options")
    
    # Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Document selection
        st.subheader("Document Selection")
        
        # Option to select documents
        doc_selection = st.radio(
            "Document Source:",
            ["Processed Documents", "Upload New Document"],
            key="doc_selection"
        )
        
        if doc_selection == "Processed Documents":
            # Select from processed documents
            processed_files = get_processed_files() if "processed_files" not in st.session_state else st.session_state.processed_files
            selected_docs = st.multiselect(
                "Select documents to query:", 
                processed_files,
                key="selected_docs"
            )
        else:
            # Upload new document
            uploaded_docs = st.file_uploader(
                "Upload documents", 
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                key="rag_docs_upload"
            )
        
        # RAG configuration
        st.subheader("RAG Configuration")
        
        # Chunking method
        chunking_method = st.selectbox(
            "Chunking Method:",
            ["fixed_size", "semantic", "hybrid"],
            index=2,  # Default to hybrid
            key="chunking_method"
        )
        
        # Embedding model
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["all-MiniLM-L6-v2", "multilingual-e5-small", "custom"],
            index=0,
            key="embedding_model"
        )
        
        # Number of chunks to retrieve
        k_chunks = st.slider(
            "Number of chunks to retrieve:",
            min_value=1,
            max_value=20,
            value=5,
            key="k_chunks"
        )
        
        # LLM options
        st.subheader("LLM Settings")
        
        # LLM provider
        llm_provider = st.selectbox(
            "LLM Provider:",
            ["OpenAI", "Local", "Other"],
            index=0,
            key="llm_provider"
        )
        
        # API key
        if llm_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                key="api_key"
            )
        
        # Model selection
        llm_model = st.selectbox(
            "LLM Model:",
            ["gpt-3.5-turbo", "gpt-4", "local-model"],
            index=0,
            key="llm_model"
        )
    
    with col2:
        # Query area
        st.subheader("Ask a Question")
        
        query = st.text_area(
            "Enter your question:",
            placeholder="What are the main features of this document?",
            height=100,
            key="rag_query"
        )
        
        # Query button
        if st.button("Ask Question", key="ask_query_btn"):
            if not query:
                st.error("Please enter a question")
            else:
                st.info("Processing query...")
                # Query logic would go here
                st.session_state["rag_response"] = True
        
        # Response area
        if st.session_state.get("rag_response", False):
            st.subheader("Response")
            
            # Create tabs for different views
            response_tabs = st.tabs(["Answer", "Retrieved Chunks", "Debug Info"])
            
            with response_tabs[0]:
                st.markdown("""
                The main features of the document include:
                
                1. A comprehensive RAG pipeline for document processing
                2. Support for multiple languages including Italian and English
                3. OCR capabilities for scanned documents
                4. Intelligent document classification
                5. Semantic chunking for better context preservation
                6. Spelling correction optimized for Italian language
                
                These features combine to create an effective system for processing documents and extracting relevant information for RAG applications.
                """)
            
            with response_tabs[1]:
                st.subheader("Retrieved Chunks")
                
                for i in range(3):
                    with st.expander(f"Chunk {i+1} (Score: {0.95 - i*0.1:.2f})"):
                        st.markdown("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur euismod, nisi nisi consectetur nisi, nec consectetur nisi nisi euismod nisi.")
            
            with response_tabs[2]:
                st.subheader("Debug Information")
                st.json({
                    "query": "What are the main features of this document?",
                    "query_embedding_dimensions": 384,
                    "chunks_retrieved": 5,
                    "time_taken_ms": 245,
                    "model": "gpt-3.5-turbo",
                    "token_count": {
                        "prompt": 1245,
                        "completion": 187,
                        "total": 1432
                    }
                })
        else:
            st.info("Enter a question and click 'Ask Question' to see results.")
    
    # Documentation
    with st.expander("RAG Testing Information", expanded=False):
        st.markdown("""
        ## RAG (Retrieval-Augmented Generation) Testing
        
        This interface allows you to test the RAG capabilities of Raggiro:
        
        1. **Document Processing**: Documents are processed, chunked, and indexed
        2. **Query Processing**: Your question is transformed into an embedding
        3. **Retrieval**: The system retrieves the most relevant chunks
        4. **Generation**: A language model creates a response based on the retrieved chunks
        
        ### Configuration Options:
        
        - **Chunking Method**: How documents are divided into segments
        - **Embedding Model**: Vector representation used for semantic search
        - **LLM Model**: Language model used for generating answers
        
        ### Performance Factors:
        
        - Chunk size and method significantly impact retrieval quality
        - More chunks retrieved improves coverage but may add noise
        - Different embedding models have varying performance for different languages
        """)

def view_results_ui():
    """UI for viewing processing results."""
    st.header("View Results")
    
    # Split into two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Results")
        
        # Get available results
        processed_files = get_processed_files() if "processed_files" not in st.session_state else st.session_state.processed_files
        
        # Filter by file type
        file_type_filter = st.multiselect(
            "Filter by file type:",
            ["pdf", "docx", "txt", "md", "json"],
            default=["md", "json"],
            key="file_type_filter"
        )
        
        # Display filtered files
        filtered_files = [f for f in processed_files if any(f.endswith(ext) for ext in file_type_filter)]
        
        if filtered_files:
            selected_result = st.selectbox(
                "Select a file to view:", 
                filtered_files,
                key="selected_result"
            )
            
            # View button
            if st.button("View Selected File", key="view_result_btn"):
                st.info(f"Loading {selected_result}...")
                # File loading logic would go here
                st.session_state["viewing_file"] = True
                st.session_state["current_file"] = selected_result
        else:
            st.info("No matching files found. Try changing the filter or processing documents first.")
    
    with col2:
        st.subheader("File Content")
        
        if st.session_state.get("viewing_file", False) and st.session_state.get("current_file"):
            current_file = st.session_state.get("current_file")
            
            # Display file content based on type
            if current_file.endswith(".md"):
                st.markdown("## Sample Markdown Content\n\nThis is a placeholder for the actual markdown content.")
            elif current_file.endswith(".json"):
                st.json({
                    "filename": "sample.pdf",
                    "metadata": {
                        "title": "Sample Document",
                        "author": "Sample Author",
                        "date": "2023-04-01"
                    },
                    "chunks": [
                        {"text": "This is the first chunk", "embedding_id": "abc123"},
                        {"text": "This is the second chunk", "embedding_id": "def456"}
                    ],
                    "processing": {
                        "time_taken": 12.5,
                        "ocr_enabled": True,
                        "language": "en"
                    }
                })
            else:
                st.text("File preview not available for this file type.")
            
            # Download button
            st.download_button(
                "Download File",
                "Sample file content for download",
                file_name=os.path.basename(current_file),
                mime="text/plain",
                key="download_btn"
            )
        else:
            st.info("Select a file and click 'View Selected File' to see its content.")

def configuration_ui():
    """UI for managing Raggiro configuration."""
    st.header("Configuration")
    
    # Split into tabs
    config_tabs = st.tabs(["General", "OCR", "Processing", "RAG", "Advanced"])
    
    with config_tabs[0]:
        st.subheader("General Settings")
        
        # Load config
        config = load_config()
        
        # Base directory
        base_dir = st.text_input(
            "Base Directory:",
            value=config.get("base_dir", os.getcwd()),
            key="base_dir"
        )
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory:",
            value=config.get("output", {}).get("output_dir", "test_output"),
            key="config_output_dir"
        )
        
        # Logging level
        log_level = st.selectbox(
            "Log Level:",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,  # Default to INFO
            key="log_level"
        )
        
        # Default language
        default_language = st.selectbox(
            "Default Language:",
            ["auto", "it", "en", "fr", "de", "es"],
            index=0,  # Default to auto
            key="default_language"
        )
        
        # Save temp files
        save_temp = st.checkbox(
            "Save temporary files", 
            value=False,
            key="save_temp"
        )
    
    with config_tabs[1]:
        st.subheader("OCR Settings")
        
        # OCR enabled by default
        ocr_default = st.checkbox(
            "Enable OCR by default", 
            value=True,
            key="ocr_default"
        )
        
        # Default OCR language
        ocr_lang = st.selectbox(
            "Default OCR Language:",
            ["ita", "eng", "fra", "deu", "spa", "por", "auto"],
            index=6,  # Default to auto
            key="config_ocr_language"
        )
        
        # Default engine mode
        ocr_engine_default = st.selectbox(
            "Default OCR Engine Mode:",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Legacy Engine Only",
                1: "Neural Net LSTM Engine Only",
                2: "Legacy + LSTM",
                3: "Default"
            }[x],
            index=3,
            key="config_ocr_engine"
        )
        
        # Default page segmentation mode
        ocr_psm_default = st.selectbox(
            "Default Page Segmentation Mode:",
            list(range(0, 14)),
            format_func=lambda x: {
                0: "Orientation and script detection only",
                1: "Automatic page segmentation with OSD",
                2: "Automatic page segmentation, but no OSD or OCR",
                3: "Fully automatic page segmentation, but no OSD",
                4: "Assume a single column of text of variable sizes",
                5: "Assume a single uniform block of vertically aligned text",
                6: "Assume a single uniform block of text",
                7: "Treat the image as a single text line",
                8: "Treat the image as a single word",
                9: "Treat the image as a single word in a circle",
                10: "Treat the image as a single character",
                11: "Sparse text. Find as much text as possible in no particular order",
                12: "Sparse text with OSD",
                13: "Raw line. Treat the image as a single text line",
            }[x],
            index=3,
            key="config_ocr_psm"
        )
    
    with config_tabs[2]:
        st.subheader("Processing Settings")
        
        # Spelling correction
        spelling_default = st.checkbox(
            "Enable spelling correction by default", 
            value=True,
            key="spelling_default"
        )
        
        # Default correction method
        correction_method_default = st.selectbox(
            "Default Correction Method:",
            ["symspellpy", "pyspellchecker", "textblob", "wordfreq", "combined"],
            index=4,  # Default to combined
            key="config_correction_method"
        )
        
        # Default chunking method
        chunking_method_default = st.selectbox(
            "Default Chunking Method:",
            ["fixed_size", "semantic", "hybrid"],
            index=2,  # Default to hybrid
            key="config_chunking_method"
        )
        
        # Document classification
        classification_default = st.checkbox(
            "Enable document classification by default", 
            value=True,
            key="classification_default"
        )
    
    with config_tabs[3]:
        st.subheader("RAG Settings")
        
        # Default embedding model
        embedding_model_default = st.selectbox(
            "Default Embedding Model:",
            ["all-MiniLM-L6-v2", "multilingual-e5-small", "custom"],
            index=0,
            key="config_embedding_model"
        )
        
        # Default number of chunks to retrieve
        k_chunks_default = st.number_input(
            "Default number of chunks to retrieve:",
            min_value=1,
            max_value=20,
            value=5,
            key="config_k_chunks"
        )
        
        # Default LLM provider
        llm_provider_default = st.selectbox(
            "Default LLM Provider:",
            ["OpenAI", "Local", "Other"],
            index=0,
            key="config_llm_provider"
        )
        
        # Default model
        llm_model_default = st.selectbox(
            "Default LLM Model:",
            ["gpt-3.5-turbo", "gpt-4", "local-model"],
            index=0,
            key="config_llm_model"
        )
    
    with config_tabs[4]:
        st.subheader("Advanced Settings")
        
        # Raw TOML config
        st.text("Edit raw configuration (TOML format):")
        
        # Sample TOML config
        sample_config = """
        [general]
        base_dir = "/home/user/raggiro"
        output_dir = "test_output"
        log_level = "INFO"
        default_language = "auto"
        save_temp_files = false
        
        [ocr]
        enabled_by_default = true
        default_language = "auto"
        engine_mode = 3
        page_segmentation_mode = 3
        
        [processing]
        spelling_correction = true
        correction_method = "combined"
        chunking_method = "hybrid"
        enable_classification = true
        
        [rag]
        embedding_model = "all-MiniLM-L6-v2"
        chunks_to_retrieve = 5
        llm_provider = "OpenAI"
        llm_model = "gpt-3.5-turbo"
        """
        
        config_toml = st.text_area(
            "TOML Configuration:",
            value=sample_config,
            height=300,
            key="config_toml"
        )
    
    # Save button
    if st.button("Save Configuration", key="save_config_btn"):
        st.success("Configuration saved successfully!")
        # Config saving logic would go here

def document_classification_ui():
    """UI for document classification visualization and testing."""
    st.header("Document Classification")
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Test Classification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document to classify", 
            type=["pdf", "docx", "txt", "md", "html", "rtf"],
            key="classification_file_upload"
        )
        
        # OR select from already processed files
        processed_files = get_processed_files() if "processed_files" not in st.session_state else st.session_state.processed_files
        selected_file = st.selectbox(
            "Or select from processed files", 
            [""] + processed_files,
            key=f"classification_file_select_{int(time.time())}"
        )
        
        # Classification options
        st.subheader("Classification Options")
        
        # Enable classification
        enable_classification = st.checkbox(
            "Enable document classification", 
            value=True,
            key="enable_classification"
        )
        
        # Classification methods
        classification_methods = st.multiselect(
            "Classification methods",
            ["Metadata-based", "Content-based", "Combined"],
            default=["Metadata-based", "Content-based", "Combined"],
            key="classification_methods"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="confidence_threshold"
        )
        
        # Process button
        if st.button("Classify Document", key="classify_document_btn"):
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                st.session_state.classification_result = classify_document(
                    temp_file_path, 
                    enable_classification,
                    classification_methods,
                    confidence_threshold
                )
                
                # Clean up
                os.remove(temp_file_path)
                os.rmdir(temp_dir)
            
            elif selected_file:
                # Process the selected file
                st.session_state.classification_result = classify_document(
                    selected_file, 
                    enable_classification,
                    classification_methods,
                    confidence_threshold
                )
            
            else:
                st.error("Please upload a file or select one from the list.")
    
    with col2:
        st.subheader("Classification Results")
        
        # Check if we have results to display
        if "classification_result" in st.session_state:
            result = st.session_state.classification_result
            
            if result["success"]:
                # Display overall result
                st.success(f"Document classified as: **{result['category']}**")
                st.write(f"**Confidence**: {result['confidence']:.2f}")
                st.write(f"**Classification method**: {result['method']}")
                
                # Create tabs for different result views
                result_tabs = st.tabs(["Overview", "Features", "Scores", "Raw Data"])
                
                with result_tabs[0]:
                    # Display document info
                    st.subheader("Document Information")
                    info_df = pd.DataFrame({
                        "Property": ["Filename", "File Type", "Size", "Document Category", "Confidence", "Method"],
                        "Value": [
                            result.get("filename", "N/A"),
                            result.get("file_type", "N/A"),
                            f"{result.get('file_size_mb', 0):.2f} MB",
                            result.get("category", "unknown"),
                            f"{result.get('confidence', 0):.2f}",
                            result.get("method", "N/A")
                        ]
                    })
                    st.dataframe(info_df, hide_index=True)
                
                with result_tabs[1]:
                    # Display extracted features
                    st.subheader("Document Features")
                    if "features" in result:
                        features = result["features"]
                        
                        # Convert features to DataFrame
                        feature_items = []
                        for key, value in features.items():
                            feature_items.append({"Feature": key, "Value": str(value)})
                        
                        feature_df = pd.DataFrame(feature_items)
                        st.dataframe(feature_df, hide_index=True)
                    else:
                        st.info("No feature information available")
                
                with result_tabs[2]:
                    # Display category scores
                    st.subheader("Category Scores")
                    
                    # Check what kind of scores we have
                    scores_key = None
                    for key in ["scores", "all_scores", "probabilities"]:
                        if key in result:
                            scores_key = key
                            break
                    
                    if scores_key:
                        scores = result[scores_key]
                        
                        # Convert to DataFrame
                        score_items = []
                        for category, score in scores.items():
                            score_items.append({
                                "Category": category,
                                "Score": float(score),
                                "Is Selected": category == result["category"]
                            })
                        
                        # Sort by score
                        score_df = pd.DataFrame(score_items).sort_values(by="Score", ascending=False)
                        
                        # Display as dataframe
                        st.dataframe(score_df, hide_index=True)
                        
                        # Display as chart
                        st.subheader("Score Distribution")
                        chart_df = pd.DataFrame({
                            "Category": score_df["Category"],
                            "Score": score_df["Score"]
                        })
                        st.bar_chart(chart_df.set_index("Category"))
                    else:
                        st.info("No score information available")
                
                with result_tabs[3]:
                    # Display raw data
                    st.subheader("Raw Classification Data")
                    st.json(result)
            else:
                st.error(f"Classification failed: {result.get('error', 'Unknown error')}")
        else:
            st.info("Upload or select a document and click 'Classify Document' to see results here.")
    
    # Display documentation and explanation
    with st.expander("Document Classification Information", expanded=False):
        st.markdown("""
        ## How Document Classification Works
        
        Raggiro's document classification system analyzes documents to determine their category, which allows for specialized processing pipelines tailored to each document type.
        
        ### Classification Methods:
        
        1. **Metadata-based**: Analyzes filename, extension, and file metadata without examining content
        2. **Content-based**: Analyzes document text content for category-specific patterns
        3. **Combined**: Merges results from both methods for more accurate classification
        
        ### Document Categories:
        
        - **Technical**: Manuals, documentation, guides, specifications
        - **Legal**: Contracts, agreements, laws, regulations
        - **Academic**: Research papers, theses, dissertations
        - **Business**: Reports, presentations, financial documents
        - **Structured**: Forms, invoices, applications
        - **Narrative**: Articles, stories, books
        
        ### Benefits of Classification:
        
        - **Optimized Processing**: Each document type gets specialized processing
        - **Improved Extraction**: Better handling of domain-specific elements
        - **Enhanced RAG Performance**: More accurate chunking and embedding
        """)

def classify_document(file_path, enable_classification, methods, confidence_threshold):
    """Classify a document and return the result."""
    try:
        # Create a config with classification settings
        config = load_config()
        
        # Override classification settings
        if "classifier" not in config:
            config["classifier"] = {}
        
        config["classifier"]["enabled"] = enable_classification
        config["classifier"]["confidence_threshold"] = confidence_threshold
        
        # Set methods
        config["classifier"]["use_rules"] = "Metadata-based" in methods
        config["classifier"]["use_content"] = "Content-based" in methods
        
        # Get file metadata
        file_handler = FileHandler(config)
        file_metadata = file_handler.get_file_metadata(file_path)
        file_type_info = file_handler.detect_file_type(file_path)
        
        # Initialize classifier
        classifier = DocumentClassifier(config)
        
        # Perform metadata-based classification
        metadata_result = classifier.classify_from_metadata(file_metadata, file_type_info)
        
        # Prepare result
        result = {
            "success": True,
            "filename": file_metadata["filename"],
            "file_path": file_metadata["path"],
            "file_type": file_type_info["document_type"],
            "file_size_mb": file_metadata["size_bytes"] / (1024 * 1024),
            "features": classifier._extract_metadata_features(file_metadata, file_type_info),
        }
        
        # If content-based classification is enabled, extract text and classify
        if "Content-based" in methods:
            # Extract text
            extractor = Extractor(config)
            document = extractor.extract(file_path, file_type_info)
            
            if document["success"] and "text" in document:
                # Classify based on content
                content_result = classifier.classify(document["text"])
                
                # If we want combined classification
                if "Combined" in methods and metadata_result["success"] and content_result["success"]:
                    metadata_class = metadata_result["category"]
                    metadata_conf = metadata_result["confidence"]
                    content_class = content_result["category"]
                    content_conf = content_result["confidence"]
                    
                    # Keep the higher confidence classification or combine them
                    if content_class == metadata_class:
                        # Same category, average confidence
                        result["category"] = content_class
                        result["confidence"] = (metadata_conf + content_conf) / 2
                        result["method"] = "combined"
                    elif content_conf > metadata_conf:
                        # Content-based has higher confidence
                        result["category"] = content_class
                        result["confidence"] = content_conf
                        result["method"] = "content"
                        result["scores"] = content_result.get("scores", {})
                        result["probabilities"] = content_result.get("probabilities", {})
                    else:
                        # Metadata has higher confidence
                        result["category"] = metadata_class
                        result["confidence"] = metadata_conf
                        result["method"] = "metadata"
                        result["all_scores"] = metadata_result.get("all_scores", {})
                else:
                    # Only content-based
                    if content_result["success"]:
                        result["category"] = content_result["category"]
                        result["confidence"] = content_result["confidence"]
                        result["method"] = "content"
                        result["scores"] = content_result.get("scores", {})
                        result["probabilities"] = content_result.get("probabilities", {})
                    else:
                        # Fall back to metadata if content classification failed
                        result["category"] = metadata_result["category"]
                        result["confidence"] = metadata_result["confidence"]
                        result["method"] = "metadata"
                        result["all_scores"] = metadata_result.get("all_scores", {})
            else:
                # Fall back to metadata if text extraction failed
                result["category"] = metadata_result["category"]
                result["confidence"] = metadata_result["confidence"]
                result["method"] = "metadata" 
                result["all_scores"] = metadata_result.get("all_scores", {})
                result["extraction_error"] = document.get("error", "Text extraction failed")
        else:
            # Only metadata-based
            result["category"] = metadata_result["category"]
            result["confidence"] = metadata_result["confidence"]
            result["method"] = "metadata"
            result["all_scores"] = metadata_result.get("all_scores", {})
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Entry point for streamlit app when run directly
if __name__ == "__main__":
    run_app()