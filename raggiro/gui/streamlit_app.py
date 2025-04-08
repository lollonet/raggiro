#!/usr/bin/env python3
"""Streamlit-based GUI for Raggiro document processing."""

from raggiro.gui.imports_for_streamlit import *
from raggiro.gui.imports_for_streamlit import import_all

# Ensure all imports are available
import_all()

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