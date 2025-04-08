"""Document classification models."""

import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Document categories
DOCUMENT_CATEGORIES = {
    "technical": ["manual", "guide", "documentation", "specification", "technical"],
    "legal": ["contract", "agreement", "legal", "law", "regulation", "policy"],
    "academic": ["paper", "thesis", "dissertation", "research", "journal", "academic"],
    "business": ["report", "presentation", "financial", "business", "corporate"],
    "structured": ["form", "invoice", "cv", "resume", "application", "questionnaire"],
    "narrative": ["article", "story", "book", "novel", "blog"]
}

class DocumentClassifier:
    """Classifies documents into categories."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the document classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configure classifier settings
        classifier_config = self.config.get("classifier", {})
        self.model_type = classifier_config.get("model_type", "tfidf_svm")
        self.model_path = classifier_config.get("model_path", "")
        self.rule_based = classifier_config.get("use_rules", True)
        self.content_based = classifier_config.get("use_content", True)
        self.confidence_threshold = classifier_config.get("confidence_threshold", 0.6)
        
        # Initialize model
        self.model = None
        self.vectorizer = None
        self.classes = []
        
        # Load model if path is provided
        if self.model_path:
            self.load_model(self.model_path)
    
    def classify_from_metadata(self, file_metadata: Dict, file_type_info: Dict) -> Dict:
        """Classify a document based on its metadata.
        
        Args:
            file_metadata: File metadata from FileHandler
            file_type_info: File type information from FileHandler
            
        Returns:
            Classification result
        """
        if not self.rule_based:
            return {
                "success": False,
                "error": "Rule-based classification is disabled",
            }
        
        try:
            # Initialize result
            result = {
                "success": True,
                "method": "metadata",
                "confidence": 0.0,
                "category": "unknown",
                "features": {},
            }
            
            # Extract features from metadata
            features = self._extract_metadata_features(file_metadata, file_type_info)
            result["features"] = features
            
            # Use rules to determine document category
            category_scores = {}
            
            # Score each category based on keywords in filename and MIME type
            filename = file_metadata.get("filename", "").lower()
            description = file_type_info.get("description", "").lower()
            
            for category, keywords in DOCUMENT_CATEGORIES.items():
                score = 0
                
                # Check filename
                for keyword in keywords:
                    if keyword in filename:
                        score += 2
                
                # Check file description
                for keyword in keywords:
                    if keyword in description:
                        score += 1
                
                # Calculate normalized score (0-1)
                if score > 0:
                    category_scores[category] = min(score / 5.0, 1.0)
            
            # Find the category with the highest score
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                result["category"] = best_category[0]
                result["confidence"] = best_category[1]
                result["all_scores"] = category_scores
                
                # Only return a category if confidence is high enough
                if result["confidence"] < self.confidence_threshold:
                    result["category"] = "unknown"
            
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def classify(self, text: str) -> Dict:
        """Classify a document based on its content.
        
        Args:
            text: Document text
            
        Returns:
            Classification result
        """
        if not self.content_based:
            return {
                "success": False,
                "error": "Content-based classification is disabled",
            }
        
        if not self.model or not self.vectorizer:
            # Use rule-based classification for text if model is not available
            return self._classify_by_rules(text)
        
        try:
            # Vectorize the text
            features = self.vectorizer.transform([text])
            
            # Predict the class
            prediction = self.model.predict(features)[0]
            
            # Get class probabilities if available
            probabilities = {}
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                probabilities = {class_name: float(prob) for class_name, prob in zip(self.classes, proba)}
                
                # Find the highest probability
                best_prob = max(probabilities.values())
                
                # Only use the prediction if confidence is high enough
                if best_prob < self.confidence_threshold:
                    prediction = "unknown"
            
            return {
                "success": True,
                "method": "model",
                "category": prediction,
                "confidence": probabilities.get(prediction, 0.0) if probabilities else 0.8,
                "probabilities": probabilities,
            }
        except Exception as e:
            # Fall back to rule-based classification on failure
            rule_result = self._classify_by_rules(text)
            if rule_result["success"]:
                return rule_result
                
            return {
                "success": False,
                "error": str(e),
            }
    
    def _classify_by_rules(self, text: str) -> Dict:
        """Rule-based classification when no model is available.
        
        Args:
            text: Document text
            
        Returns:
            Classification result
        """
        result = {
            "success": True,
            "method": "rule",
            "category": "unknown",
            "confidence": 0.0,
            "scores": {},
        }
        
        # Use a sample of the text (beginning, middle, and end)
        text = text.lower()
        text_length = len(text)
        
        sample_size = min(5000, text_length // 3)
        
        # Beginning, middle, and end samples
        beginning = text[:sample_size]
        middle_start = max(0, (text_length // 2) - (sample_size // 2))
        middle = text[middle_start:middle_start + sample_size]
        end = text[max(0, text_length - sample_size):]
        
        sample_text = f"{beginning} {middle} {end}"
        
        # Score each category based on keyword frequency
        category_scores = {}
        
        for category, keywords in DOCUMENT_CATEGORIES.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', sample_text))
                score += count
            
            # Normalize score (0-1)
            if score > 0:
                normalized_score = min(score / 20.0, 1.0)  # Cap at 1.0
                category_scores[category] = normalized_score
        
        # Find the category with the highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            result["category"] = best_category[0]
            result["confidence"] = best_category[1]
            result["scores"] = category_scores
            
            # Only return a category if confidence is high enough
            if result["confidence"] < self.confidence_threshold:
                result["category"] = "unknown"
        
        return result
        
    def _extract_metadata_features(self, file_metadata: Dict, file_type_info: Dict) -> Dict:
        """Extract features from file metadata for classification.
        
        Args:
            file_metadata: File metadata from FileHandler
            file_type_info: File type information from FileHandler
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract relevant features
        if "filename" in file_metadata:
            features["filename"] = file_metadata["filename"]
        
        if "extension" in file_metadata:
            features["extension"] = file_metadata["extension"]
            
        if "size_bytes" in file_metadata:
            size_mb = file_metadata["size_bytes"] / (1024 * 1024)
            features["size_mb"] = size_mb
            
            # Categorize by size
            if size_mb < 0.1:
                features["size_category"] = "very_small"
            elif size_mb < 1:
                features["size_category"] = "small"
            elif size_mb < 10:
                features["size_category"] = "medium"
            elif size_mb < 50:
                features["size_category"] = "large"
            else:
                features["size_category"] = "very_large"
        
        if "mime_type" in file_type_info:
            features["mime_type"] = file_type_info["mime_type"]
        
        if "description" in file_type_info:
            features["description"] = file_type_info["description"]
        
        if "document_type" in file_type_info:
            features["document_type"] = file_type_info["document_type"]
        
        return features
    
    def load_model(self, model_path: Union[str, Path]) -> Dict:
        """Load a trained model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Loading result
        """
        model_path = Path(model_path)
        
        # Check if the directory exists
        if not model_path.exists():
            return {
                "success": False,
                "error": f"Model path does not exist: {model_path}",
            }
        
        try:
            # Load the model
            with open(model_path / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            # Load the vectorizer
            with open(model_path / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load the classes
            with open(model_path / "classes.pkl", "rb") as f:
                self.classes = pickle.load(f)
            
            return {
                "success": True,
                "model_type": self.model_type,
                "classes": self.classes,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load model: {str(e)}",
            }
    
    def train(self, texts: List[str], labels: List[str], output_path: Union[str, Path]) -> Dict:
        """Train a new classification model.
        
        Args:
            texts: List of document texts
            labels: List of document labels
            output_path: Path to save the model
            
        Returns:
            Training result
        """
        output_path = Path(output_path)
        
        # Check if inputs are valid
        if len(texts) != len(labels):
            return {
                "success": False,
                "error": "Number of texts and labels must match",
            }
        
        if not texts:
            return {
                "success": False,
                "error": "No training data provided",
            }
        
        try:
            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Train based on model type
            if self.model_type == "tfidf_svm":
                result = self._train_tfidf_svm(texts, labels, output_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported model type: {self.model_type}",
                }
            
            # Save model type and classes
            with open(output_path / "classes.pkl", "wb") as f:
                pickle.dump(self.classes, f)
            
            with open(output_path / "model_info.txt", "w") as f:
                f.write(f"Model type: {self.model_type}\n")
                f.write(f"Classes: {', '.join(self.classes)}\n")
                f.write(f"Training samples: {len(texts)}\n")
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to train model: {str(e)}",
            }
    
    def _train_tfidf_svm(self, texts: List[str], labels: List[str], output_path: Path) -> Dict:
        """Train a TF-IDF + SVM model.
        
        Args:
            texts: List of document texts
            labels: List of document labels
            output_path: Path to save the model
            
        Returns:
            Training result
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import classification_report
            
            # Get unique classes
            self.classes = sorted(list(set(labels)))
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
            )
            
            # Create SVM classifier
            self.model = SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
            )
            
            # Train the vectorizer
            X = self.vectorizer.fit_transform(texts)
            
            # Train the model
            self.model.fit(X, labels)
            
            # Save the model and vectorizer
            with open(output_path / "model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            
            with open(output_path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            # Evaluate the model
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            X_train_vec = self.vectorizer.transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            self.model.fit(X_train_vec, y_train)
            y_pred = self.model.predict(X_test_vec)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Cross-validation
            pipeline = Pipeline([
                ("vectorizer", self.vectorizer),
                ("classifier", self.model),
            ])
            
            cv_scores = cross_val_score(pipeline, texts, labels, cv=5)
            
            return {
                "success": True,
                "accuracy": float(report["accuracy"]),
                "cv_scores": [float(score) for score in cv_scores],
                "cv_mean_score": float(cv_scores.mean()),
                "classes": self.classes,
                "model_path": str(output_path),
            }
        except ImportError:
            return {
                "success": False,
                "error": "scikit-learn is not installed",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }