"""Document classification models."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

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
        
        # Initialize model
        self.model = None
        self.vectorizer = None
        self.classes = []
        
        # Load model if path is provided
        if self.model_path:
            self.load_model(self.model_path)
    
    def classify(self, text: str) -> Dict:
        """Classify a document.
        
        Args:
            text: Document text
            
        Returns:
            Classification result
        """
        if not self.model or not self.vectorizer:
            return {
                "success": False,
                "error": "No model loaded",
            }
        
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
            
            return {
                "success": True,
                "category": prediction,
                "probabilities": probabilities,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
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