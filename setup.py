"""Setup script for Raggiro."""

from setuptools import setup, find_packages

setup(
    name="raggiro",
    version="0.1.0",
    description="Advanced document processing pipeline for RAG applications",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/raggiro",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pymupdf",
        "pdfminer.six",
        "python-docx",
        "pytesseract",
        "openpyxl",
        "pandas",
        "beautifulsoup4",
        "langdetect",
        "spacy",
        "dateparser",
        "click",
        "streamlit",
        "textual",
        "chardet",
        "faiss-cpu",
        "sentence-transformers",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
            "ruff",
        ],
        "qdrant": [
            "qdrant-client",
        ],
        "test": [
            "promptfoo",
        ],
    },
    entry_points={
        "console_scripts": [
            "raggiro=raggiro.cli.main:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)