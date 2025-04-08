#!/bin/bash
# Script per configurare l'ambiente di sviluppo per Raggiro

set -e  # Esci immediatamente se un comando termina con uno stato diverso da zero

echo "==== Installazione ambiente di sviluppo Raggiro ===="

# Verifica se l'ambiente virtuale esiste
if [ ! -d "venv" ]; then
    echo "Creazione ambiente virtuale Python..."
    python3 -m venv venv
fi

# Attiva l'ambiente virtuale
source venv/bin/activate

echo "Ambiente virtuale attivato."

# Installa uv
echo "Installazione di uv (gestore pacchetti ultra-veloce)..."
pip install --upgrade pip
pip install uv

# Installa dipendenze del progetto
echo "Installazione delle dipendenze di sviluppo con uv..."
uv pip install -e ".[dev]"

# Installa i modelli spacy
echo "Installazione dei modelli spaCy..."
python -m spacy download xx_sent_ud_sm
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm

# Configura pre-commit
echo "Configurazione pre-commit hooks..."
pre-commit install

# Verifica pre-commit
echo "Verifica pre-commit hooks..."
pre-commit run --all-files || echo "Alcuni pre-commit hook hanno fallito, ma continueremo."

# Verifica tesseract
echo "Verifica installazione Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "ATTENZIONE: Tesseract OCR non sembra essere installato."
    echo "Per installarlo:"
    echo "  Ubuntu/Debian: sudo apt-get install -y tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng"
    echo "  macOS: brew install tesseract tesseract-lang"
    echo "  Windows: Scarica da https://github.com/UB-Mannheim/tesseract/wiki"
else
    echo "Tesseract OCR installato: $(tesseract --version | head -1)"
    echo "Lingue disponibili: $(tesseract --list-langs)"
fi

echo ""
echo "==== Setup completato con successo! ===="
echo "Ambiente di sviluppo Raggiro pronto all'uso."
echo "Per eseguire il test del sistema: pytest"
echo "Per avviare l'interfaccia GUI: python scripts/gui/launch_gui.py"