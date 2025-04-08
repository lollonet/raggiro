#!/bin/bash
# Script per configurare l'ambiente di sviluppo per Raggiro usando esclusivamente uv

set -e  # Esci immediatamente se un comando termina con uno stato diverso da zero

echo "==== Installazione ambiente di sviluppo Raggiro ===="

# Installa uv se non è già installato
if ! command -v uv &> /dev/null; then
    echo "Installazione di uv (gestore pacchetti e ambiente virtuale ultra-veloce)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Aggiorna PATH per il processo corrente
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo "uv installato correttamente."
else
    echo "uv è già installato: $(uv --version)"
fi

source $HOME/.local/bin/env

# Crea e attiva un ambiente virtuale con uv
echo "Creazione ambiente virtuale con uv..."
uv venv

# Installa dipendenze del progetto
echo "Installazione delle dipendenze di sviluppo con uv..."
uv pip install -e ".[dev]"

# Installa i modelli spacy
echo "Installazione dei modelli spaCy..."
python3 -m spacy download xx_sent_ud_sm
python3 -m spacy download it_core_news_sm
python3 -m spacy download en_core_web_sm

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
