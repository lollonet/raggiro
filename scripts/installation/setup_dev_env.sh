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

# Assicurati che spacy sia correttamente installato
echo "Verifico installazione di spaCy..."
uv pip install spacy

# Installa i modelli spacy
echo "Installazione dei modelli spaCy..."

# Crea uno script temporaneo per installare i modelli spaCy
cat > /tmp/spacy_model_download.py << 'EOF'
import spacy
import subprocess
import sys
import os

def download_model(model_name):
    print(f"Tentativo di installare modello {model_name}...")
    try:
        # Prova a scaricare direttamente con subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
        print(f"Modello {model_name} installato correttamente.")
        return True
    except subprocess.CalledProcessError:
        print(f"Errore nell'installazione di {model_name} con spacy download.")
        try:
            # Prova con pip come fallback
            subprocess.run([sys.executable, "-m", "pip", "install", model_name], check=True)
            # Verifica se il modello è stato effettivamente installato
            try:
                spacy.load(model_name)
                print(f"Modello {model_name} installato correttamente con pip.")
                return True
            except OSError:
                print(f"Installazione di {model_name} fallita anche con pip.")
                return False
        except subprocess.CalledProcessError:
            print(f"Fallimento completo nell'installazione di {model_name}.")
            return False

# Lista dei modelli da installare in ordine di priorità
models = [
    "xx_sent_ud_sm",  # Multilingua (priorità)
    "it_core_news_sm",  # Italiano
    "en_core_web_sm",  # Inglese
    # Gli altri modelli possono essere installati manualmente se necessario
]

# Installa i modelli prioritari
for model in models:
    success = download_model(model)
    if not model.startswith("xx_") and not success:
        print(f"ATTENZIONE: Impossibile installare {model}, ma il progetto potrà funzionare con funzionalità ridotte.")
EOF

# Esegui lo script di installazione dei modelli
uv run python /tmp/spacy_model_download.py

# Pulisci
rm /tmp/spacy_model_download.py

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
