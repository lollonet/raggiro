# Installazione

## Installazione di base

```bash
# Da PyPI (non ancora disponibile)
pip install raggiro

# Dal repository con pip
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -e .

# Dal repository con requirements.txt
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Download modello linguistico
```

## Installazione con dipendenze opzionali

```bash
# Per strumenti di sviluppo
pip install -e ".[dev]"
# oppure
pip install -r requirements-dev.txt

# Per supporto database vettoriale
pip install -e ".[qdrant]"

# Per capacità di testing (compreso PromptFoo)
pip install -e ".[test]"

# Per tutte le dipendenze opzionali
pip install -e ".[dev,qdrant,test]"
```

## Requisiti di sistema

Alcune funzionalità richiedono dipendenze di sistema aggiuntive:

- **OCR**: Richiede Tesseract OCR
  ```bash
  # Ubuntu/Debian
  sudo apt install tesseract-ocr
  sudo apt install tesseract-ocr-ita tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa  # Lingue aggiuntive
  
  # macOS
  brew install tesseract
  brew install tesseract-lang  # Per supporto multilingua
  ```

- **Elaborazione PDF**: Alcune operazioni PDF potrebbero richiedere Poppler
  ```bash
  # Ubuntu/Debian
  sudo apt install poppler-utils
  
  # macOS
  brew install poppler
  ```

## Dipendenze di testing

Raggiro utilizza [PromptFoo](https://www.promptfoo.dev/) per la valutazione avanzata e il testing di RAG. Questo consente di confrontare diverse strategie di chunking, valutare la qualità delle risposte e generare metriche sulle prestazioni del sistema RAG.

### Installazione di PromptFoo

PromptFoo è un'applicazione Node.js che deve essere installata tramite npm. Per utilizzare tutte le funzionalità di testing:

```bash
# Assicurarsi che npm sia installato prima
npm install -g promptfoo
```

Puoi anche utilizzare lo script di installazione fornito:

```bash
# Eseguire lo script di installazione
chmod +x install_promptfoo.sh
./install_promptfoo.sh
```

Se ricevi un errore "command not found" dopo l'installazione, potrebbe essere necessario aggiungere la directory bin globale di npm al tuo PATH:

```bash
# Aggiungi questo al tuo .bashrc o .zshrc
export PATH="$(npm config get prefix)/bin:$PATH"
```

Se PromptFoo non è installato, l'interfaccia Streamlit mostrerà un messaggio di errore con le istruzioni di installazione. I test di base funzioneranno comunque, ma le funzionalità di valutazione avanzate saranno limitate.