# Installazione

## Installazione di base

Raggiro utilizza [uv](https://github.com/astral-sh/uv) come gestore di pacchetti e ambienti virtuali ultra-veloce al posto di pip e virtualenv. uv offre prestazioni 10-100 volte superiori e una gestione più coerente delle dipendenze.

### Metodo 1: Installazione automatica (raccomandato)

```bash
# Clona il repository
git clone https://github.com/lollonet/raggiro.git
cd raggiro

# Esegui lo script di setup che configura tutto automaticamente
./scripts/installation/setup_dev_env.sh
```

### Metodo 2: Installazione manuale

```bash
# Clona il repository
git clone https://github.com/lollonet/raggiro.git
cd raggiro

# Installa uv (ultra-veloce gestore pacchetti e ambiente virtuale)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crea un ambiente virtuale con uv
uv venv

# Installa il pacchetto in modalità sviluppo
uv pip install -e .

# Download del modello linguistico multilingue
python -m spacy download xx_sent_ud_sm
```

## Installazione con dipendenze opzionali

```bash
# Per strumenti di sviluppo (ruff, black, isort, mypy, pre-commit...)
uv pip install -e ".[dev]"

# Per supporto database vettoriale
uv pip install -e ".[qdrant]"

# Per capacità di testing
uv pip install -e ".[test]"

# Per documentazione
uv pip install -e ".[docs]"

# Per tutte le dipendenze opzionali
uv pip install -e ".[dev,qdrant,test,docs]"
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

- **Modelli linguistici spaCy**: Richiede modelli specifici per l'elaborazione linguistica
  ```bash
  # Modello multilingue (raccomandato, supporta tutte le principali lingue europee)
  python -m spacy download xx_sent_ud_sm
  
  # Modelli specifici per lingua (opzionali, per prestazioni migliori)
  python -m spacy download it_core_news_sm  # Italiano
  python -m spacy download en_core_web_sm  # Inglese
  python -m spacy download fr_core_news_sm  # Francese
  python -m spacy download de_core_news_sm  # Tedesco
  python -m spacy download es_core_news_sm  # Spagnolo
  
  # Verifica modelli installati
  python -m spacy info --all
  ```
  
  Per maggiori dettagli sull'utilizzo di spaCy, consulta la [documentazione specifica](spacy.md).

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