# Installazione

## Installazione di base

Raggiro utilizza [uv](https://github.com/astral-sh/uv) come gestore di pacchetti e ambienti virtuali ultra-veloce al posto di pip e virtualenv. uv offre prestazioni 10-100 volte superiori e una gestione più coerente delle dipendenze.

> **IMPORTANTE**: Per la migliore compatibilità con spaCy e i suoi modelli linguistici, si consiglia di usare Python 3.9, 3.10 o 3.11. Python 3.8 è supportato ma potrebbe causare problemi con i modelli spaCy, mentre Python 3.12+ potrebbe non essere completamente supportato da tutte le dipendenze.

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

# Crea un ambiente virtuale con uv (specifica una versione compatibile di Python)
# Usa Python 3.9, 3.10 o 3.11 per la migliore compatibilità con spaCy e i suoi modelli
uv venv --python=python3.11  # Cambia con python3.9 o python3.10 se preferisci

# Attiva l'ambiente virtuale (importante per i passaggi successivi)
# Su Linux/macOS:
source .venv/bin/activate
# Su Windows:
# .venv\Scripts\activate

# Installa il pacchetto in modalità sviluppo (senza dipendenze extra)
uv pip install -e .

# Installa streamlit per l'interfaccia grafica
uv pip install streamlit

# Download dei modelli linguistici (potrebbe richiedere alcuni minuti)
python -m spacy download xx_sent_ud_sm  # Modello multilingua (essenziale)
python -m spacy download it_core_news_sm  # Italiano (raccomandato)
python -m spacy download en_core_web_sm  # Inglese (raccomandato)
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

# Per tutte le dipendenze opzionali (ECCETTO SPACY - i modelli vanno installati separatamente)
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
  
  > **NOTA IMPORTANTE**: I modelli spaCy devono essere installati manualmente e potrebbero non funzionare con tutti i gestori di pacchetti. Si consiglia di usare Python 3.9-3.11 per la massima compatibilità.
  
  ```bash
  # IMPORTANTE: Assicurati di attivare l'ambiente virtuale prima di eseguire questi comandi
  
  # Modello multilingue (ESSENZIALE - il sistema richiede almeno questo modello)
  python -m spacy download xx_sent_ud_sm
  
  # Modelli specifici per lingua (RACCOMANDATI - migliorano significativamente le prestazioni)
  python -m spacy download it_core_news_sm  # Italiano
  python -m spacy download en_core_web_sm  # Inglese
  
  # Modelli linguistici aggiuntivi (OPZIONALI)
  python -m spacy download fr_core_news_sm  # Francese
  python -m spacy download es_core_news_sm  # Spagnolo
  
  # Verifica modelli installati
  python -m spacy info --all
  ```
  
  Se riscontri errori durante l'installazione dei modelli con `uv`, prova con questo metodo alternativo:
  
  ```python
  # Crea un file install_spacy_models.py con questo contenuto:
  import spacy
  import subprocess
  import sys
  
  def download_model(model_name):
      try:
          # Prova a scaricare con subprocess
          print(f"Installazione del modello {model_name}...")
          subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
          return True
      except:
          try:
              # Prova con pip come alternativa
              subprocess.run([sys.executable, "-m", "pip", "install", model_name], check=True)
              # Verifica se il modello è stato installato
              spacy.load(model_name)
              return True
          except:
              print(f"Errore nell'installazione di {model_name}")
              return False
  
  # Installa i modelli essenziali
  download_model("xx_sent_ud_sm")
  download_model("it_core_news_sm")
  download_model("en_core_web_sm")
  ```
  
  Per maggiori dettagli sull'utilizzo di spaCy, consulta la [documentazione specifica](spacy.md).

## Avvio dell'interfaccia GUI

Raggiro offre un'interfaccia grafica basata su Streamlit per un'interazione più semplice.

### Metodo 1: Avvio diretto (raccomandato)

```bash
# Assicurati che streamlit sia installato
uv pip install streamlit

# Avvia direttamente l'app streamlit
python -m streamlit run raggiro/gui/streamlit_app.py
```

### Metodo 2: Script di avvio

```bash
# Avvia tramite lo script launcher (richiede streamlit installato nel Python di sistema)
python scripts/gui/launch_gui.py
```

### Metodo 3: Avvio con uv (più robusto)

```bash
# Avvia tramite uv run (utilizzando l'ambiente virtuale di uv)
# Questo metodo è il più robusto quando ci sono conflitti di dipendenze
uv run python -m streamlit run raggiro/gui/streamlit_app.py
```

### Risoluzione problemi di compatibilità

Se riscontri errori di compatibilità tra uv, spaCy e i suoi modelli, prova questo approccio alternativo:

```bash
# Crea un ambiente virtuale Python standard
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Installa il pacchetto e streamlit
pip install -e .
pip install streamlit

# Installa modelli spaCy
python -m spacy download xx_sent_ud_sm
python -m spacy download it_core_news_sm

# Avvia l'app
python -m streamlit run raggiro/gui/streamlit_app.py
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