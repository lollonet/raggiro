# Raggiro

Pipeline avanzata per l'elaborazione di documenti per applicazioni RAG (Retrieval-Augmented Generation)

![Raggiro](https://img.shields.io/badge/Raggiro-v0.1.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## Panoramica

Raggiro è un framework completo per l'elaborazione di documenti progettato per costruire sistemi RAG locali e offline. Fornisce capacità end-to-end dall'ingestione dei documenti alla generazione di risposte, con un focus su componenti modulari e configurabili che possono essere utilizzati insieme o indipendentemente.

## Caratteristiche

- **Supporto documenti completo**: PDF (nativi e scansionati), DOCX, TXT, HTML, RTF, XLSX, immagini con testo
- **Preprocessing avanzato**: Estrazione, pulizia, normalizzazione e segmentazione logica
- **Estrazione OCR ottimizzata**: Supporto batch per documenti grandi con rilevamento automatico della lingua
- **Correzione ortografica intelligente**: Miglioramento automatico della qualità del testo OCR con dizionari standard multilingua (italiano, inglese, francese, tedesco, spagnolo)
- **Chunking semantico adattivo**: Divisione intelligente del contenuto basata sul significato con ottimizzazioni per documenti OCR
- **Metadati arricchiti per i chunk**: Sommari automatici generati per ogni chunk che ne sintetizzano il contenuto in modo estrattivo
- **Estrazione metadata**: Titolo, autore, data, lingua, tipo di documento, rilevamento categoria, tabella dei contenuti
- **Output strutturato**: Formati Markdown e JSON con tutti i metadata
- **Classificazione intelligente dei documenti**: Sistema che rileva automaticamente la categoria del documento (tecnico, legale, accademico, ecc.) e applica pipeline specializzate
- **Rilevamento avanzato della tabella dei contenuti**: Supporto per tutte le lingue europee e estrazione da PDF nativo
- **Interfaccia GUI dedicata**: Sezioni specializzate per OCR, classificazione documenti e correzione ortografica/semantica
- **Funzionamento completamente offline**: Funziona senza dipendenze API esterne
- **Pipeline RAG completa**: Indicizzazione vettoriale, recupero e generazione di risposte integrati
- **Utilità di testing**: Strumenti per il benchmarking e confronto tra strategie di chunking
- **Supporto multilingua**: Rilevamento automatico della lingua e mantenimento della coerenza linguistica
- **Visualizzazione avanzata**: Dashboard per analizzare la qualità del chunking e della correzione ortografica

## Requisiti di sistema

### Software Python

Il progetto richiede Python 3.8 o superiore con le seguenti librerie principali:

- **Elaborazione PDF**: pymupdf, pdfminer.six
- **OCR**: pytesseract
- **Elaborazione documenti**: python-docx, openpyxl, pandas, beautifulsoup4
- **NLP e analisi testo**: spacy, langdetect, symspellpy, textblob
- **RAG**: sentence-transformers, faiss-cpu, qdrant-client
- **GUI**: streamlit, textual

### Programmi esterni richiesti

1. **Tesseract OCR** - Richiesto per OCR su PDF e immagini
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng`
   - **macOS**: `brew install tesseract tesseract-lang`
   - **Windows**: Scarica e installa da [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. **Poppler** - Richiesto per l'elaborazione PDF
   - **Ubuntu/Debian**: `sudo apt-get install poppler-utils`
   - **macOS**: `brew install poppler`
   - **Windows**: Scarica da [poppler-windows](https://github.com/oschwartz10612/poppler-windows)

3. **Libmagic** - Per il rilevamento dei tipi MIME
   - **Ubuntu/Debian**: `sudo apt-get install libmagic-dev`
   - **macOS**: `brew install libmagic`
   - **Windows**: Installato automaticamente con python-magic-bin

4. **Modelli linguistici Spacy e pacchetti di correzione ortografica**:
   - Modello multilingue (supporta più lingue contemporaneamente):
     - `uv run python -m spacy download xx_sent_ud_sm`
   - Modelli linguistici specifici per singole lingue:
     - `uv run python -m spacy download it_core_news_sm` (italiano)
     - `uv run python -m spacy download en_core_web_sm` (inglese)
     - `uv run python -m spacy download fr_core_news_sm` (francese)
     - `uv run python -m spacy download de_core_news_sm` (tedesco)
     - `uv run python -m spacy download es_core_news_sm` (spagnolo)
     - `uv run python -m spacy download pt_core_news_sm` (portoghese)
     - `uv run python -m spacy download nl_core_news_sm` (olandese)
   - `uv pip install pyspellchecker` (per dizionari di correzione ortografica standard)

## Documentazione

Raggiro include una documentazione completa divisa nelle seguenti sezioni:

- [Installazione](docs/installation.md) - Guida all'installazione e requisiti di sistema
- [Configurazione](docs/configuration.md) - Configurazione TOML e opzioni personalizzabili
- [Riferimento CLI](docs/commands.md) - Comandi e opzioni della riga di comando
- [Interfacce GUI](docs/gui.md) - Utilizzo delle interfacce Streamlit e Textual
- [Classificazione Documenti](docs/document_classification.md) - Sistema di classificazione intelligente dei documenti
- [Pipeline RAG](docs/rag.md) - Componenti e utilizzo della pipeline RAG
- [Testing e valutazione](docs/testing.md) - Strumenti per testare e valutare il sistema RAG
- [Riferimento API](docs/api.md) - Documentazione dell'API Python per sviluppatori

## Installazione

### 1. Installare programmi esterni richiesti

Prima di installare Raggiro, assicurati di aver installato tutti i programmi esterni richiesti:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-deu poppler-utils libmagic-dev

# Per macOS usando Homebrew
brew install tesseract tesseract-lang poppler libmagic
```

### 2. Clonare e installare Raggiro

```bash
# Clonare il repository
git clone https://github.com/lollonet/raggiro.git
cd raggiro

# Metodo 1: Installazione rapida (raccomandato per sviluppatori)
./scripts/installation/setup_dev_env.sh

# Metodo 2: Installazione manuale

# Installare uv (gestore pacchetti e ambiente virtuale ultra-veloce)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Creare un ambiente virtuale con uv
uv venv

# Installare dipendenze Python (estremamente più veloce di pip)
uv pip install -e .
# Per installare anche le dipendenze di sviluppo
uv pip install -e ".[dev]"
# Per installare le dipendenze per la documentazione
uv pip install -e ".[docs]"

# Installare i modelli linguistici richiesti
uv run python -m spacy download xx_sent_ud_sm  # Modello multilingue base
uv run python -m spacy download it_core_news_sm  # Italiano
uv run python -m spacy download en_core_web_sm  # Inglese
# Altri modelli linguistici opzionali:
# uv run python -m spacy download fr_core_news_sm  # Francese
# uv run python -m spacy download de_core_news_sm  # Tedesco
# uv run python -m spacy download es_core_news_sm  # Spagnolo

# Installare le risorse NLTK necessarie per la generazione dei sommari
bash scripts/installation/install_nltk.sh
```

### 3. Configurare pre-commit hooks (per sviluppatori)

Per garantire la qualità del codice, il progetto utilizza pre-commit hooks:

```bash
# Installa pre-commit se non è già installato
uv pip install pre-commit

# Configura i pre-commit hooks
pre-commit install

# Verifica l'installazione
pre-commit run --all-files
```

### 4. Verifica installazione Tesseract

Raggiro usa Tesseract per OCR. Verifica che sia installato correttamente:

```bash
# Controlla versione tesseract
tesseract --version

# Verifica lingue disponibili
tesseract --list-langs
```

## Avvio rapido

```bash
# Elabora un documento con OCR
raggiro process document.pdf --ocr --output output_dir

# Avvia l'interfaccia GUI Streamlit
python launch_gui.py
# oppure
bash run_streamlit.sh
```

### Interfaccia Grafica (GUI)

L'interfaccia utente Streamlit offre diverse schede funzionali:

1. **Process Documents**: Elaborazione base dei documenti
2. **OCR & Correction**: OCR e correzione ortografica avanzata 
3. **Document Structure**: Visualizzazione della struttura e tabella dei contenuti
4. **Document Classification**: Classificazione intelligente dei documenti
5. **Test RAG**: Test e valutazione di query RAG
6. **View Results**: Visualizzazione dei risultati dell'elaborazione
7. **Configuration**: Configurazione del sistema

![Diagramma del flusso di classificazione](docs/images/document_classification_flow.md)

### Analisi della Tabella dei Contenuti

La scheda "Document Structure" consente di visualizzare e analizzare la struttura del documento:

1. Carica un documento PDF o seleziona un file già elaborato
2. Visualizza la tabella dei contenuti con supporto multilingua
3. Esplora i metadati del documento e i riepiloghi dei chunk
4. Analizza la struttura gerarchica del documento con visualizzazioni

### Nota importante per l'OCR di documenti multi-pagina

Per elaborare correttamente documenti PDF con molte pagine:
1. Nella scheda "OCR & Correction", assicurati che "Max Pages to Process" sia impostato a 0 (elabora tutte le pagine)
2. Aumenta "Batch Size" (20-30) per migliorare le prestazioni
3. Il parametro "Process Every N Pages" può essere usato per elaborare solo alcune pagine (1 = tutte le pagine)

### Novità

- **Classificazione intelligente dei documenti**: Sistema avanzato per categorizzare automaticamente i documenti:
  1. **Rilevamento automatico categoria**: Classifica documenti in tecnici, legali, accademici, aziendali, strutturati o narrativi
  2. **Pipeline specializzate**: Applica strategie di elaborazione ottimizzate per ogni categoria di documento
  3. **Configurazione flessibile**: Parametri personalizzabili per ciascuna fase dell'elaborazione in base al tipo di documento
  4. **Interfaccia di classificazione**: Dashboard dedicata per testare e visualizzare i risultati della classificazione

- **Rilevamento tabella dei contenuti multilingua**: Supporto per identificazione e estrazione della tabella dei contenuti in tutte le lingue europee:
  1. **Riconoscimento automatico**: Identifica le tabelle dei contenuti basate su pattern specifici per lingua
  2. **Estrazione da outline PDF**: Supporto per estrarre indici direttamente dai bookmark nativi del PDF
  3. **Rilevamento struttura gerarchica**: Identifica i livelli di annidamento nelle tabelle dei contenuti
- **Ricerca avanzata con sommari**: Tre nuove strategie per migliorare la precisione della ricerca:
  1. **Embedding duali (testo + sommario)**: Indicizzazione vettoriale che combina il testo completo con il suo sommario estrattivo
  2. **Filtri di rilevanza basati su sommari**: Boosting dei risultati di ricerca quando il sommario è particolarmente rilevante
  3. **Template di risposta arricchiti**: Generazione di risposte che incorpora i sommari per un migliore contesto
- **Generazione automatica di sommari per i chunk**: Ogni chunk include un sommario estrattivo che ne sintetizza il contenuto principale
- **Sincronizzazione OCR-Spelling**: Sincronizzazione automatica tra lingua OCR e correzione ortografica
- **Dizionari standard italiani**: Supporto completo per la correzione ortografica in italiano
- **Elaborazione PDF multipagina**: Miglioramenti nell'estrazione di tutte le pagine dei documenti PDF
- **Sezione OCR e Correzione Ortografica**: Interfaccia dedicata con opzioni avanzate
- **Rilevamento automatico lingua OCR**: Ottimizza l'estrazione di testo dalle immagini
- **Miglioramenti al chunking semantico**: Algoritmi avanzati per documenti OCR
- **Visualizzazione analisi chunk**: Metriche dettagliate sulla qualità della segmentazione

## Contribuire

I contributi sono benvenuti! Per favore consulta le nostre [Linee guida per i contributi](https://github.com/lollonet/raggiro/wiki/Contributing) per dettagli su come inviare pull request, segnalare problemi e suggerire miglioramenti.

## Licenza

Questo progetto è rilasciato sotto la licenza MIT - vedi il file LICENSE per i dettagli.

## Riconoscimenti

- Costruito con ispirazione dalle moderne architetture RAG e pipeline di elaborazione documenti
- Utilizza molte eccellenti librerie open-source per il parsing dei documenti e l'elaborazione del testo