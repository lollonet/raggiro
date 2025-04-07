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
- **Estrazione metadata**: Titolo, autore, data, lingua, tipo di documento, rilevamento categoria
- **Output strutturato**: Formati Markdown e JSON con tutti i metadata
- **Interfaccia GUI dedicata**: Sezioni specializzate per OCR e correzione ortografica/semantica
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
   - `python -m spacy download en_core_web_sm`
   - `python -m spacy download it_core_news_sm` (per supporto italiano)
   - `pip install pyspellchecker` (per dizionari di correzione ortografica standard)

## Documentazione

Raggiro include una documentazione completa divisa nelle seguenti sezioni:

- [Installazione](docs/installation.md) - Guida all'installazione e requisiti di sistema
- [Configurazione](docs/configuration.md) - Configurazione TOML e opzioni personalizzabili
- [Riferimento CLI](docs/commands.md) - Comandi e opzioni della riga di comando
- [Interfacce GUI](docs/gui.md) - Utilizzo delle interfacce Streamlit e Textual
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

# Installare dipendenze Python
pip install -e .
# Oppure specificamente con tutte le dipendenze
pip install -r requirements.txt

# Installare i modelli linguistici richiesti
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm  # Opzionale per supporto italiano
```

### 3. Verifica installazione Tesseract

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

# Usa l'interfaccia OCR specializzata
# Seleziona la scheda "OCR & Correction" nell'interfaccia GUI
```

### Nota importante per l'OCR di documenti multi-pagina

Per elaborare correttamente documenti PDF con molte pagine:
1. Nella scheda "OCR & Correction", assicurati che "Max Pages to Process" sia impostato a 0 (elabora tutte le pagine)
2. Aumenta "Batch Size" (20-30) per migliorare le prestazioni
3. Il parametro "Process Every N Pages" può essere usato per elaborare solo alcune pagine (1 = tutte le pagine)

### Novità

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