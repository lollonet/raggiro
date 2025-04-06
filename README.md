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
- **Chunking semantico**: Divisione intelligente del contenuto basata sul significato piuttosto che solo sulla dimensione
- **Estrazione metadata**: Titolo, autore, data, lingua, tipo di documento, rilevamento categoria
- **Output strutturato**: Formati Markdown e JSON con tutti i metadata
- **Architettura modulare**: Interfacce CLI, API Python e GUI (Streamlit/Textual)
- **Funzionamento completamente offline**: Funziona senza dipendenze API esterne
- **Pipeline RAG completa**: Indicizzazione vettoriale, recupero e generazione di risposte integrati
- **Utilità di testing**: Strumenti per il benchmarking e confronto tra strategie di chunking
- **Supporto multilingua**: Rilevamento automatico della lingua e mantenimento della coerenza linguistica

## Documentazione

Raggiro include una documentazione completa divisa nelle seguenti sezioni:

- [Installazione](docs/installation.md) - Guida all'installazione e requisiti di sistema
- [Configurazione](docs/configuration.md) - Configurazione TOML e opzioni personalizzabili
- [Riferimento CLI](docs/commands.md) - Comandi e opzioni della riga di comando
- [Interfacce GUI](docs/gui.md) - Utilizzo delle interfacce Streamlit e Textual
- [Pipeline RAG](docs/rag.md) - Componenti e utilizzo della pipeline RAG
- [Testing e valutazione](docs/testing.md) - Strumenti per testare e valutare il sistema RAG
- [Riferimento API](docs/api.md) - Documentazione dell'API Python per sviluppatori

## Avvio rapido

```bash
# Installazione
git clone https://github.com/lollonet/raggiro.git
cd raggiro
pip install -e .

# Elabora un documento
raggiro process document.pdf --output output_dir

# Avvia l'interfaccia GUI
raggiro gui
```

## Contribuire

I contributi sono benvenuti! Per favore consulta le nostre [Linee guida per i contributi](https://github.com/lollonet/raggiro/wiki/Contributing) per dettagli su come inviare pull request, segnalare problemi e suggerire miglioramenti.

## Licenza

Questo progetto è rilasciato sotto la licenza MIT - vedi il file LICENSE per i dettagli.

## Riconoscimenti

- Costruito con ispirazione dalle moderne architetture RAG e pipeline di elaborazione documenti
- Utilizza molte eccellenti librerie open-source per il parsing dei documenti e l'elaborazione del testo