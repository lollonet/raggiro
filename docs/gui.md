# Interfacce GUI

Le interfacce GUI di Raggiro forniscono un'elaborazione interattiva dei documenti senza dover utilizzare parametri da riga di comando.

## Interfaccia Streamlit (basata sul web)

L'interfaccia Streamlit offre un ambiente completo accessibile tramite browser web:

```bash
# Avvia l'interfaccia Streamlit
raggiro gui
# oppure
streamlit run $(which raggiro)
# o direttamente dallo script
python scripts/gui/launch_gui.py
# o utilizzando lo script bash
./scripts/gui/run_streamlit.sh
```

L'interfaccia Streamlit include quattro schede principali:

### 1. Elaborazione documenti
- Carica e processa documenti tramite browser
- Opzioni configurabili per l'elaborazione (OCR, formato, ecc.)
- Visualizzazione dei risultati dell'elaborazione
- Indicatori di progresso in tempo reale
- Gestione degli errori con messaggi dettagliati

### 2. Test RAG
- Esegui test su documenti elaborati
- Prompt predefiniti o personalizzati
- Selezione dinamica del modello Ollama
- Monitoraggio dell'esecuzione del test in tempo reale
- Configurazione semplificata tramite interfaccia grafica

### 3. Visualizzazione risultati
- Analizza i risultati dei test con visualizzazioni
- Confronta le prestazioni di diverse strategie di chunking
- Visualizza metriche e statistiche dettagliate
- Esplora la cronologia dei test precedenti

### 4. Configurazione
- Modifica le impostazioni di configurazione per l'intera pipeline
- Configurazione visuale delle impostazioni LLM
- Validazione in tempo reale dei parametri
- Gestione dei percorsi e delle opzioni di sistema

### Caratteristiche principali
- **Integrazione con Ollama**: Recupera dinamicamente i modelli disponibili dal server Ollama
- **Gestione documenti**: Interfaccia drag-and-drop per il caricamento dei documenti
- **Visualizzazione avanzata**: Grafici interattivi per le metriche di prestazione
- **Feedback immediato**: Messaggi di stato e notifiche in tempo reale

## Interfaccia Textual (basata sul terminale)

L'interfaccia Textual fornisce un'esperienza basata sul terminale ideale per ambienti senza browser:

```bash
# Avvia l'interfaccia Textual
raggiro gui --tui
```

### Caratteristiche dell'interfaccia Textual
- **Interfaccia testuale**: UI completa all'interno del terminale
- **Navigazione da tastiera**: Interfaccia guidata da tastiera per un uso efficiente
- **Consumo ridotto di risorse**: Ideale per server o ambienti con risorse limitate
- **Accessibilità remota**: Perfetta per l'uso tramite SSH o connessioni remote

### Funzionalità supportate
- Elaborazione documenti
- Configurazione di base
- Visualizzazione dei risultati in formato testo
- Gestione dei log e notifiche di stato

L'interfaccia Textual è particolarmente utile per:
- Ambienti server senza GUI
- Sistemi con risorse limitate
- Utenti che preferiscono interfacce terminale
- Accesso remoto tramite SSH