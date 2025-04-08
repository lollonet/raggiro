# Configurazione

Raggiro utilizza file di configurazione TOML per la personalizzazione. Puoi creare un file di configurazione in `~/.raggiro/config.toml` o specificare un percorso personalizzato con `--config`.

## Struttura della configurazione

Raggiro segue una chiara separazione delle preoccupazioni nella sua configurazione:

1. **File di configurazione TOML** (`config/config.toml`): Contiene tutte le impostazioni LLM, tra cui:
   - URL del server Ollama
   - Nomi dei modelli per diversi componenti (riscrittura, generazione)
   - Impostazioni specifiche del provider
   - Parametri di prestazione (temperatura, max_tokens)
   
2. **File di prompt di test** (`test_prompts/*.yaml`): Contengono solo impostazioni specifiche del test:
   - Prompt di test per diversi documenti
   - Criteri di valutazione e asserzioni
   - Configurazione della strategia di chunking

Questa separazione consente di mantenere una configurazione LLM centralizzata pur avendo prompt di test specifici per documento.

## Impostazioni OCR

Raggiro supporta l'estrazione OCR avanzata con diverse impostazioni di configurazione:

```toml
[extraction]
ocr_enabled = true
ocr_language = "eng+ita+spa+fra+deu"  # Supporto multilingua
ocr_dpi = 300  # DPI per la conversione dell'immagine (più alto = migliore qualità ma più memoria)
ocr_max_image_size = 4000  # Dimensione massima (larghezza o altezza) in pixel per l'elaborazione OCR
ocr_batch_size = 10  # Elabora questo numero di pagine in ogni batch per evitare problemi di memoria
```

Queste impostazioni consentono di ottimizzare la qualità del riconoscimento OCR bilanciando consumo di memoria e prestazioni:

- **ocr_language**: Formato concatenato con `+` per supportare più lingue
- **ocr_dpi**: Risoluzione (punti per pollice) per la conversione dell'immagine
- **ocr_max_image_size**: Limita le dimensioni dell'immagine per evitare problemi di memoria
- **ocr_batch_size**: Suddivide documenti grandi in batch più piccoli

## Correzione ortografica

Raggiro include funzionalità di correzione ortografica automatica, particolarmente utile per migliorare la qualità del testo estratto tramite OCR:

```toml
[spelling]
enabled = true
language = "auto"  # Può essere 'auto', 'en', 'it', 'es', 'fr', 'de', ecc.
backend = "symspellpy"  # Può essere 'symspellpy', 'textblob', o 'wordfreq'
max_edit_distance = 2
always_correct = false  # Se true, applica la correzione a tutti i documenti, non solo OCR
```

La correzione ortografica supporta diverse opzioni:

- **language**: Impostato su "auto" rileva automaticamente la lingua del documento, oppure può essere specificata manualmente
- **backend**: Algoritmo di correzione ortografica da utilizzare (symspellpy è il più veloce ed efficace)
- **max_edit_distance**: Controllo della "distanza di edit" massima per le correzioni (2 è un buon compromesso tra precisione e recall)
- **always_correct**: Normalmente la correzione è applicata solo ai documenti OCR, attivando questa opzione si applica a tutti i documenti

## Configurazione del supporto multilingua

Raggiro supporta l'elaborazione di documenti in più lingue:

```toml
# Impostazioni di segmentazione
[segmentation]
use_spacy = true
spacy_model = "xx_sent_ud_sm"  # Modello multilingue per supporto a più lingue
# Modelli linguistici alternativi per lingue specifiche:
# - "it_core_news_sm" per italiano
# - "en_core_web_sm" per inglese
# - "fr_core_news_sm" per francese
# - "de_core_news_sm" per tedesco
# - "es_core_news_sm" per spagnolo
# - "pt_core_news_sm" per portoghese
# - "nl_core_news_sm" per olandese

# Impostazioni per la rilevazione della tabella dei contenuti
[segmentation.toc_detection]
enabled = true
min_entries = 3  # Numero minimo di voci richieste per considerare valida una TOC
max_entries = 150  # Numero massimo di voci da estrarre
aggressiveness = 2  # Livello di aggressività nella ricerca (1=conservativo, 2=normale, 3=aggressivo)
```

Per documenti multilingua, è consigliabile:
1. Installare il modello spaCy multilingua: `uv run python -m spacy download xx_sent_ud_sm`
2. Aggiornare la configurazione per utilizzare questo modello
3. Assicurarsi che le lingue appropriate siano installate per Tesseract OCR

Per informazioni più dettagliate sull'uso di spaCy e sui relativi comandi, vedere la [documentazione spaCy](spacy.md).

## Rilevamento della tabella dei contenuti

Raggiro include funzionalità avanzate per rilevare e estrarre la tabella dei contenuti (indice) dai documenti:

```toml
# Impostazioni per la rilevazione della tabella dei contenuti
[segmentation.toc_detection]
enabled = true
min_entries = 3  # Numero minimo di voci richieste per considerare valida una TOC
max_entries = 150  # Numero massimo di voci da estrarre
aggressiveness = 2  # Livello di aggressività nella ricerca (1=conservativo, 2=normale, 3=aggressivo)
```

Caratteristiche principali del rilevamento tabella dei contenuti:

- **Supporto multilingua**: Rileva automaticamente tabelle dei contenuti in tutte le lingue europee
- **Estrazione da PDF nativi**: Utilizza l'outline/bookmarks dei PDF quando disponibili
- **Rilevamento pattern**: Identifica le tabelle dei contenuti anche quando non sono esplicitamente marcate
- **Analisi strutturale**: Estrae i livelli gerarchici e la struttura della tabella dei contenuti
- **Supporto numeri di pagina**: Mappa correttamente le voci dell'indice alle pagine corrispondenti

## Esempio di configurazione completa

```toml
# Impostazioni di elaborazione
[processing]
dry_run = false
recursive = true

# Impostazioni di estrazione
[extraction]
ocr_enabled = true
ocr_language = "eng+ita+spa+fra+deu"  # Supporto multilingua
ocr_dpi = 300
ocr_max_image_size = 4000
ocr_batch_size = 10

# Impostazioni di segmentazione
[segmentation]
use_spacy = true
spacy_model = "xx_sent_ud_sm"
semantic_chunking = true
chunking_strategy = "hybrid"

# Rilevamento tabella dei contenuti
[segmentation.toc_detection]
enabled = true
min_entries = 3
max_entries = 150
aggressiveness = 2

# Impostazioni LLM condivise
[llm]
provider = "ollama"  # "ollama", "llamacpp", "openai", "replicate"
ollama_base_url = "http://ollama:11434"  # URL API Ollama
ollama_timeout = 30  # Timeout in secondi

# Impostazioni di riscrittura delle query
[rewriting]
enabled = true
llm_type = ${llm.provider}  # Eredita dalla sezione llm
temperature = 0.1
max_tokens = 200
ollama_model = "llama3"  # Nome del modello per Ollama
```

Per dettagli completi su tutte le opzioni di configurazione, consulta il file di esempio `config/config.toml`.