# Utilizzo di spaCy in Raggiro

Raggiro utilizza [spaCy](https://spacy.io/) come motore linguistico principale per diverse funzionalità critiche, in particolare per la segmentazione semantica e l'analisi avanzata del testo. Questa documentazione copre l'installazione, la configurazione e l'utilizzo di spaCy all'interno del sistema Raggiro.

## Installazione dei modelli linguistici

> **NOTA IMPORTANTE**: I modelli linguistici di spaCy non possono essere installati direttamente come dipendenze in `pyproject.toml` quando si usa `uv`. È necessario installarli manualmente come mostrato di seguito.

### Modello multilingue (raccomandato)

Il modello multilingue `xx_sent_ud_sm` è consigliato per la maggior parte degli utenti in quanto supporta contemporaneamente tutte le principali lingue europee:

```bash
uv run python -m spacy download xx_sent_ud_sm
```

### Modelli specifici per lingua

Per prestazioni ottimali con singole lingue, è possibile installare modelli linguistici specifici:

```bash
# Italiano (consigliato per documenti italiani)
uv run python -m spacy download it_core_news_sm

# Inglese
uv run python -m spacy download en_core_web_sm

# Francese
uv run python -m spacy download fr_core_news_sm

# Tedesco
uv run python -m spacy download de_core_news_sm

# Spagnolo
uv run python -m spacy download es_core_news_sm

# Portoghese
uv run python -m spacy download pt_core_news_sm

# Olandese
uv run python -m spacy download nl_core_news_sm
```

### Fallback automatico

Il sistema è progettato per funzionare anche se non tutti i modelli sono installati:

1. Tenterà di caricare il modello specificato in config.toml
2. Se non disponibile, proverà a caricare modelli alternativi nell'ordine: `xx_sent_ud_sm`, `it_core_news_sm`, `en_core_web_sm`
3. Come ultima risorsa, utilizzerà un modello blank che offre funzionalità limitate

### Comandi utili per la gestione dei modelli

```bash
# Elencare tutti i modelli spaCy installati
uv run python -m spacy info --all

# Verificare se un modello è installato
uv run python -m spacy validate

# Verificare le informazioni su un modello specifico
uv run python -m spacy info it_core_news_sm

# Aggiornare un modello esistente
uv run python -m spacy download it_core_news_sm --force
```

## Configurazione spaCy in Raggiro

La configurazione di spaCy viene gestita nella sezione `[segmentation]` del file `config.toml`:

```toml
[segmentation]
use_spacy = true
spacy_model = "xx_sent_ud_sm"  # Modello multilingue per supporto a più lingue
# Alternative per lingue specifiche:
# - "it_core_news_sm" per italiano
# - "en_core_web_sm" per inglese
# - "fr_core_news_sm" per francese
# - "de_core_news_sm" per tedesco
# - "es_core_news_sm" per spagnolo
# - "pt_core_news_sm" per portoghese
# - "nl_core_news_sm" per olandese
min_section_length = 100
max_chunk_size = 1500
chunk_overlap = 200
semantic_chunking = true  # Abilita il chunking basato su semantica
chunking_strategy = "hybrid"  # Opzioni: "size", "semantic", "hybrid"
semantic_similarity_threshold = 0.55
```

## Funzionalità spaCy in Raggiro

### 1. Segmentazione semantica

spaCy viene utilizzato per dividere i documenti in chunk semanticamente coerenti, il che è fondamentale per ottimizzare le prestazioni dei sistemi RAG. Il processo:

1. Analizza la struttura linguistica del testo (frasi, paragrafi)
2. Identifica i confini naturali di significato
3. Crea chunk che mantengono l'integrità semantica
4. Ottimizza i chunk per le query di recupero

### 2. Analisi linguistica

spaCy fornisce analisi linguistica approfondita:

- Tokenizzazione e lemmatizzazione
- Identificazione di parti del discorso (POS tagging)
- Riconoscimento di entità nominate (NER)
- Analisi delle dipendenze sintattiche

### 3. Ottimizzazione per RAG

L'uso di spaCy migliora significativamente la qualità dei risultati RAG:

- Chunk più coerenti portano a migliori risultati di ricerca
- L'analisi delle entità migliora la pertinenza delle risposte
- La comprensione della struttura linguistica aiuta a mantenere il contesto

## Esempi di utilizzo diretto (per sviluppatori)

Per gli sviluppatori che desiderano estendere Raggiro, ecco alcuni esempi di utilizzo diretto di spaCy:

```python
import spacy

# Carica il modello
nlp = spacy.load("it_core_news_sm")

# Analizza un testo
doc = nlp("Raggiro è un framework per l'elaborazione di documenti con supporto RAG.")

# Estrazione di entità
for ent in doc.ents:
    print(f"Entità: {ent.text}, Tipo: {ent.label_}")

# Analisi sintattica
for token in doc:
    print(f"{token.text}: POS={token.pos_}, DEP={token.dep_}, Lemma={token.lemma_}")

# Trovare confini di frase (utile per chunking)
for sent in doc.sents:
    print(f"Frase: {sent.text}")

# Calcolo di similarità semantica (utile per chunking semantico)
doc1 = nlp("Il sistema RAG migliora le risposte dell'AI.")
doc2 = nlp("Le risposte dell'intelligenza artificiale sono migliorate dal RAG.")
similarity = doc1.similarity(doc2)
print(f"Similarità: {similarity}")  # Valore tra 0 e 1
```

## Personalizzazione avanzata

Per progetti che richiedono personalizzazione avanzata, spaCy supporta:

1. **Pipeline personalizzate**: Aggiungere componenti custom alla pipeline NLP
2. **Addestramento di modelli**: Perfezionare i modelli esistenti con dati specifici del dominio
3. **Regole linguistiche personalizzate**: Definire pattern specifici per l'estrazione di entità

Per maggiori informazioni sulla personalizzazione avanzata, consultare la [documentazione ufficiale di spaCy](https://spacy.io/usage/training).

## Risoluzione dei problemi

### Problemi comuni

1. **Errore "Model not found" o "Can't find model"**:
   ```
   Soluzione: uv run python -m spacy download <nome_modello>
   ```
   
   Se il comando precedente fallisce con errori di installazione quando si usa `uv`:
   ```
   # Soluzione alternativa con script Python
   uv run python -c "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '<nome_modello>'])"
   ```

2. **Prestazioni lente con documenti lunghi**:
   ```
   Soluzione: Usa modelli più piccoli (con suffisso _sm) o regola max_chunk_size
   ```

3. **Problemi con lingue specifiche**:
   ```
   Soluzione: Assicurati di aver installato il modello specifico per la lingua
   ```

4. **Errori di memoria**:
   ```
   Soluzione: Ridurre batch_size o utilizzare modelli più piccoli
   ```
   
5. **Modelli non disponibili con uv**:
   ```
   # Se con uv i modelli non sono installabili, puoi passare a un ambiente virtuale standard
   python -m venv venv
   source venv/bin/activate  # o venv\Scripts\activate su Windows
   pip install spacy
   python -m spacy download xx_sent_ud_sm
   ```

### Quando contattare il supporto

Se riscontri problemi persistenti con spaCy in Raggiro, controlla:

1. Versione di spaCy installata (`uv pip show spacy`)
2. Modelli installati (`uv run python -m spacy info --all`)
3. Log dettagliati di errore
4. Dimensioni e tipo di documento che causa il problema

Fornisci queste informazioni quando apri una issue su GitHub per ricevere supporto più rapido ed efficace.