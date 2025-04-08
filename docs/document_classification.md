# Classificazione dei Documenti

## Introduzione

La classificazione dei documenti è una fase preliminare fondamentale nel sistema Raggiro, che consente di identificare automaticamente il tipo e il formato di un documento prima della sua elaborazione. Questo approccio permette di adattare dinamicamente le pipeline di elaborazione in base alle caratteristiche specifiche di ciascun documento, migliorando l'efficacia e l'efficienza dell'intero processo.

## Vantaggi della Classificazione

1. **Pipeline specializzate**: Ogni tipo di documento può essere elaborato con una pipeline ottimizzata per le sue caratteristiche specifiche.
2. **Estrazione mirata dei dati**: Diversi documenti strutturati (fatture, contratti, moduli) richiedono approcci diversi per estrarre informazioni rilevanti.
3. **Selezione intelligente dei parametri**: La classificazione permette di selezionare automaticamente i migliori parametri per segmentazione, pulizia e altre fasi di elaborazione.
4. **Rilevamento del multilingua**: Identificazione automatica della lingua del documento per adattare algoritmi di NLP.
5. **Ottimizzazione delle risorse**: Allocazione efficiente delle risorse computazionali in base alla complessità del documento.

## Architettura del Sistema di Classificazione

### Flusso di Elaborazione

1. **Pre-classificazione**: Analisi iniziale basata su metadati (estensione file, MIME type)
2. **Classificazione approfondita**: Analisi del contenuto per determinare tipo e struttura del documento
3. **Selezione della pipeline**: Identificazione della pipeline di elaborazione ottimale
4. **Elaborazione specializzata**: Applicazione di algoritmi specifici per il tipo di documento
5. **Post-elaborazione**: Raffinamento dei risultati in base alla categoria del documento

### Categorie di Documenti

Il sistema classifica i documenti in diverse categorie principali:

| Categoria | Descrizione | Esempi | Pipeline specializzata |
|-----------|-------------|--------|------------------------|
| **Tecnico** | Documentazione tecnica, manuali | Manuali di prodotto, specifiche tecniche | Estrazione di dettagli tecnici, diagrammi, formule |
| **Legale** | Documenti legali, normativi | Contratti, leggi, regolamenti | Identificazione clausole, riferimenti normativi |
| **Accademico** | Pubblicazioni scientifiche | Paper scientifici, tesi | Estrazione di citazioni, equazioni, riferimenti bibliografici |
| **Aziendale** | Documenti aziendali | Report finanziari, presentazioni | Estrazione di dati finanziari, KPI, grafici |
| **Strutturato** | Documenti con formato rigido | Moduli, fatture, CV | Estrazione di campi specifici e tabelle |
| **Narrativo** | Testo narrativo, prosa | Articoli, racconti, libri | Analisi semantica, estrazione entità |

### Caratteristiche Analizzate

- **Strutturali**: Layout, presenza di tabelle/grafici, densità del testo
- **Linguistiche**: Lingua, registro, terminologia specialistica
- **Formali**: Font, stili, formattazione, presenza di elementi grafici
- **Contenutistiche**: Parole chiave, entità menzionate, argomenti trattati
- **Metadata**: Informazioni su autore, data, proprietà del documento

## Implementazione Tecnica

### Classe DocumentClassifier

La classe `DocumentClassifier` è il componente centrale del sistema di classificazione:

```python
class DocumentClassifier:
    def __init__(self, config=None):
        # Inizializzazione con configurazione personalizzabile
        self.config = config or {}
        self.model = self._load_model()

    def classify(self, document):
        # Analisi e classificazione del documento
        # Restituisce categoria e confidenza

    def _extract_features(self, document):
        # Estrazione di caratteristiche rilevanti dal documento
        
    def _load_model(self):
        # Caricamento del modello di classificazione
```

### Integrazione con la Pipeline Esistente

Il classificatore è integrato all'inizio della pipeline di elaborazione:

```python
class DocumentProcessor:
    def __init__(self, config=None):
        # Componenti esistenti
        # ...
        
        # Nuovo componente di classificazione
        self.classifier = DocumentClassifier(config)
        
        # Pipeline specializzate per tipo di documento
        self.pipelines = {
            "technical": TechnicalDocumentPipeline(config),
            "legal": LegalDocumentPipeline(config),
            # ...
        }
    
    def process_file(self, file_path, output_dir):
        # Classificazione del documento
        document_type = self.classifier.classify(file_path)
        
        # Selezione della pipeline appropriata
        pipeline = self.pipelines.get(document_type, self.default_pipeline)
        
        # Elaborazione con pipeline specializzata
        return pipeline.process(file_path, output_dir)
```

## Pipeline Specializzate

Per ogni categoria di documento, è stata sviluppata una pipeline specializzata che ottimizza l'elaborazione:

### Esempio: Pipeline per Documenti Tecnici

```python
class TechnicalDocumentPipeline:
    def __init__(self, config):
        # Configurazione specifica per documenti tecnici
        self.extractor = TechnicalExtractor(config)
        self.segmenter = TechnicalSegmenter(config)
        # ...
    
    def process(self, file_path, output_dir):
        # Elaborazione ottimizzata per documenti tecnici
```

## Modelli di Classificazione

### Approcci Implementati

1. **Basato su regole**: Per classificazione rapida basata su caratteristiche evidenti
2. **TF-IDF + SVM**: Per classificazione basata sul contenuto testuale
3. **Embedding + Classifier**: Per comprensione semantica più profonda del documento
4. **Ensemble**: Combinazione di più classificatori per maggiore accuratezza

### Addestramento e Miglioramento

Il sistema supporta:

- Addestramento iniziale su dataset etichettati
- Apprendimento continuo con feedback dell'utente
- Fine-tuning su domini specifici

## Configurazione

L'intero sistema di classificazione è configurabile tramite il file `config.toml`:

```toml
[classifier]
enabled = true
model_type = "ensemble"  # "rules", "tfidf_svm", "embedding", "ensemble"
confidence_threshold = 0.75
model_path = "models/document_classifier"

[classifier.rules]
use_file_metadata = true
use_content_features = true

[pipeline]
use_specialized_pipelines = true
default_pipeline = "general"
```

## Guida all'Uso

### Classificazione Esplicita

```python
processor = DocumentProcessor()
classification = processor.classifier.classify("path/to/document.pdf")
print(f"Documento classificato come: {classification['category']}")
print(f"Confidenza: {classification['confidence']}")
```

### Elaborazione Completa

```python
result = processor.process_file("path/to/document.pdf", "output/")
print(f"Documento elaborato con pipeline: {result['pipeline_used']}")
```

## Conclusioni

L'implementazione del sistema di classificazione dei documenti rappresenta un significativo miglioramento dell'architettura Raggiro, consentendo un'elaborazione più intelligente e adattiva. Questo approccio permette di ottimizzare ogni fase del processo in base alle caratteristiche specifiche del documento, migliorando la qualità complessiva dell'elaborazione e dei risultati di RAG.

Grazie alla classificazione preliminare, il sistema è ora in grado di riconoscere automaticamente il tipo di documento e selezionare la pipeline di elaborazione più appropriata, offrendo risultati più accurati e pertinenti per ciascuna tipologia documentale.