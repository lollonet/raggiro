# Valutazione Sistema di Storage Vettoriale per Raggiro

## Introduzione

Questo documento valuta le opzioni per lo storage degli embedding vettoriali generati dal sistema Raggiro. Un database vettoriale efficiente è fondamentale per supportare operazioni di ricerca semantica, retrieval e RAG (Retrieval-Augmented Generation).

## pgvector: Soluzione Raccomandata

**[pgvector](https://github.com/pgvector/pgvector)** viene identificato come la soluzione preferita per l'implementazione attuale di Raggiro per i seguenti motivi:

### Vantaggi principali

1. **Integrazione in PostgreSQL**:
   - Estensione nativa per PostgreSQL, sfruttando un DBMS maturo ed enterprise-ready
   - Possibilità di conservare metadati, testo e vettori in un unico sistema di database

2. **Indici efficienti**:
   - Supporto per indici IVFFlat (Inverted File con Flat Compression)
   - Supporto per indici HNSW (Hierarchical Navigable Small World)
   - Buone prestazioni con milioni di vettori

3. **SQL e transazionalità**:
   - Utilizzo di SQL standard per query e manipolazione dati
   - Supporto completo ACID per le transazioni
   - Gestione ottimale di operazioni concorrenti

4. **Maturità e supporto**:
   - Progetto attivamente mantenuto
   - Ampia adozione nella community
   - Documentazione completa e casi d'uso ben documentati

5. **Scalabilità**:
   - Supporto per clustering e replica standard di PostgreSQL
   - Possibilità di crescere da deployment semplici a distribuzioni enterprise

6. **Operazioni vettoriali avanzate**:
   - Supporto per più metriche di distanza (L2, inner product, cosine)
   - API semplice per operazioni di ricerca KNN

7. **Costo e licenza**:
   - Open source con licenza compatibile
   - Nessun costo di licenza aggiuntivo oltre a PostgreSQL

## Altre opzioni considerate

### Faiss (Facebook AI Similarity Search)
- **Pro**: Alte prestazioni, supporto per GPU, molte opzioni di indici
- **Contro**: Solo in-memory, persistenza limitata, no transazionalità

### Qdrant
- **Pro**: Progettato specificamente per vettori, payload filterable, alta efficienza
- **Contro**: Deployment e manutenzione separata, integrazione aggiuntiva

### Chroma
- **Pro**: Facile da usare, deployment semplice, integrazione con LangChain
- **Contro**: Meno maturo, limitazioni con dataset molto grandi

### Weaviate
- **Pro**: Modello graph-like, ricchezza funzionale, filtri avanzati
- **Contro**: Complessità di configurazione, overhead di sistema

### Milvus
- **Pro**: Alta scalabilità, ottimizzazione cloud-native
- **Contro**: Complessità operativa, overkill per deployment più piccoli

## Piano di implementazione con pgvector

### Fase 1: Setup iniziale
1. Aggiungere dipendenze PostgreSQL e pgvector al progetto
2. Definire schema del database per documenti, chunk e metadati
3. Implementare funzioni helper per connessione e operazioni comuni

### Fase 2: Integrazione con Raggiro
1. Costruire adapter tra pipeline RAG e pgvector
2. Implementare funzioni di:
   - Inserimento embedding con metadati associati
   - Ricerca semantica con filtri sui metadati
   - Gestione delle collezioni/indici

### Fase 3: Ottimizzazione
1. Configurare indici appropriati in base al volume di dati
2. Implementare strategie di caching se necessario
3. Ottimizzare query per casi d'uso specifici

### Considerazioni per la configurazione
- Utilizzare indici HNSW per dataset di grandi dimensioni
- Configurare dimensione appropriata dei vettori (384 per all-MiniLM-L6-v2)
- Ottimizzare parametri PostgreSQL per carichi di lavoro vettoriali:
  - `maintenance_work_mem`: incrementare per costruzione indici
  - `shared_buffers`: ottimizzare per caching
  - `effective_cache_size`: adattare alla RAM disponibile

## Esempio di implementazione

```python
# Schema pgvector (pseudocodice SQL)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    source TEXT,
    language TEXT,
    category TEXT,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    text TEXT,
    embedding VECTOR(384),
    metadata JSONB,
    chunk_index INTEGER
);

CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

```python
# Esempio di classe adapter
class PgVectorStorage:
    def __init__(self, connection_string):
        self.conn = psycopg.connect(connection_string)
    
    def add_document(self, title, source, language, category):
        # Implementazione inserimento documento
        pass
        
    def add_chunks(self, document_id, chunks, embeddings):
        # Implementazione inserimento chunks con embeddings
        pass
        
    def semantic_search(self, query_vector, filter_criteria=None, limit=5):
        # Implementazione ricerca semantica con filtri opzionali
        pass
```

## Conclusioni

pgvector rappresenta la soluzione ottimale per lo storage vettoriale in Raggiro, combinando robustezza, prestazioni e semplicità di integrazione. La sua natura transazionale e il supporto SQL offrono vantaggi significativi rispetto ad alternative più specializzate ma meno integrate.

Si raccomanda di implementare pgvector come prima opzione, mantenendo comunque un design sufficientemente astratto da permettere in futuro la sostituzione con alternative qualora emergessero requisiti specifici non soddisfatti.

---

*Documento creato il: 8 Aprile 2025*  
*Da rivedere: Quando il volume di dati supera il milione di documenti o se emergono requisiti particolari di latenza*