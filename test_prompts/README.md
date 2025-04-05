# Configurazioni di test per Raggiro

Questo directory contiene configurazioni di test per diversi documenti PDF che possono essere utilizzati con la funzionalità di test RAG di Raggiro.

## Utilizzo

Per eseguire i test, utilizzare i seguenti comandi:

```bash
# 1. Elabora il documento
python -m raggiro.examples.scripts.test_semantic_chunking --input /path/to/your/document.pdf --output test_output_document

# 2. Esegui i test promptfoo
python -m raggiro.testing.promptfoo_runner /home/ubuntu/raggiro/test_prompts/nome_configurazione.yaml test_output_document
```

## Configurazioni disponibili

| Documento | File di configurazione | Descrizione |
|-----------|------------------------|-------------|
| Scrum Guide | `scrum_guide.yaml` | Test sulla Guida Scrum 2020 in italiano |
| WEF Future of Jobs Report | `future_of_jobs.yaml` | Test sul report del World Economic Forum sul futuro del lavoro |
| Humanizar | `humanizar.yaml` | Test sul documento di umanizzazione dei servizi |
| PSN Allegato Tecnico | `psn_allegato.yaml` | Test sull'allegato tecnico del Piano Strategico Nazionale |
| Hornresp Manual | `hornresp_manual.yaml` | Test sul manuale del software Hornresp |
| Canción de peregrino | `cancion_peregrino.yaml` | Test sul testo poetico Canción de peregrino |
| Capitolato Tecnico | `capitolato_tecnico.yaml` | Test sul capitolato tecnico e suoi allegati |
| Effortless Mastery | `kenny_werner.yaml` | Test sul libro "Effortless Mastery" di Kenny Werner |

## Struttura delle configurazioni

Ogni configurazione YAML include:

- `prompts`: Lista di domande specifiche per il documento
- `tests`: Asserzioni per verificare la qualità delle risposte
- `outputs`: Configurazione dei file di output
- `variants`: Configurazione delle varianti di test (chunking semantico vs. dimensionale)

## Esempio di comando completo

```bash
# Processo completo per il libro di Kenny Werner
python -m raggiro.examples.scripts.test_semantic_chunking --input /home/ubuntu/raggiro/tmp/Kenny_Werner_Effortless_Mastery_Liberati.pdf --output test_output_werner
python -m raggiro.testing.promptfoo_runner /home/ubuntu/raggiro/test_prompts/kenny_werner.yaml test_output_werner
```