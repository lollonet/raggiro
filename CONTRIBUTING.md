# Guida per Contributors

Grazie per il tuo interesse a contribuire a Raggiro! Questo documento fornisce le linee guida per partecipare allo sviluppo del progetto.

## Configurazione dell'ambiente di sviluppo

1. **Clona il repository e installa le dipendenze**

```bash
# Clona il repository
git clone https://github.com/lollonet/raggiro.git
cd raggiro

# Crea e attiva un ambiente virtuale (raccomandato)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
# venv\Scripts\activate  # Windows

# Installa uv (gestore pacchetti ultra-veloce)
pip install uv

# Installa il pacchetto in modalità sviluppo con le dipendenze dev
uv pip install -e ".[dev]"

# Installa anche i modelli spaCy necessari
python -m spacy download xx_sent_ud_sm
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

2. **Configura pre-commit hooks**

Utilizziamo pre-commit per garantire la qualità del codice:

```bash
pre-commit install
```

## Workflow di sviluppo

1. **Crea un branch per le tue modifiche**

```bash
git checkout -b feature/nome-della-feature  # per nuove funzionalità
# oppure
git checkout -b fix/nome-del-fix  # per bug fix
```

2. **Sviluppa la tua funzionalità o correzione**

- Segui le convenzioni di codice descritte in CLAUDE.md
- Scrivi codice PEP 8 compliant
- Includi docstring in formato Google style
- Aggiungi type hints alle funzioni e metodi

3. **Verifica che il tuo codice passi i controlli di qualità**

```bash
# Esegui manualmente i pre-commit hooks
pre-commit run --all-files

# Esegui i test
pytest
```

4. **Commit e push delle modifiche**

```bash
git add .
git commit -m "Descrizione significativa delle modifiche"
git push origin nome-del-tuo-branch
```

5. **Apri una Pull Request**

- Vai su GitHub e apri una pull request verso il branch `main`
- Descrivi in dettaglio le modifiche apportate
- Collega eventuali issue correlate

## Linee guida per il codice

- **Stile del codice**: Seguiamo PEP 8 e utilizziamo Black per la formattazione
- **Docstring**: Utilizziamo il formato Google-style per le docstring
- **Type hints**: Tutte le funzioni e metodi devono avere type hints
- **Test**: Ogni nuova funzionalità dovrebbe essere accompagnata da test
- **Lingue**: Commenti inline in italiano, docstring in inglese

## Processo di review

- Le pull request richiedono almeno una review approvata
- I commenti devono essere risolti prima del merge
- Tutti i controlli automatici devono passare
- Manteniamo una comunicazione rispettosa e costruttiva

## Risorse utili

- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

Grazie per contribuire a rendere Raggiro migliore!