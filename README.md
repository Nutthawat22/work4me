# work4me

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

- REPL: `python src/pipeline/runner.py`
- Handoff ingestion: `python src/pipeline/ingest.py --handoff <dir>`

Pipeline success means deterministic static validation passed; generated-project
runtime tests/builds are not executed. Each run writes `testing-handoff.md` and
`testing-handoff.json` with an isolated external-testing procedure and result
contract.
