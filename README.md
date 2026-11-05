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

Run external checks by explicitly delegating the handoff to the Docker-isolated
agent. See [External Testing Agent](docs/external-testing-agent.md) for CLI/ACP
usage, supported runtimes, result artifacts, and isolation limits.

For `web-sdlc-proposal`, register the repository's [SKILL.md](SKILL.md) as an
external Code implementer with `bind: git` so ignored runtime files stay outside
its fingerprint. Refresh provider activation after updating the registered
source/skill files. The canonical ACP entry point is
`venv/bin/python src/pipeline/acp_server.py`. Configure `existing_project` with
the host's Code worktree. ACP status schema 2 reports `outcome: "passed"` for
implementation completion; `pipeline_outcome`, `validation_status`, and
`external_testing_status` describe verification separately. The finalized
`implementation_result_path` points to task and declared AC-coverage evidence
for the host's Code artifacts. See [ACP Integration](docs/acp-integration.md).
