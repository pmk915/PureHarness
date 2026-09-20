# PureHarness

PureHarness is a small, transparent, benchmark-driven CLI agent harness for
studying and running reliable long-horizon, tool-using agents. It is an
educational and experimental systems project focused on execution, durable
conversation state, model-facing context, observable traces, replayable run
evidence, and external task verification.

The coding agent is the main workload used to exercise the runtime. It is not
the runtime kernel itself.

PureHarness is not a LangChain replacement, SaaS backend, multi-agent
platform, production security boundary, or full coding-agent product. The
project deliberately favors explicit Python components over a broad framework.

## Architecture

```text
                    CLI
                     |
                     v
                Agent Runtime
           /          |           \
       Context      Session       Tools
          |            |            |
      TaskState    Events/Trace   ToolPolicy
                       |            |
             Durable Session   ApprovalHandler
                 + RunRecord       |
                            ExecutionBackend
                                /       \
                               Local     Docker

             External Benchmark + Verifier
                          |
                      Experiment
```

The runtime kernel depends on narrow model, context, tool, policy, and
execution interfaces. The CLI and benchmark layers compose those interfaces;
the kernel does not depend on either layer. See
[the architecture document](docs/architecture.md) for the implemented
boundaries.

## Quickstart

PureHarness requires Python 3.11 or newer.

```bash
git clone https://github.com/pmk915/pureharness.git
cd pureharness
python -m venv .venv
.venv/bin/python -m pip install -e '.[cli]'
export DEEPSEEK_API_KEY='your-key'
.venv/bin/pureharness --help
```

The interactive CLI uses DeepSeek by default. It loads an uncommitted `.env`
file as well as the process environment. Start it in the project you want the
agent to inspect and edit:

```bash
cd /path/to/workspace
/path/to/pureharness/.venv/bin/pureharness
```

One interactive conversation keeps one Session across multiple `Agent.run()`
calls and process restarts. Use `/help`, `/status`, or `/exit`; Ctrl+D exits
cleanly. Sessions are stored under `~/.pureharness/sessions` by default. Set
`PUREHARNESS_HOME` to isolate or relocate that state. To export an additional
RunRecord per turn, pass `--record-dir PATH` before entering the session.

## Commands

Run a single task and optionally save its structured evidence:

```bash
pureharness run "fix the failing test" --workspace . --record run.json
pureharness inspect run.json
```

Discover and resume durable interactive sessions:

```bash
pureharness sessions
pureharness resume <session-id>
```

Resume loads prior conversation state passively. It does not call the model or
re-execute historical tools; the next ordinary user message starts a new Run.
`/status`, `/help`, and `/exit` remain available without an API key.

Use the versioned local machine interfaces when output will be consumed by a
program:

```bash
pureharness run "fix the failing test" --output jsonl
pureharness inspect run.json --json
pureharness sessions --json
```

JSONL run mode writes only one JSON event per stdout line. Configuration and
startup errors go to stderr and the process exit code remains authoritative.
Interactive mode stays human-facing.

When an injected policy returns `REQUIRE_APPROVAL`, interactive mode displays a
redacted argument preview and asks for a one-time decision:

```text
Approval required
Tool: run_command
Arguments:
  argv: ["make", "clean"]
Approve this action? [y/N]: y
```

Only `y` or `yes` (case-insensitive) approves. Empty, invalid, EOF, or Ctrl+C
input denies the action. One-shot mode has no interactive approval handler and
therefore fails closed for approval-gated actions.

Run a deliberately small real-model benchmark experiment:

```bash
pureharness benchmark \
  --task simple_fix \
  --config raw_baseline \
  --repetitions 1 \
  --output benchmarks/results/first-run.jsonl
```

The benchmark command makes nondeterministic, potentially paid API calls. Its
ordinary unit tests use scripted models and never require an API key. Read the
[benchmark guide](benchmarks/README.md) and
[real-model experiment protocol](benchmarks/REAL_MODEL_EXPERIMENT.md) before
running the full matrix.

## Core ideas

- **History is not model context.** Session keeps the raw trajectory; context
  projection, TaskState, compaction, and token-budget selection are derived
  model-facing views.
- **Completed is not task success.** An Agent can finish normally while an
  external trusted benchmark verifier still rejects its work.
- **Replay is not re-execution.** Observational replay reads RunRecord evidence
  without calling a model, tool, policy, or execution backend again.
- **Resume is not replay.** Resume restores durable logical conversation state
  and waits for a new turn; it never repeats historical model or tool work.
- **Policy is not approval or isolation.** ToolPolicy classifies an action;
  ApprovalHandler obtains a host decision when required; ExecutionBackend
  decides where an approved action runs. Approval does not make code safe.
- **Session is not Run.** A Session spans conversation turns, while every
  `Agent.run()` produces its own RunRecord.
- **Interrupted is not failed.** Ctrl+C finalizes the current RunRecord as
  `interrupted`, rolls Session back to its last durable logical boundary, and
  returns control to the prompt. An uncertain in-flight tool is never resumed
  or automatically retried.
- **Live events are not persisted evidence.** JSONL exposes execution while it
  happens; RunRecord is finalized evidence; replay is a read-only ordered view
  derived from that evidence.

## Testing

```bash
.venv/bin/python -m pip install -e '.[cli,dev]'
.venv/bin/python -m pytest
```

The configured suite is offline and deterministic. Docker-specific tests skip
when Docker is unavailable.

## Security and limitations

The default local execution backend launches host subprocesses and is not a
sandbox. The Docker backend adds useful isolation controls but is not a
hardened hostile multi-tenant boundary. Do not expose untrusted workspaces or
secrets on the assumption that ToolPolicy alone provides containment. See the
[security model](docs/security.md) for the exact boundary and current
limitations.

PureHarness intentionally omits exact call-stack continuation, automatic retry
of interrupted tools, concurrent writers for one session, workspace
relocation, session branching, a full-screen TUI, web services, multi-agent
orchestration, and a plugin framework.

The machine interface is a local schema-version-1 JSON/JSONL surface, not an
OpenTelemetry exporter, remote logging service, RPC protocol, or interactive
machine-control API. See the [observability guide](docs/observability.md).
