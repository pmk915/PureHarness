# MiniHarness

MiniHarness is a small, transparent, benchmark-driven CLI agent harness for
studying and running reliable long-horizon, tool-using agents. It is an
educational and experimental systems project focused on execution, durable
conversation state, model-facing context, observable traces, replayable run
evidence, and external task verification.

The coding agent is the main workload used to exercise the runtime. It is not
the runtime kernel itself.

MiniHarness is not a LangChain replacement, SaaS backend, multi-agent
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
                   RunRecord   ExecutionBackend
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

MiniHarness requires Python 3.11 or newer.

```bash
git clone https://github.com/pmk915/miniharness.git
cd miniharness
python -m venv .venv
.venv/bin/python -m pip install -e '.[cli]'
export DEEPSEEK_API_KEY='your-key'
.venv/bin/miniharness --help
```

The interactive CLI uses DeepSeek by default. It loads an uncommitted `.env`
file as well as the process environment. Start it in the project you want the
agent to inspect and edit:

```bash
cd /path/to/workspace
/path/to/miniharness/.venv/bin/miniharness
```

One interactive process keeps one Session across multiple `Agent.run()` calls.
Use `/help`, `/status`, or `/exit`; Ctrl+D exits cleanly. To retain one
RunRecord per turn, pass `--record-dir PATH` before entering the session.

## Commands

Run a single task and optionally save its structured evidence:

```bash
miniharness run "fix the failing test" --workspace . --record run.json
miniharness inspect run.json
```

Run a deliberately small real-model benchmark experiment:

```bash
miniharness benchmark \
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
- **Policy is not isolation.** ToolPolicy decides whether a capability may run;
  an ExecutionBackend decides where command execution occurs.
- **Session is not Run.** A Session spans conversation turns, while every
  `Agent.run()` produces its own RunRecord.

## Testing

```bash
.venv/bin/python -m pip install -e . pytest
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

MiniHarness v0.1 intentionally omits durable interactive resume, crash
recovery, a full-screen TUI, web services, multi-agent orchestration, and a
plugin framework.
