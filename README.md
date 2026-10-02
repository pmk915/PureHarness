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
calls and process restarts. With the `cli` extra installed, real terminal output
uses compact Rich event presentation, a workspace/model/session header,
and a separated assistant response. Redirected output and injected output
callables use deterministic plain text. Choose presentation explicitly with:

```bash
pureharness                         # compact Rich interactive presentation
pureharness --verbose               # detailed human observability
pureharness --plain                 # deterministic plain presentation
pureharness --locale en
pureharness --locale zh-CN
```

Compact display emphasizes file actions, commands, new verification outcomes,
approvals, failures, and recovery. It changes presentation only: runtime events
and persisted evidence remain complete. `--verbose` restores detailed context,
model, policy, and evidence output; it is not debug logging and does not force
Rich on redirected/injected output. `--plain --verbose` is rejected.

The default locale is `en`; plain output remains English. The optional
prompt-toolkit input adapter provides the `You ›` prompt, in-process history
(Up/Down), Tab completion for slash commands, and basic terminal editing. Enter
submits; Alt+Enter (or Escape then Enter) inserts a newline. History is kept only
in memory, and approval responses are excluded from it. Without the optional
dependency or terminal input/output, the CLI falls back to ordinary input.
Custom `input_fn` / `output_fn` callables remain authoritative for embedding and
tests. Ctrl+C cancels current input or interrupts an active Run and returns to
the prompt; Ctrl+D exits cleanly.

| Command | Purpose |
| --- | --- |
| `/help` | Show interactive commands |
| `/status` | Show session, workspace, model, latest Run, and context usage |
| `/session` | Show durable identity, UTC timestamps, and history/Run counts |
| `/runs` | Show the latest 10 RunRecords in this session |
| `/eval` | Show execution metrics, diagnosis, and advisory recovery for the latest finalized Run |
| `/exit` | Save and leave the session |

Sessions are stored under `~/.pureharness/sessions` by default. Set
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
pureharness --continue
pureharness --workspace /path/to/workspace --continue
```

Resume loads prior conversation state passively. It does not call the model or
re-execute historical tools; the next ordinary user message starts a new Run.
`--continue` selects the most recently updated durable session whose persisted
workspace equals the resolved selected workspace. It reports an error when no
matching session exists, rather than starting a new one. Explicit resume uses
the saved workspace and model. Rich presentation flags can also follow resume:
`pureharness resume <session-id> --locale zh-CN --plain`.
`--verbose` works before or after `resume`, and with `--continue`.
All slash commands remain available without an API key, including `/eval` on
the latest persisted Run immediately after resume.

After each finalized Run, compact Rich displays a minimal M22 evaluation:

```text
Run
  Protocol completion: 1.000
  Diagnosis: none
```

Verbose and plain modes retain the existing automatic metrics block, including
step efficiency and tool reliability.
`/eval` adds Run ID, end reason, steps, tool counts, diagnosis confidence/reason,
and a RecoverySignal. Recovery actions are suggestions only; they never retry a
tool or alter a prompt. Completion means protocol execution completed, not
verified task correctness. The deterministic M22 metrics observe RunRecord;
trusted benchmark verification remains a separate external oracle. A completed
Run may still contain tool errors or fail the benchmark verifier.

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

For an interactive demo, start in a disposable workspace, submit a small task,
and inspect the streamed context/model/tool events, final response, and compact
evaluation. Then try `/status`, `/session`, `/runs`, `/eval`, `/help`, and `/exit`.
Run `pureharness sessions`, followed by `pureharness --continue` from that same
workspace; it should restore the session and wait for input. Repeat with
`--plain` to check the fallback. Ordinary task turns use the configured provider
and may incur API costs; the M23 tests cover these flows with scripted models,
including actual prompt-toolkit input via pipes, without paid API calls.

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
