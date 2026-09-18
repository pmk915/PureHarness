# MiniHarness architecture

This document separates the implementation that exists today from the intended
architecture. Sections marked **Current** describe repository behavior through
the M5B ToolPolicy and ToolExecutor milestone. Sections marked **Target** describe
direction, not implemented APIs.

## 1. Project positioning

MiniHarness is a small, inspectable agent runtime: it coordinates a model,
conversation state, context selection, tools, execution traces, and observable
events through an explicit loop. Its product values are transparency,
reliability, and measurability.

The coding agent is the primary workload and benchmark used to develop that
runtime. File reading, file writing, and command execution demonstrate what the
runtime can host, but these capabilities are not part of the kernel's identity.
Removing the coding tools should leave a coherent agent runtime.

MiniHarness favors a small kernel, replaceable integrations, and ordinary Python
over a broad framework or a coding-agent product.

## 2. Current architecture

**Current:** the repository is a compact Python package under
`src/miniharness`. The runtime flow is:

```text
user input
   |
   v
Agent -> Session.snapshot() -> ContextBuilder -> Model
  |                                           |
  |                 Message or ToolCall(s) <--+
  |                           |
  +-> ToolExecutor -> ToolRegistry lookup
  |              `-> ToolPolicy -> Tool callable
  |
  +-> Session + RunTrace + AgentEvent listeners

External lifecycle -> SessionStore <-> Session -> Agent
```

### Agent

`Agent` owns the synchronous execution loop. At the start of each run it resets
the per-run trace, events, and listener errors, appends the user message to its
session, and iterates up to `max_steps`. Each step builds model context from a
session snapshot and calls the model. An assistant `Message` completes the run;
one or more `ToolCall` objects are executed before the next model step.

Tool exceptions are converted into error `ToolResult` observations. A
`ModelError` and exhaustion of the step limit fail the run with a recorded end
reason. The session remains across calls to `run`; the current working tree also
allows an existing `Session` to be supplied to `Agent`.

### Model

`Model` is a structural `Protocol` with a synchronous `generate` method. It
receives context items and the registered `Tool` objects, then returns either an
assistant `Message` or a list of `ToolCall` objects. `EchoModel` and `AddModel`
are small local implementations used by tests and examples.

`DeepSeekModel` is the concrete provider adapter. It translates core messages
and tools to the OpenAI Responses client format and reads
`DEEPSEEK_API_KEY`. The Agent depends on the `Model` protocol, not this adapter.

### Tool, ToolRegistry, ToolPolicy, and ToolExecutor

`Tool` currently combines a name, description, JSON-schema-like parameters, and
the Python callable that executes the tool. M2 adds explicit `category`,
`RiskLevel`, and `side_effects` capability metadata while retaining sensible
read-only defaults for backward compatibility. `RiskLevel` contains only
`READ`, `WRITE`, `EXECUTE`, and the reserved `DESTRUCTIVE` value.

`ToolRegistry` only registers tools by name, looks them up, and lists them for the
model. Registration replaces an existing tool with the same name. It does not
execute tools, select tools, enforce policy, inspect model context, or choose an
execution backend.

`ToolExecutor` is the canonical runtime path from an Agent tool call to a tool
implementation. It looks up the tool, asks the injected `ToolPolicy` to evaluate
the tool and arguments, and invokes `Tool.execute()` only after an `ALLOW`
decision. `DENY` and `REQUIRE_APPROVAL` raise a concise `ToolPolicyError` before
the tool function can run. The Agent's existing exception handling turns that
failure into `ToolResult(is_error=True)`, allowing the model to react.
Optional synchronous decision/start callbacks let the Agent emit correctly
ordered events; the executor does not store events or own result orchestration.

`ToolPolicy` is a replaceable protocol. `DefaultToolPolicy` uses `RiskLevel`
directly and has this complete mapping:

| Risk level | Default decision |
| --- | --- |
| `READ` | `ALLOW` |
| `WRITE` | `ALLOW` |
| `EXECUTE` | `ALLOW` |
| `DESTRUCTIVE` | `DENY` |

`category` and `side_effects` remain descriptive metadata; the default mapping
does not combine them into extra rules.

`REQUIRE_APPROVAL` is a supported decision, but M5B has no approval handler, so
it safely means the tool is not executed. Tool policy remains independent of
backend selection. Command-based tool callables may delegate to the separately
injected `ExecutionBackend` port only after policy authorization.

### ContextBuilder

`ContextBuilder` returns a copy of full session history. The replaceable
`RecentContextBuilder` selects a recent slice while walking backward to a user
message so that it does not cut a tool interaction away from its initiating
turn. The Agent accepts a context builder, but there is not yet a formal
`ContextCompiler` port or token-aware compiler.

### Session

`Session` contains the ordered `Message`, `ToolCall`, and `ToolResult` history.
It owns only this in-memory domain state: `append()` extends the history and
`snapshot()` returns a new list for context/runtime consumers. The history is
the source of truth. `Session` does not own file paths, serialization, schema
versions, or persistence identity.

### Session stores

`SessionStore` is the current persistence port, with only `save(session_id,
session)` and `load(session_id)`. `MemorySessionStore` is an in-memory adapter
that deep-copies on both operations so callers cannot mutate stored state
without another save. `JsonlSessionStore` is a local JSONL adapter that writes a
complete Session snapshot for each save. The external lifecycle chooses a
session ID and explicitly loads or saves; the Agent receives only a `Session`
and never imports or invokes a concrete store.

Each JSONL file begins with schema metadata, followed by one record per item:

```json
{"type": "session_meta", "schema_version": 1}
{"type": "message", "role": "user", "content": "你好"}
{"type": "tool_call", "name": "read_file", "arguments": {"path": "a.py"}, "call_id": "1"}
{"type": "tool_result", "name": "read_file", "content": "...", "call_id": "1", "is_error": false}
```

The metadata record is not an `AgentItem`. Saves write and flush a temporary
file in the destination directory and atomically replace the target, preserving
the prior valid file when failure occurs before replacement. Unsupported
versions, malformed data, missing sessions, and filesystem failures raise
`SessionStoreError`. This is snapshot persistence, not an append-only event log.
A durable `RunRecord`, replay facility, checkpoint policy, and `TaskState` remain
future work.

### Trace

`RunTrace` is a per-run record of step outputs, associated tool results, and an
end reason (`completed`, `max_steps_exceeded`, or `model_error`). It is reset for
each `Agent.run` call, unlike the conversation session. It is useful runtime
evidence, but it is not currently a durable `RunRecord`.

### Events and listeners

The Agent emits a structured lifecycle for agent, context, model, and tool
phases. Events are retained on the Agent for the current run and synchronously
delivered to callable listeners:

```text
agent_started
  context_build_started -> context_built
  model_started -> model_completed | model_failed
  tool_policy_evaluated                (zero or more tools)
    ALLOW -> tool_started -> tool_completed
    DENY | REQUIRE_APPROVAL -> no tool execution event
agent_completed | agent_failed
```

The lifecycle inside the loop repeats for each model step. `model_failed` is
followed by `agent_failed`; an allowed tool that starts and then raises retains a
`tool_completed` event with `is_error=True`. A policy rejection creates a
model-visible `ToolResult(is_error=True)` without `tool_started` or
`tool_completed`, because tool execution never began. Events describe runtime
execution while ToolResult describes the observation supplied to the model.

Event payloads use the following current contract:

| Event | Payload |
| --- | --- |
| `agent_started` | `history_item_count` before the new user message |
| `context_build_started` | `step`, `history_item_count` |
| `context_built` | `step`, history/context counts, `context_strategy` |
| `model_started` | `step` |
| `model_completed` | `step`, `output_kind`, `tool_call_count` |
| `model_failed` | `step`, `reason`, `error_type` |
| `tool_policy_evaluated` | `step`, `name`, `call_id`, `risk_level`, `decision` |
| `tool_started` | `step`, `name`, `call_id`, `arguments_preview` |
| `tool_completed` | `step`, `name`, `call_id`, `is_error`, `duration_seconds`, `result_character_count` |
| `agent_completed` | `reason`, `step_count` |
| `agent_failed` | `reason`, `step_count` |

Policy events contain no arguments. `arguments_preview` is deterministic,
limited to eight fields and 120 characters
per value, and recursively redacts obvious sensitive keys. Events do not include
complete tool results, model reasoning, or hidden chain-of-thought. Result
character count is metadata, not content.

Listener exceptions are caught, stored in `listener_errors`, and do not stop the
run or block later listeners. The callable listener API is the current event
consumer boundary; there is no event-sink framework or persistent event log.

`RichTerminalRenderer` is an optional callable listener with explicit English
and Simplified Chinese templates. It imports Rich only from its adapter module,
and Rich is provided by the `cli` optional dependency. The Agent neither imports
Rich nor invokes terminal APIs. A renderer can be added or removed without
changing runtime execution.

### Coding tools

`coding_tools.py` remains outside the runtime loop and exposes explicit factory
functions assembled by `create_coding_tools(workspace, execution_backend=...)`:

| Tool | Category | Risk | Side effects |
| --- | --- | --- | --- |
| `list_files` | filesystem | `READ` | no |
| `search_text` | filesystem | `READ` | no |
| `read_file` | filesystem | `READ` | no |
| `write_file` | filesystem | `WRITE` | yes |
| `apply_patch` | filesystem | `WRITE` | yes |
| `run_command` | execution | `EXECUTE` | yes |
| `git_status` | git | `READ` | no |
| `git_diff` | git | `READ` | no |

Filesystem tools reject resolved paths outside the selected workspace.
`list_files` is deterministic, depth/entry bounded, and skips noisy directories
and symlinks. `search_text` performs bounded case-sensitive literal search over
small UTF-8 files and skips binary, undecodable, large, noisy-directory, and
symlink content. `apply_patch` performs one exact replacement only after proving
the old text occurs exactly once.

`run_command`, `git_status`, and `git_diff` send argv, the resolved workspace,
and a timeout through the same injected `ExecutionBackend`. Git tools retain
fixed local read-only commands without arbitrary Git arguments or remote access;
`git_diff` also disables external diff drivers and text conversion. The tool
layer converts `CommandResult` back to the existing model-facing strings and
preserves non-zero Git handling. Filesystem tools continue to use direct Python
filesystem APIs; M5A still does not introduce a filesystem backend.

### Command execution

`ExecutionBackend` is the current narrow process-execution port:

```text
                                      +-> LocalExecutionBackend -> host process
command tool -> ExecutionBackend -----|
                                      +-> DockerExecutionBackend -> container
```

`CommandResult` contains only `exit_code`, `stdout`, and `stderr`. A normal
non-zero exit remains a result. Failure to start a process, invalid local setup,
or timeout raises `ExecutionError`; the backend does not create Agent-domain
`ToolResult` objects or know how the Agent observes tool failures.

`LocalExecutionBackend` uses argv execution without a shell, captures text
stdout/stderr, enforces the supplied timeout, uses the supplied cwd, and inherits
the host environment. It runs with host process privileges and is **not a
sandbox**; it is suitable only for trusted local execution. The coding-tool
factory creates one local backend by default for backward compatibility, while
allowing a different backend to be injected.

`DockerExecutionBackend` is an explicitly selected adapter for practical
container isolation. Its trusted constructor owns the host workspace, image,
resource limits, numeric user/group IDs, and Docker executable; none of these
are model-facing tool arguments. It defaults to `python:3.12-bookworm`, which
matches the project's Python baseline and includes Git for `git_status` and
`git_diff`. `--pull never` prevents implicit image downloads, so the image must
be installed by the operator.

For every call it resolves `cwd`, rejects paths and symlinks escaping the
configured workspace, maps the workspace to writable `/workspace`, maps nested
working directories below that path, and starts a fresh named container with:

- `--rm`, network mode `none`, and no shell wrapping;
- a read-only root filesystem and a writable, bounded `/tmp` tmpfs;
- the host developer's numeric UID/GID, with a non-root fallback when the host
  identity is root;
- all Linux capabilities dropped and `no-new-privileges` enabled;
- default limits of 512 MiB memory, 1 CPU, and 128 PIDs;
- only fixed safe environment values (`HOME`, `TMPDIR`, locale, and Python
  bytecode behavior), never host environment values or secret variables; and
- exactly one host bind mount: the configured writable workspace. The Docker
  socket, host root, devices, and host namespaces are never mounted or enabled.

The Docker CLI is invoked directly with argv. Exit codes 125-127 and recognizable
daemon failures are infrastructure `ExecutionError`s; ordinary application
non-zero exits remain `CommandResult`s. On timeout the named container is
force-removed with an explicit best-effort cleanup call. Docker CLI and daemon
availability are checked only when this optional adapter is used. Unit tests do
not pull images; integration tests require the configured image to exist locally
and otherwise skip clearly.

Filesystem tools remain host-side while command and Git tools use the selected
backend. A Docker command can modify the same writable workspace immediately
visible to host-side tools. `ToolPolicy` authorization happens above both local
and Docker execution and does not inspect or select either backend.

## 3. Design principles

- Keep the kernel small and execution-first: it coordinates components rather
  than absorbing provider, UI, persistence, sandbox, or coding behavior.
- Keep concrete providers and adapters behind narrow boundaries. The kernel must
  not depend directly on DeepSeek, Rich, Docker, JSONL storage, or coding tools.
- Keep concrete plugins independent of one another. They communicate through
  stable core domain types and ports rather than importing concrete peers.
- Preserve `Session` as the source of truth. Derived state and context must be
  reproducible or replaceable without erasing original history.
- Treat context as a per-inference projection, not as the session itself.
- Evolve tool specification, selection, policy, execution, and backend concerns
  separately, in their planned milestones.
- Expose execution through structured events. Renderers and observers consume
  events; they do not drive the Agent loop.
- Isolate auxiliary failures where practical. In particular, preserve current
  listener failure isolation.
- Treat requested actions as untrusted until policy and execution boundaries
  have evaluated them.
- Default sandbox adapters to least privilege: no network or host secrets,
  non-root execution, workspace-scoped files, resource limits, timeouts, and
  explicit environment values.
- Measure changes to context and execution strategies with benchmarks before
  claiming improvement.
- Prefer inspectable, explicit control flow to framework-style indirection.
- Preserve behavior and public APIs unless a milestone deliberately changes
  semantics.

## 4. Conceptual target architecture

**Target:** four conceptual planes clarify ownership without requiring four
packages or a class for every box:

```text
                    Brain plane
 Agent Runtime | Model | Context Compiler | Tool Selector
                         |
             decisions and tool calls
                         v
                    Action plane
 Tool Catalog -> Tool Policy -> Tool Executor -> ExecutionBackend
                         |
                  results and events
                         v
                     State plane
        Session | SessionStore | TaskState | RunRecord
                         |
                 structured events
                         v
                Observability plane
        Event sinks | terminal | logs | metrics
```

- The **Brain plane** decides the next step using a model and a compiled context.
- The **State plane** preserves history and derived working state.
- The **Action plane** controls and performs requested effects.
- The **Observability plane** presents and measures behavior without controlling
  execution.

The current `Agent` spans coordination concerns that will be separated only when
their roadmap milestones require it. It delegates authorization and invocation
to `ToolExecutor` while continuing to own runtime event and result orchestration.

## 5. Plugin boundary

Some concepts are stable domain language rather than plugins. Today these
include `Message`, `ToolCall`, `ToolResult`, `AgentEvent`, `StepTrace`, and
`RunTrace`. Future stable types may include structured execution and run results.
Simple data objects should remain simple.

Capabilities with plausible alternative implementations belong behind narrow
ports. `Model`, `SessionStore`, `ExecutionBackend`, and `ToolPolicy` are
protocol-shaped ports; context building is currently replaceable by constructor
injection, and `ToolExecutor` is the small policy-enforced invocation service.
Planned ports include `ContextCompiler` and `EventSink`, introduced only as their
milestones need them.

Adapters implement those ports: for example, DeepSeek for `Model`, the current
memory and JSONL adapters for `SessionStore`, the current local and Docker
adapters for `ExecutionBackend`, or a terminal renderer for a future `EventSink`.
Concrete adapters must not import or control one another. This avoids
combinations such as a Docker backend coupled to a terminal renderer or a
context compiler coupled to JSONL persistence, and keeps each integration
replaceable and independently testable.

## 6. State, task state, and context

The intended distinction is:

```text
Session   = source-of-truth history of what happened
TaskState = derived working state for the active task (planned; not present)
Context   = per-inference projection sent to the model
```

**Current:** `Session` stores ordered agent items, and `ContextBuilder` projects
the list returned by `Session.snapshot()` for each model call. This snapshot is
an isolated in-memory list view; it is distinct from the complete durable
snapshot written by `JsonlSessionStore.save()`. `SessionStore` separates
persistence from domain state, and the external lifecycle owns `session_id` and
save/load timing. There is no `TaskState`, automatic checkpointing, durable
runtime event log, or replay.

**Target:** summaries, compacted observations, and task state can help construct
context, but they never silently replace original session history. Context
strategies remain swappable without modifying the Agent execution loop.

## 7. Tool definition and execution

**Current:** the flow is effectively:

```text
ToolCall -> ToolExecutor -> ToolRegistry lookup -> ToolPolicy
                                        |
                                     ALLOW only
                                        v
                                  Tool.function -> ToolResult

command Tool.function -> ExecutionBackend -> CommandResult
```

The same `Tool` object carries the model-facing definition, capability metadata,
and executable callable. `ToolExecutor` is the only production runtime invoker;
the registry remains discovery and lookup only. The default policy consumes
existing risk metadata without duplicating it. Command-based coding tools
delegate process execution through `ExecutionBackend`; other tools retain their
existing direct callable implementations. The Agent catches execution and policy
exceptions and turns them into error results. There is no selective exposure
consumer yet.

**Target:** responsibilities should evolve, milestone by milestone, toward:

```text
ToolSpec -> ToolSelector -> ToolExecutor -> ToolPolicy -> Tool -> ExecutionBackend
```

`ToolSpec` describes an available operation. Selection limits what the model can
see. Policy allows, denies, or requests approval. The implemented executor
manages authorized invocation, while the implemented backend port provides the
process-execution replacement point. Separate `ToolSpec` and `ToolSelector`
concepts remain future work.

## 8. Failure isolation

**Current:** listener exceptions are isolated from the Agent and from other
listeners. Tool exceptions become `ToolResult(is_error=True)` observations so a
model can react. Model request errors and maximum-step exhaustion stop the run
with explicit trace reasons and failure events.

Requested Session persistence has explicit failure behavior through
`SessionStoreError`; unlike non-critical listener failures, store failures are
not swallowed. Local process start/setup/timeout failures become
`ExecutionError`, which follows the existing Agent tool-exception path. Docker
CLI, daemon, container-start, and timeout failures use the same error boundary.
Normal non-zero process exits remain `CommandResult` values. **Target:** event
sinks, metrics, and future backends should likewise have explicit failure
behavior. Policy denial and approval-required decisions already become ordinary
tool-error observations. Where recovery is reasonable,
backend or sandbox failure should become a structured tool failure instead of
destroying the whole run. Durable session lifecycle and runtime lifecycle should
also be separable. Isolation must not hide failures: errors remain observable.

## 9. Security boundary

**Current:** every production Agent tool invocation passes through
`ToolExecutor` and its injected `ToolPolicy` before `Tool.execute()`. This
establishes the invariant that no tool side effect occurs before an `ALLOW`
decision. For command tools, it also means `ExecutionBackend.execute()` cannot
be reached after `DENY` or `REQUIRE_APPROVAL`.

Policy is capability-level, not an argument security analyzer. `RiskLevel`
describes a tool capability class, not the safety of every possible argument.
The default policy allows `EXECUTE` because tests and builds are a core coding
workflow; it does not parse argv, use command blacklists, or prove that an
arbitrary command is safe. A destructive command passed to an allowed
`run_command` tool can still damage the writable workspace.

Command-based coding tools use the replaceable `ExecutionBackend` port. The
default `LocalExecutionBackend` invokes a host subprocess, inherits the host
environment, and provides no isolation; it is retained for lightweight, trusted
local use. Explicit `DockerExecutionBackend` injection adds the practical
container controls described above without changing Agent or tool semantics.

The local backend is **not a secure sandbox**. The Docker backend materially
reduces exposure, but its writable workspace is intentionally mutable, Docker
shares the host kernel, the Docker daemon remains a privileged host component,
and container/runtime/kernel vulnerabilities remain possible. It must not be
presented as a perfect boundary for hostile multi-tenant workloads.

There is no real approval mechanism in M5B: `REQUIRE_APPROVAL` is a fail-closed
state. Future mitigations may include argument-aware policy, staged workspaces,
diff/apply approval, or restricted command profiles. Network enablement, secret
injection, and broader host access remain denied by the M5A adapter rather than
becoming model-controlled options. `ToolPolicy` knows nothing about local versus
Docker execution, and `ExecutionBackend` remains responsible only for where and
how a command runs.

## 10. Roadmap

- **M0 — Architecture Baseline:** document current and target boundaries without
  changing runtime behavior.
- **M1 — Observable Runtime / Terminal UI:** consume structured runtime events in
  an inspectable terminal experience.
- **M2 — Tool Architecture v2 and CodingToolSet:** improve tool boundaries while
  keeping coding capabilities outside the kernel.
- **M3 — State Plane / SessionStore separation:** implemented; durable snapshot
  persistence is separate from the in-memory session model.
- **M4 — ExecutionBackend abstraction:** implemented; command-based coding tools
  share a replaceable process-execution boundary.
- **M5A — Docker Sandbox v0.1:** implemented; add explicit, constrained Docker
  command execution while keeping local execution as the default.
- **M5B — ToolPolicy + ToolExecutor:** implemented; add allow, deny, and
  approval-required decisions without changing backend isolation
  responsibilities.
- **M6 — Context Compiler / token-aware context:** replace simple slicing with a
  measurable, token-aware projection strategy.
- **M7 — Structured Tool Results + Context Compaction:** improve result semantics
  and compact model context without losing source history.
- **M8 — Structured TaskState:** add explicit derived working state.
- **M9 — Selective Tool Exposure:** control which tool definitions are available
  to each inference.
- **M10 — Resume / RunRecord / Replay:** extend beyond M3's Session-level
  save/load continuation to make runtime executions recoverable and inspectable
  across process lifecycles.
- **M11 — Context Benchmark and comparison:** compare context strategies using
  task success, tokens, calls, steps, constraint violations, and recovery.

After M11, supporting work may include GitHub Actions, README improvements, an
architecture diagram, a benchmark report, a terminal demo, a security model,
limitations, and a v0.1 interview release.

## 11. Non-goals for v0.1

- Multi-agent orchestration
- A full browser agent or web UI
- Vector-database memory
- A complex autonomous planner
- A large MCP ecosystem
- Distributed workers or a microVM sandbox
- Automatic Git push or pull-request creation
- Many model providers
- Enterprise authentication or role-based access control

These exclusions keep implementation effort focused on a reliable, measurable,
inspectable single-agent runtime.
