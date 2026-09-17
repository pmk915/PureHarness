# MiniHarness architecture

This document separates the implementation that exists today from the intended
architecture. Sections marked **Current** describe repository behavior at the M0
baseline. Sections marked **Target** describe direction, not implemented APIs.

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
Agent -> Session snapshot -> ContextBuilder -> Model
  |                                         |
  |               Message or ToolCall(s) <--+
  |                         |
  +-> ToolRegistry -> Tool callable
  |
  +-> Session + RunTrace + AgentEvent listeners
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

### Tool and ToolRegistry

`Tool` currently combines a name, description, JSON-schema-like parameters, and
the Python callable that executes the tool. `ToolRegistry` registers tools by
name, lists them for the model, and dispatches execution. Registration replaces
an existing tool with the same name.

This is intentionally simpler than the target tool architecture. There is no
separate selector, policy, executor, approval flow, or execution backend today.

### ContextBuilder

`ContextBuilder` returns a copy of full session history. The replaceable
`RecentContextBuilder` selects a recent slice while walking backward to a user
message so that it does not cut a tool interaction away from its initiating
turn. The Agent accepts a context builder, but there is not yet a formal
`ContextCompiler` port or token-aware compiler.

### Session

`Session` contains the ordered `Message`, `ToolCall`, and `ToolResult` history.
It can return a snapshot and currently implements JSONL save/load directly.
This history is the source of truth; persistence currently requires an explicit
JSONL save. A separate `SessionStore`, run record, replay facility, and
`TaskState` do not yet exist.

### Trace

`RunTrace` is a per-run record of step outputs, associated tool results, and an
end reason (`completed`, `max_steps_exceeded`, or `model_error`). It is reset for
each `Agent.run` call, unlike the conversation session. It is useful runtime
evidence, but it is not currently a durable `RunRecord`.

### Events and listeners

The Agent emits structured lifecycle events for agent start/completion/failure,
model completion, and tool start/completion. Events are retained on the Agent
for the current run and synchronously delivered to listeners. Their current
`data` payloads are small dictionaries containing step and lifecycle metadata;
they do not yet form a complete typed record of model outputs and tool results.

Listener exceptions are caught, stored in `listener_errors`, and do not stop the
run or block later listeners. There is no event-sink protocol, terminal renderer,
metrics backend, or persistent event log yet.

### Coding tools

`coding_tools.py` provides workload-specific factories for UTF-8 file reading,
file writing, and argv-based local command execution. File tools reject resolved
paths outside the selected workspace; commands run with that workspace as their
current directory and have a timeout. These tools are outside the runtime loop
and are assembled by the coding demo.

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
- Default future sandboxes to least privilege: no network, no secrets, non-root
  execution, workspace-scoped files, resource limits, timeouts, and explicit
  environment allowlists.
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
their roadmap milestones require it. M0 does not introduce these abstractions.

## 5. Plugin boundary

Some concepts are stable domain language rather than plugins. Today these
include `Message`, `ToolCall`, `ToolResult`, `AgentEvent`, `StepTrace`, and
`RunTrace`. Future stable types may include structured execution and run results.
Simple data objects should remain simple.

Capabilities with plausible alternative implementations belong behind narrow
ports. `Model` is already a protocol-shaped port; context building is currently
replaceable by constructor injection. Planned ports include `ContextCompiler`,
`SessionStore`, `ToolExecutor`, `ExecutionBackend`, `EventSink`, and
`ToolPolicy`, introduced only as their milestones need them.

Adapters implement those ports: for example, DeepSeek for `Model`, JSONL for a
future `SessionStore`, Docker for a future `ExecutionBackend`, or a terminal
renderer for a future `EventSink`. Concrete adapters must not import or control
one another. This avoids combinations such as a Docker backend coupled to a
terminal renderer or a context compiler coupled to JSONL persistence, and keeps
each integration replaceable and independently testable.

## 6. State, task state, and context

The intended distinction is:

```text
Session   = source-of-truth history of what happened
TaskState = derived working state for the active task (planned; not present)
Context   = per-inference projection sent to the model
```

**Current:** `Session` stores ordered agent items, and `ContextBuilder` projects
that history for each model call. There is no `TaskState`. JSONL persistence is a
method on `Session`, not a separate store.

**Target:** summaries, compacted observations, and task state can help construct
context, but they never silently replace original session history. Context
strategies remain swappable without modifying the Agent execution loop.

## 7. Tool definition and execution

**Current:** the flow is effectively:

```text
ToolCall -> ToolRegistry -> Tool.function -> ToolResult
```

The same `Tool` object carries both the model-facing definition and executable
callable. The Agent catches execution exceptions and turns them into error
results.

**Target:** responsibilities should evolve, milestone by milestone, toward:

```text
ToolSpec -> ToolSelector -> ToolPolicy -> ToolExecutor -> ExecutionBackend
```

`ToolSpec` describes an available operation. Selection limits what the model can
see. Policy allows, denies, or requests approval. The executor manages invocation
and structured results. The backend provides the execution environment. This is
direction only: none of these new abstractions is introduced in M0.

## 8. Failure isolation

**Current:** listener exceptions are isolated from the Agent and from other
listeners. Tool exceptions become `ToolResult(is_error=True)` observations so a
model can react. Model request errors and maximum-step exhaustion stop the run
with explicit trace reasons and failure events.

**Target:** event sinks, renderers, metrics, persistence, policy, and execution
backends should have explicit failure behavior. Where recovery is reasonable,
backend or sandbox failure should become a structured tool failure instead of
destroying the whole run. Durable session lifecycle and runtime lifecycle should
also be separable. Isolation must not hide failures: errors remain observable.

## 9. Security boundary

**Current:** `create_run_command_tool` invokes a local subprocess directly. It
sets the working directory and a timeout, but it does not provide container or OS
isolation, filter the inherited environment, disable network access, prevent
arbitrary process behavior, or enforce resource limits. Workspace path checks on
the file tools do not turn local command execution into a secure sandbox.

Therefore the current local execution capability is **not a secure sandbox** and
must not be described as one. In particular, secrets such as API credentials may
be present in the parent environment and should not be exposed to untrusted
commands.

**Target:** M5 may add a Docker-backed execution adapter and tool policy with
least-privilege defaults: no network or secrets by default, non-root execution,
workspace-scoped mounts, resource limits, timeouts, and explicit environment
allowlists. Docker is a future adapter, not a kernel dependency, and this
high-level direction is not a security guarantee.

## 10. Roadmap

- **M0 — Architecture Baseline:** document current and target boundaries without
  changing runtime behavior.
- **M1 — Observable Runtime / Terminal UI:** consume structured runtime events in
  an inspectable terminal experience.
- **M2 — Tool Architecture v2 and CodingToolSet:** improve tool boundaries while
  keeping coding capabilities outside the kernel.
- **M3 — State Plane / SessionStore separation:** separate durable persistence
  from the in-memory session model.
- **M4 — ExecutionBackend abstraction:** define a replaceable execution boundary.
- **M5 — Docker Sandbox + ToolPolicy:** introduce controlled, least-privilege
  execution and policy decisions.
- **M6 — Context Compiler / token-aware context:** replace simple slicing with a
  measurable, token-aware projection strategy.
- **M7 — Structured Tool Results + Context Compaction:** improve result semantics
  and compact model context without losing source history.
- **M8 — Structured TaskState:** add explicit derived working state.
- **M9 — Selective Tool Exposure:** control which tool definitions are available
  to each inference.
- **M10 — Resume / RunRecord / Replay:** make runs recoverable and inspectable
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
