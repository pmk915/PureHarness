# MiniHarness architecture

This document separates the implementation that exists today from the intended
architecture. Sections marked **Current** describe repository behavior through
the M16 observability and machine-interface milestone. Sections marked **Target**
describe direction, not implemented APIs.

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
                             +-> TaskStateReducer -> system state view --+
CLI/user input -> Agent -> Session.snapshot()                            +-> Model
  |                          +-> Context compiler -> trajectory view ----+    ^
  |                                                                           |
  |              ToolRegistry -> ToolSelector -> selected tool schemas -------+
  |                                                                           |
  |                                            Message or ToolCall(s) <--------+
  +-> exposure validation -> ToolExecutor -> ToolRegistry lookup
                    -> ToolPolicy -> ApprovalHandler? -> Tool callable
  +-> Session + RunTrace + AgentEvent listeners
                                  |-> human renderer
                                  `-> JSONL renderer
                    |
                    +-> finalized RunRecord -> observational replay

External lifecycle -> SessionStore <-> Session -> Agent

RunRecord -> inspect / observational replay

External benchmark -> fresh Agent workspace -> trusted verifier -> Experiment
```

The installed `miniharness` command is a thin composition layer around these
components. CLI input, terminal rendering, benchmark selection, and provider
construction do not enter the Agent kernel.

### Agent

`Agent` owns the synchronous execution loop. At the start of each run it resets
the per-run trace, events, and listener errors, appends the user message to its
session, and iterates up to `max_steps`. Each step builds model context from a
session snapshot and calls the model. It independently derives TaskState from
raw history and compiles the model-facing trajectory, then places the derived
system state view before the trajectory. An assistant `Message` completes the
run; one or more `ToolCall` objects are executed before the next model step.

Tool exceptions are converted into error `ToolResult` observations. A context
compilation error, `ModelError`, or exhaustion of the step limit fails the run
with a recorded end reason. The session remains across calls to `run`; an
existing `Session` can be supplied to `Agent`.

For every inference, after TaskState and trajectory compilation, the Agent asks
the injected `ToolSelector` for a model-facing view of the complete registry.
It validates and measures that view before emitting `model_started`. A model
call to a tool absent from that inference's view becomes an error ToolResult
before ToolExecutor is reached; this is protocol consistency, not authorization.

Each known completion or failure path finalizes one `RunRecord`. The existing
`run()` return value remains the assistant text for compatibility; callers read
the structured result from `agent.last_run_record`. Recording consumes facts
already produced by context compilation, TaskState reduction, tool selection,
execution, and the trace. It does not control any of those operations.

TaskState model injection is enabled by default. Setting
`include_task_state=False` keeps deriving the same deterministic TaskState for
ToolSelectionContext and lifecycle statistics, but does not render or prepend
the system message and records zero estimated TaskState tokens.

### Model

`Model` is a structural `Protocol` with a synchronous `generate` method. It
receives context items and the selected `Tool` objects, then returns either an
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

`ToolRegistry` only registers tools by name, looks them up, and lists its full
capability set. Registration replaces an existing tool with the same name. It
does not execute tools, select tools, enforce policy, inspect model context, or
choose an execution backend.

`ToolSelector` is the narrow per-inference visibility port. Its input is the
registry-ordered complete Tool sequence plus a frozen
`ToolSelectionContext(step, task_state)`. The context makes future deterministic
selectors possible without changing the port, but both M10 implementations
ignore TaskState: `AllToolsSelector` preserves the pre-M10 all-tools behavior,
and `StaticToolSelector` selects an explicit set of names. Static configuration
with unknown names fails; an empty set intentionally produces tool-free model
inference. No profiles, request keywords, semantic classification, model calls,
embeddings, policy lookup, or backend inspection are used.

Every selector result is validated against the exact registered Tool instances,
rejects duplicates and unregistered objects, and is normalized back to registry
order. Selection therefore never mutates or filters the registry itself. Whole
Tool definitions are the atomic exposure unit; descriptions and parameter
schemas are never truncated or rewritten.

`ToolExecutor` is the canonical runtime path from an Agent tool call to a tool
implementation. It looks up the tool and asks the injected `ToolPolicy` to
classify the tool and arguments. `ALLOW` proceeds directly; `DENY` raises a
concise `ToolPolicyError`; `REQUIRE_APPROVAL` constructs an `ApprovalRequest`
and consults the injected `ApprovalHandler`. Only `ApprovalDecision.APPROVE`
continues to `Tool.execute()`. A denied decision raises `ToolApprovalError`, and
no handler is equivalent to denial. The Agent converts both rejection paths to
distinct model-visible `ToolResult(is_error=True)` observations.

`ApprovalHandler` is a synchronous replaceable port that only obtains a host
decision. `AutoDenyApprovalHandler` is the safe automation baseline;
`AutoApproveApprovalHandler` is available only for explicit composition and
tests. `TerminalApprovalHandler` lives in a terminal adapter module, receives
injected input/output callables, displays a bounded redacted argument preview,
and approves only explicit `y`/`yes` input. It catches EOF and Ctrl+C as denial.
Neither ToolExecutor, Agent, nor ToolPolicy calls `input()`.

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

Tool policy, approval interaction, and backend execution remain independent.
Approval does not change the policy classification, and an ApprovalHandler
never executes a tool. Command-based tool callables may delegate to the
separately injected `ExecutionBackend` only after policy and, when required,
approval authorization.

ToolSelector and ToolPolicy are deliberately orthogonal: hidden does not mean
policy-denied, and exposed does not mean authorized. Calls to exposed tools
still resolve through the complete registry and pass through ToolExecutor and
ToolPolicy. A hidden call is rejected before that path only because the model
violated the schema contract it received.

### Context compiler

The retained `ContextBuilder` name now provides the canonical `compile()` path
and the FullHistory strategy. `build()` is only a compatibility view that
delegates to `compile()` and returns its items. `RecentContextBuilder` and
`TokenBudgetContextBuilder` use the same compiler semantics, so the Agent has no
parallel legacy selection path.

```text
Session.snapshot() (complete raw trajectory)
        |
        v
ContextUnit grouping
        |
        v
ToolResultProjector
  |-- IdentityToolResultProjector
  `-- DeterministicToolResultProjector
        |
        v
TokenEstimator (model-facing representation)
        |
        v
TrajectoryCompactor
  |-- IdentityTrajectoryCompactor
  `-- DeterministicToolTrajectoryCompactor
        |
        v
FullHistory | Recent | TokenBudget strategy
        |
        v
CompiledContext.items -> Model
```

`ContextUnit(items: tuple[AgentItem, ...])` is the atomic selection unit.
Ordinary messages are individual units. All contiguous `ToolCall` and
`ToolResult` items between messages form one tool-execution unit. This matches
the current interleaved multi-tool Session layout and conservatively keeps
adjacent tool-only model steps together when the Session has no boundary that
can distinguish them. Results must match an earlier call in the same unit;
missing legacy `call_id` values are matched deterministically by tool name and
order, while clearly unmatched results fail with `ContextCompileError`.

After grouping raw history, the compiler projects each `ToolResult` into a
fresh model-facing copy. `ToolResultProjector` is a narrow replaceable protocol.
`IdentityToolResultProjector` supplies an unmodified-copy baseline, while the
default `DeterministicToolResultProjector` leaves normal results unchanged and
compacts results over 12,000 characters by retaining 6,000 leading and 4,000
trailing characters around an explicit omission marker. The marker reports the
omitted, original, and retained character counts. Projection changes only
content; `name`, `call_id`, `is_error`, order, and tool-unit atomicity are
preserved. The compacting projector is the default so the normal Agent path
handles oversized results, while its threshold preserves existing small-result
behavior and explicit identity injection preserves raw-text benchmarks.

The raw `Session` objects are never changed. Projection is derived independently
for each compilation, and the JSONL persistence representation continues to
store the complete raw result. This result-level operation is not an LLM
summary, `TaskState`, or compaction of an older trajectory. Head/tail retention
is not semantic summarization and may omit important middle content. The token
estimate remains provider-neutral rather than using a provider-specific
tokenizer.

The current strategies are:

- **FullHistory:** represent every semantic unit in original order. Its
  model-facing result text may still be projected; inject the identity
  projector when a raw-text benchmark baseline is required.
- **Recent:** preserve the existing `max_items`-based recent-user-turn behavior,
  but expand selection only across whole semantic units.
- **TokenBudget:** walk newest units backward, include the newest contiguous
  suffix that fits, then return it in chronological order. If the newest
  indivisible unit alone exceeds the budget, raise `ContextBudgetExceeded`
  rather than truncating it.

Projection occurs before token estimation and strategy selection. Consequently,
`TokenEstimator` estimates the actual model-facing `AgentItem` representation
produced by the projector, not the raw `ToolResult`. This lets an otherwise
oversized unit fit a `TokenBudget` without weakening the budget. If the newest
atomic unit remains too large after projection, compilation still raises
`ContextBudgetExceeded` and does not split or repeatedly truncate the unit.

After projection and its first estimate, `TrajectoryCompactor` provides the M9
replaceable boundary. `IdentityTrajectoryCompactor` disables trajectory
compaction for benchmarks. The default
`DeterministicToolTrajectoryCompactor` performs no work at or below its
high-water trigger. Above it, the compactor walks complete projected units
backward by estimated tokens to reserve a recent raw suffix. The newest unit is
kept whole even when it alone exceeds the reserve. Every unit is therefore
entirely OLD or RECENT; multi-tool units are never split.

Only complete OLD ToolCall/ToolResult units are eligible. Each eligible unit is
replaced in place, and only when the replacement is smaller, by one explicit
system-role message headed `[MiniHarness Compacted Tool History]`. Its stable
structural lines retain tool name, success/failure, and bounded safe targets:
known path tools use `path`, `search_text` uses bounded query/path fields, and
`run_command` uses at most six bounded/redacted argv entries. Successful output
bodies and patch/file content are omitted. Failed actions may include only a
bounded generic tail. Unpaired calls, all User and Assistant messages, and all
RECENT units remain unchanged. Multiple compact blocks are intentionally kept
where natural-language messages separate tool activity, preserving chronology.

FullHistory and Recent use centralized absolute defaults: a 12,000 estimated
token high-water trigger and a 4,000-token recent raw reserve. TokenBudget uses
a centralized budget-relative default: a trigger at integer 80% of the history
budget and a recent reserve at integer one third, with safe minimums for tiny
budgets. Explicit compactor injection overrides these defaults. These estimates
are provider-neutral, and the compacted representation is re-estimated using
the same `TokenEstimator`.

The authoritative TokenBudget suffix selection runs after M9 and can still
drop complete old units—including old natural-language messages—when required
by the hard budget. M9 itself never rewrites or drops those messages. If the
newest indivisible post-projection/post-compaction unit cannot fit, the existing
`ContextBudgetExceeded` behavior remains. Deterministic tool compaction reduces
historical cost but does not guarantee that every trajectory fits every budget;
large natural-language history or a huge recent atomic unit can still prevent
that.

`TokenEstimator` is a replaceable protocol over a sequence of `AgentItem`s. The
default `ApproximateTokenEstimator` deterministically serializes Message
content, ToolCall names/arguments, and projected ToolResult content, then
applies a simple character heuristic. `CompiledContext` reports the selected
model-facing items, estimated history tokens, total/included/dropped unit
counts, strategy, optional history budget, and aggregate selected-result
projection statistics: projected/compacted result counts and raw/projected
character counts. M9 additionally reports whether trajectory compaction ran,
source-unit/action counts, original and compacted trajectory estimates, recent
raw unit/token estimates, and the compactor strategy. These are approximate
**historical trajectory** tokens only:
system instructions, tool definitions, provider wrappers, and output-token
reservation are deliberately outside the current budget.

M10 measures selected Tool schemas separately. It serializes each complete
model-facing function definition (`type`, `name`, `description`, and
`parameters`) as Unicode-safe stable JSON with sorted object keys, then applies
the same provider-neutral character heuristic used for trajectory estimates.
`estimated_tool_schema_tokens` remains separate from TaskState and history
estimates. Selection also reports registered/exposed counts, the all-tools
schema estimate, and estimated savings. None of these fields is an exact
provider input-token count.

`CompiledContext.items` remains the trajectory view and does not mix in
TaskState. By default, immediately before a model call the Agent prepends one
derived `Message(role="system")` rendered from TaskState. That message is
estimated separately through the same `TokenEstimator` as
`estimated_task_state_tokens`; `estimated_history_tokens` and an optional
history budget retain their trajectory-only meanings. M8 intentionally has no
unified provider-request budget allocator. The M12-introduced disabled mode
omits that message and records zero for this estimate without changing Session,
trajectory compilation, TaskState derivation, or selector input.

Tool selection does not belong to `CompiledContext`. Model request preparation
keeps three independently measurable components: the derived TaskState message,
the compiled trajectory, and selected complete Tool schemas. It does not label
their sum as exact input tokens because provider wrappers, instructions,
tokenization, and output reservation remain outside these estimates.

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
TaskState is not persisted: loading this unchanged schema-version-1 Session and
running the reducer reconstructs it. RunRecord serialization is separate from
this basic format.

M14 adds `JsonlDurableSessionStore` for CLI lifecycle ownership. It reuses the
same AgentItem serialization and atomic JSONL replacement boundary, but stores
one complete durable session asset per file:

```text
durable metadata (schema, identity, timestamps, workspace, model)
Session items (raw logical history)
finalized RunRecords (execution evidence for this session)
```

Run count and last-run status are derived from the RunRecords rather than
maintained as duplicate counters. The CLI stores these files under
`$MINIHARNESS_HOME/sessions`, defaulting to `~/.miniharness/sessions`; Session
itself does not know that path. Saves flush a temporary file and atomically
replace the complete prior snapshot, so conversation state and associated run
evidence cross the commit boundary together. Schema version 1 is strict;
malformed, inconsistent, or unsupported files fail closed and are not
overwritten during load.

The CLI writes a new session identity before the first Agent turn. After every
finalized Run—including `interrupted`—it saves the safe Session state and its
RunRecord. A hard process crash before that commit leaves the previous complete
snapshot. Concurrent multi-process writers for one session are not supported.

### Trace

`RunTrace` is a per-run record of step outputs, associated tool results,
structured resolved approval decisions, and an
end reason (`completed`, `max_steps_exceeded`, `context_error`, `model_error`,
`tool_selection_error`, or `interrupted`). It is reset for
each `Agent.run` call, unlike the conversation session. Explicit `to_dict()` and
`from_dict()` support make this existing trace the step-level portion of a
RunRecord rather than introducing another trace system.

### RunRecord and observational replay

`RunRecord` describes exactly one `Agent.run()` invocation. `Session` remains
the complete semantic history and may span any number of runs:

```text
                  Session
             semantic task history
                    |
            multiple Agent.run()
                    |
        +-----------+-----------+
        |                       |
        v                       v
   RunRecord A             RunRecord B
        |
        +-- RunTrace snapshot
        +-- per-inference metrics
        +-- canonical end reason
        +-- tool/context/exposure aggregates
        |
        v
observational replay (never execution replay)
```

The optional `session_id` belongs to external lifecycle metadata supplied to
Agent; it is not added to Session. A trusted runtime UUID is the default
`run_id`, with a small injectable factory for deterministic tests. Finalized
records use frozen outer value objects, immutable invocation tuples, and an
isolated trace snapshot, so a later run cannot alter an earlier record.

Each `ModelInvocationRecord` captures the step, context and selector strategy,
estimated history and TaskState tokens, registered/exposed tool counts,
estimated selected-schema tokens, and existing ToolResult/trajectory
compaction facts. Run-level sums are cumulative provider-neutral estimates,
not provider billing tokens. `model_call_count` counts real model attempts,
including attempts that raise `ModelError`. `tool_call_count` counts requests
returned by the model, including calls rejected before execution;
`tool_execution_count` counts calls that actually passed exposure and policy
checks and began execution. `tool_result_error_count` counts error observations
and is intentionally not named an execution-error count.

RunRecord JSON-compatible serialization has explicit schema version 1 and
rejects unsupported versions. It contains the RunTrace and safe structured
statistics, not a Session copy, lifecycle-event dump, full prompts or schemas,
or hidden chain-of-thought. Its duplicated `end_reason` is validated against
the canonical RunTrace value.

`replay_run(record)` returns a deterministic tuple of frozen `ReplayEntry`
values in recorded step and tool-call order. Entries expose only recorded
summaries and structured metadata. Replay imports no Agent, Model, Tool,
ToolExecutor, ToolPolicy, registry, or ExecutionBackend and performs no I/O;
it cannot rerun or reconstruct prompts, hidden details, or side effects.
Resolved approvals appear as `approval_decision` entries between their recorded
tool call and result, so evidence distinguishes approve/deny without replaying
the handler.

On `KeyboardInterrupt`, Agent finalizes the active record with
`end_reason="interrupted"` and restores Session to its pre-run length. Completed
step evidence can remain in the RunRecord, but no partial ToolCall is committed
to durable conversation state. A tool may already have produced an external
side effect when interruption arrives; the runtime deliberately records no
guess and never retries or resumes that call automatically.

### Deterministic benchmark

M12 adds a separate benchmark consumer around the runtime:

```text
Runtime
  |
  v
RunRecord
  |
  v
BenchmarkRunner
  |-- fresh fixture workspace per task/config
  |-- explicit BenchmarkConfig
  |-- independent argv verification oracle
  `-- BenchmarkResult
         |
         +-> JSONL
         `-> per-config summary
```

`BenchmarkTask` is inspectable static metadata: task ID, prompt, canonical
fixture path, verification argv, and optional curated tool-name metadata.
`BenchmarkConfig` names one explicit combination of context strategy,
ToolResult projection, TaskState injection, trajectory compaction, and tool
exposure. The four standard configurations are `raw_baseline`,
`budget_only`, `context_engineered`, and `full_miniharness`; they are not
generated as a factorial matrix.

For each task/config pair, the runner copies the canonical fixture into a new
temporary directory and constructs coding tools bound only to that copy.
Trusted verifier source lives outside the fixture. The runner snapshots it
before Agent execution, verifies that the canonical source remains unchanged,
and materializes the snapshot at a separate temporary path only after the Agent
stops. Workspace-local verifier edits therefore cannot forge success. It
creates a fresh model through the caller's model factory and runs cases
serially in task-then-config order. Agent failures that have a finalized
RunRecord still proceed to verification. Missing fixtures, workspace-copy
failures, invalid construction, verifier start/timeout failures, verifier
integrity failures, or an Agent failure without a RunRecord are benchmark
infrastructure errors.

Verification is a separate low-level argv execution with no shell and a
centralized 30-second timeout. It does not pass through ToolSelector,
ToolExecutor, or ToolPolicy because it judges the system under evaluation
rather than acting as that system. A zero verifier exit code alone defines
`task_success`. Therefore:

```text
end_reason = runtime outcome
task_success = external oracle outcome

completed != task_success
```

`BenchmarkResult` schema version 1 embeds the M11 RunRecord rather than
recomputing runtime metrics, retains bounded verifier output previews, and
records minimal stable configuration identity. JSONL output uses one result per
line. Frozen per-config summaries report raw success counts/rates, end reasons,
calls, steps, cumulative estimated model-facing token categories, and
compaction counts. No composite score, provider-exact usage, model comparison,
LLM judge, parallel runner, or generic RunRecordStore is present.

### Experiment runner and CLI

**Current:** `ExperimentRunner` composes the deterministic benchmark API into a
serial repeated-trial matrix. Its ordering is repetition, task, then
configuration. Every case still delegates to `BenchmarkRunner`, which creates a
fresh model and workspace. `ExperimentResult` wraps the complete
`BenchmarkResult` with model identity, one-based repetition, and monotonic case
duration. Versioned JSONL is the machine-readable evidence format; per-config
summaries report success, duration, runtime counts, estimated token totals, and
compaction counts. Provider-exact usage is intentionally absent because the
generic Model protocol does not currently expose it.

`miniharness` provides five thin command paths plus interactive mode:

```text
miniharness                 one Session, repeated Agent.run() turns
miniharness run PROMPT      one Agent run
miniharness inspect PATH    read-only RunRecord inspection
miniharness benchmark       BenchmarkRunner + ExperimentRunner
miniharness sessions        discover durable sessions, newest first
miniharness resume ID       restore a logical Session, then await input
```

Interactive `/help`, `/status`, and `/exit` are CLI concerns. Structured Agent
events feed a terminal renderer, and the renderer never controls execution. A
Session spans interactive turns and process restarts, while each turn has a
distinct RunRecord. Resume is passive until a new ordinary message lazily
creates the provider and Agent with the loaded Session. It invokes no historical
model or tool work. Ctrl+D exits cleanly; Ctrl+C cancels prompt input or marks
the active run interrupted before returning to the prompt.

Interactive Agent construction injects `TerminalApprovalHandler` using the
REPL's input/output adapters. It is called only after `REQUIRE_APPROVAL`.
One-shot `run` intentionally has no interactive handler and therefore fails
closed rather than blocking a pipe or CI job.

One-shot `run --output jsonl` replaces the human listener with the JSONL
listener and suppresses final response prose. `inspect --json` writes the
existing versioned RunRecord document, while `sessions --json` writes a
versioned newest-first summary derived from durable metadata and RunRecords.
Machine-mode stdout contains only JSON/JSONL; CLI diagnostics use stderr.
Interactive mode is intentionally not a JSON control protocol.

The default command-line provider adapter is DeepSeek, but the Agent continues
to depend only on the Model protocol. Deterministic tests inject scripted model
factories and make no network calls. Real-model trials are explicit, manual,
nondeterministic, and potentially paid.

### Events and listeners

The Agent emits a structured lifecycle for agent, context, model, approval, and
tool phases. Every `AgentEvent` captures an aware UTC occurrence timestamp and
the Agent attaches the active `run_id` and optional `session_id` before
observers receive it. Events are retained on the Agent for the current run and
synchronously delivered to callable listeners:

```text
agent_started
  context_build_started -> context_built | context_build_failed
  selection -> model_started -> model_completed | model_failed
  tool_policy_evaluated                (zero or more tools)
    ALLOW -> tool_started -> tool_completed
    DENY -> model-visible denial, no approval/tool execution event
    REQUIRE_APPROVAL -> approval_requested
      APPROVE -> approval_granted -> tool_started -> tool_completed
      DENY -> approval_denied -> model-visible denial, no tool execution event
agent_completed | agent_interrupted | agent_failed
```

The lifecycle inside the loop repeats for each model step. `model_failed` is
followed by `agent_failed`; an allowed tool that starts and then raises retains a
`tool_completed` event with `is_error=True`. A policy rejection creates a
model-visible `ToolResult(is_error=True)` without `tool_started` or
`tool_completed`, because tool execution never began. Approval rejection has
the same no-execution property but distinct approval events and error text.
Events describe runtime
execution while ToolResult describes the observation supplied to the model.
`context_build_failed` is followed by `agent_failed`, and no model request is
made with partial or malformed context. Invalid selector configuration/output
emits `agent_failed(reason="tool_selection_error")` without `model_started` or a
model request; M10 adds no retry behavior.

Event payloads use the following current contract:

| Event | Payload |
| --- | --- |
| `agent_started` | `history_item_count` before the new user message |
| `context_build_started` | `step`, `history_item_count` |
| `context_built` | `step`, history/final-context/trajectory counts, strategy, separate estimated history and TaskState tokens, safe TaskState aggregate counts, total/included/dropped units, projected/compacted result counts, raw/projected result character counts, aggregate trajectory-compaction statistics, optional history budget |
| `context_build_failed` | `step`, `reason`, `error_type` |
| `model_started` | `step`, selector strategy, registered/exposed counts, selected/all schema-token estimates, estimated savings |
| `model_completed` | `step`, `output_kind`, `tool_call_count` |
| `model_failed` | `step`, `reason`, `error_type` |
| `tool_policy_evaluated` | `step`, `name`, `call_id`, `risk_level`, `decision` |
| `approval_requested` | `step`, `name`, `call_id`, redacted `arguments_preview` |
| `approval_granted` / `approval_denied` | `step`, `name`, `call_id`, `approval_decision` |
| `tool_started` | `step`, `name`, `call_id`, `arguments_preview` |
| `tool_completed` | `step`, `name`, `call_id`, `is_error`, `duration_seconds`, `result_character_count` |
| `agent_completed` | `reason`, `step_count` |
| `agent_interrupted` | `reason`, `step_count` |
| `agent_failed` | `reason`, `step_count` |

Policy events contain no arguments. `arguments_preview` is deterministic,
limited to eight fields and 120 characters
per value, and recursively redacts obvious sensitive keys. Events do not include
complete tool results, model reasoning, or hidden chain-of-thought. Result
character count is metadata, not content. Selection metrics contain no Tool
names, descriptions, parameter schemas, or user text.

Listener exceptions are caught, stored in `listener_errors`, and do not stop the
run or block later listeners. The callable listener API is the current event
consumer boundary; there is no event-sink framework or persistent event log.

`RichTerminalRenderer` is an optional callable listener with explicit English
and Simplified Chinese templates. It imports Rich only from its adapter module,
and Rich is provided by the `cli` optional dependency. The Agent neither imports
Rich nor invokes terminal APIs. A renderer can be added or removed without
changing runtime execution. Its context summary includes one concise count of
compacted tool outputs and one concise TaskState line with modified-file and
recent-error counts. When M9 actually replaces old tool units, it adds exactly
one concise trajectory-compaction line; it never prints the derived history.
For each `model_started`, it adds one concise selected/all tool count and
approximate schema-token line without listing tool names or schemas.

M16 adds an independent `JsonlEventRenderer`. It maps every current public event
name explicitly into live-event wire schema version 1, preserves observation
order, uses the event's occurrence timestamp, promotes `step` to an optional
top-level field, and emits event-specific sanitized data under `payload`.
Unknown events, missing run identity, non-finite numbers, and unsupported Python
objects fail serialization rather than falling back to `repr`. JSONL rendering
does not alter Agent control flow; the CLI detects listener failure after the
run and reports an application/output error.

The live wire schema is not the RunRecord persistence schema. Live events are
transient execution observations; RunRecord remains finalized versioned
evidence, and replay remains a side-effect-free ordered reconstruction of that
evidence.

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
 Tool Catalog -> Tool Executor -> Tool Policy
                         |-> ApprovalHandler?
                         `-> Tool -> ExecutionBackend
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
include `Message`, `ToolCall`, `ToolResult`, `AgentEvent`, `StepTrace`,
`ApprovalRequest`, `ApprovalTrace`, `RunTrace`, `RunRecord`,
`ModelInvocationRecord`, and `ReplayEntry`.
Simple data objects should remain simple.

Capabilities with plausible alternative implementations belong behind narrow
ports. `Model`, `SessionStore`, `ExecutionBackend`, `ToolPolicy`,
`ApprovalHandler`,
`TokenEstimator`, `ToolResultProjector`, `TrajectoryCompactor`, and
`ToolSelector` are protocol-shaped ports. Context compilation strategies,
result projection, trajectory compaction, and tool exposure are replaceable by
constructor injection, and `ToolExecutor` is the small
policy-enforced invocation service. `EventSink` remains a planned port.

Adapters implement those ports: for example, DeepSeek for `Model`, the current
memory and JSONL adapters for `SessionStore`, the current local and Docker
adapters for `ExecutionBackend`, the terminal/automatic approval handlers, or a
terminal renderer for a future `EventSink`.
Concrete adapters must not import or control one another. This avoids
combinations such as a Docker backend coupled to a terminal renderer or a
context compiler coupled to JSONL persistence, and keeps each integration
replaceable and independently testable.

## 6. State, task state, and context

The intended distinction is:

```text
Session   = source-of-truth history of what happened
TaskState = derived working state for the active task
Context   = per-inference projection sent to the model
```

```text
                 Session
              raw source truth
               /          \
              v            v
    TaskStateReducer    Context compiler
              |            |
              v            v
         TaskState     trajectory view
              \            /
               v          v
                model context
```

**Current:** `TaskStateReducer.reduce(raw_items)` deterministically rebuilds a
frozen `TaskState` from the complete raw Session snapshot for every model step.
It records a bounded current request, bounded recent successful and failed
action identities, stable first-seen unique file-read/file-modification paths,
and bounded `recent_errors`. The latter means the most recent failed tool
observations, not logically unresolved errors. Action records retain only tool
name and existing call ID; they do not duplicate arguments or result bodies.

The reducer uses the same shared ToolCall/ToolResult matching rules as context
grouping. Only successful `read_file` calls contribute `files_read`; only
successful `write_file` and `apply_patch` calls contribute `files_modified`.
Paths come from structured `path` arguments and receive lexical POSIX
normalization. Arbitrary `run_command` argv is never parsed to infer filesystem
state. Failed operations remain failed actions/errors but do not claim a
successful read or modification.

TaskState contains no inferred goal, constraints, plan, pending actions,
priority, intent, or reasoning. It uses no LLM, embeddings, filesystem scan,
provider behavior, execution backend, or ToolPolicy decision. Current-request
and error text use deterministic head/middle-omission/tail bounds; completed
actions keep the latest 20, failed actions the latest 10, and recent errors the
latest 5. File tuples remain stable unique collections for the current scale.

The model-facing renderer produces one deterministic system-role message with
the current request, file lists, compact action identities, and recent errors.
The Agent prepends it to the independently compiled trajectory without writing
it to Session. Loading a Session and reducing it again recreates the same state;
there is no TaskState persistence schema or cache.

Current limitations are deliberate: runtime facts are not semantic task
understanding; successful arbitrary commands may change files without appearing
in TaskState; paths are based on trusted structured tool arguments rather than a
workspace rescan; file collections are not yet capped; token estimates remain
provider-neutral approximations. M9 compacts only old complete tool execution,
not natural-language history, and it is an ephemeral context view rather than a
persisted summary. M10 selectors are static/configuration-driven and do not
infer required tools from TaskState. There is no automatic checkpointing,
durable runtime event log, or execution replay.

## 7. Tool definition and execution

**Current:** model exposure and execution are separate flows:

```text
ToolRegistry -> ToolSelector -> selected complete schemas -> Model

Model ToolCall -> exposure validation -> ToolExecutor
                                      -> ToolRegistry lookup -> ToolPolicy
                                        |
                                     ALLOW only
                                        v
                                  Tool.function -> ToolResult

command Tool.function -> ExecutionBackend -> CommandResult
```

The same `Tool` object carries the model-facing definition, capability metadata,
and executable callable. `ToolExecutor` is the only production runtime invoker;
the registry remains the complete discovery and lookup source. The selector
chooses only the model-visible view. The default policy consumes
existing risk metadata without duplicating it. Command-based coding tools
delegate process execution through `ExecutionBackend`; other tools retain their
existing direct callable implementations. The Agent catches execution and policy
exceptions and turns them into error results. `AllToolsSelector` and
`StaticToolSelector` provide benchmark-ready visibility baselines.

**Target:** responsibilities should evolve, milestone by milestone, toward:

```text
ToolSpec -> ToolSelector -> ToolExecutor -> ToolPolicy -> Tool -> ExecutionBackend
```

`ToolSpec` describes an available operation. Selection limits what the model can
see. Policy allows, denies, or requests approval. The implemented selector
still consumes the current combined Tool object; splitting ToolSpec remains
future work. The implemented executor manages authorized invocation, while the
implemented backend port provides the process-execution replacement point.

## 8. Failure isolation

**Current:** listener exceptions are isolated from the Agent and from other
listeners. Tool exceptions become `ToolResult(is_error=True)` observations so a
model can react. Context compilation errors stop before the model request and
emit `context_build_failed` followed by `agent_failed`. Model request errors and
maximum-step exhaustion also stop the run with explicit trace reasons and
failure events.

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
establishes the invariant that no tool side effect occurs after `DENY`, or after
`REQUIRE_APPROVAL` without an explicit `APPROVE` decision. For command tools,
`ExecutionBackend.execute()` cannot be reached before both applicable gates.

Policy is capability-level, not an argument security analyzer. `RiskLevel`
describes a tool capability class, not the safety of every possible argument.
The default policy allows `EXECUTE` because tests and builds are a core coding
workflow; it does not parse argv, use command blacklists, or prove that an
arbitrary command is safe. A destructive command passed to an allowed
`run_command` tool can still damage the writable workspace.

Selective exposure is not a security boundary. A hidden-tool call is rejected
for model/runtime contract consistency, but authorization of every exposed call
still belongs to ToolPolicy. Selector configuration must not be used as proof
that an operation is safe.

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

M15 approval is authorization, not a safety proof: an approved command is not
necessarily sandboxed, reversible, or idempotent. Decisions are one-time and
are not persisted as trust rules. Resume does not reopen an old prompt or retry
an approval-gated call. Approval granted before an interrupted tool still leaves
unknown side-effect state and is never treated as safe to retry. Future
mitigations may include argument-aware policy, staged workspaces, diff/apply
approval, or restricted command profiles. `ToolPolicy` knows nothing about
local versus Docker execution, and `ExecutionBackend` remains responsible only
for where and how a command runs.

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
- **M6 — Context Compiler v1:** implemented; select raw semantic units through
  measurable FullHistory, Recent, and TokenBudget strategies.
- **M7 — Structured Tool Output:** implemented; deterministically project
  oversized model-facing ToolResult text without losing raw Session history.
- **M8 — Deterministic TaskState v1:** implemented; rebuild bounded runtime-fact
  state from raw Session and prepend a non-persistent model-facing state view.
- **M9 — Deterministic Trajectory Compaction v1:** implemented; replace complete
  old tool-execution units with smaller structural context while preserving
  recent units, natural-language messages, and durable raw Session history.
- **M10 — Selective Tool Exposure v1:** implemented; choose and measure complete
  model-facing Tool definitions per inference while preserving ToolPolicy as
  the independent authorization boundary.
- **M11 — RunRecord + Observational Replay:** implemented; capture one
  serializable structured record per run and inspect it deterministically
  without replaying models, tools, backends, or side effects. Durable Session
  resume remains a separate lifecycle responsibility implemented in M14.
- **M12 — Deterministic Context Benchmark:** implemented; compare four explicit
  runtime/context configurations over isolated curated fixtures using an
  external argv oracle, M11 RunRecords, JSONL results, and multidimensional
  per-config summaries.
- **M13 — v0.1 Evidence and CLI Release:** implemented; harden trusted
  verifiers, add controlled repeated real-model experiment support and JSONL
  evidence, provide an installed multi-turn CLI, and document/test/package the
  release boundary.
- **M14 — Durable Session Runtime:** implemented; atomically persist interactive
  identity, raw Session history, and associated RunRecords; discover and resume
  sessions without replaying historical execution; formalize interruption.
- **M15 — Human Approval & Action Safety:** implemented; resolve
  approval-required tool calls through a replaceable fail-closed handler, emit
  and persist structured decisions, and provide one-time terminal approval.
- **M16 — Observability & Machine Interface:** implemented; expose occurrence-
  time runtime events as versioned JSONL and provide JSON views of RunRecord
  evidence and durable session summaries without coupling runtime to the CLI.

Future work may address concurrent session writers, session migration or
branching, workspace relocation, and stronger cancellation of synchronous
provider/backend operations. These are not current capabilities.

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

Future selector benchmarks should compare AllTools, Static, and later smarter
strategies using task success, schema cost, selection mistakes, and
required-tool recall: whether every capability actually required for task
success was exposed. M10 does not implement an oracle or benchmark scorer.
