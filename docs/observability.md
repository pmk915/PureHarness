# PureHarness observability

PureHarness exposes one runtime through human-readable renderers and small,
versioned local machine interfaces. These adapters observe execution; they do
not select tools, make policy or approval decisions, or control the Agent loop.

## Runtime events

`AgentEvent` is the live source of truth for lifecycle observations. Every
runtime-emitted event has:

- an existing stable lowercase event name;
- an aware UTC occurrence timestamp;
- the active Run ID;
- the Session ID when one is available;
- event-specific structured data.

The Agent attaches identity before synchronously delivering an event to each
listener. Listener failure remains isolated from policy and tool execution.

## Human rendering

Interactive mode and default one-shot execution use the existing plain or Rich
terminal renderers. Human wording and localization are presentation details and
are not a wire protocol.

## JSONL live-event schema version 1

Use:

```bash
pureharness run "fix the failing test" --output jsonl
```

Each non-empty stdout line is one compact JSON object:

```json
{"schema_version":1,"event":"tool_started","timestamp":"2026-09-20T10:00:00Z","run_id":"...","session_id":"...","step":0,"payload":{"tool_name":"run_command","call_id":"...","arguments_preview":{"argv":"[\"pytest\"]"}}}
```

Top-level fields are:

| Field | Meaning |
| --- | --- |
| `schema_version` | Live-event wire version, currently `1` |
| `event` | Stable runtime event name |
| `timestamp` | Event occurrence time as UTC ISO 8601 |
| `run_id` | Run identity attached by Agent |
| `session_id` | Optional logical Session identity |
| `step` | Optional model-step index for step-specific events |
| `payload` | Event-specific JSON-safe data |

### Compatibility within version 1

The version 1 event-name namespace is additive. Producers may add new event
names while preserving the stable top-level envelope. Consumers must tolerate
unknown event names by ignoring them or handling them generically. Within
version 1, existing event names are not removed or renamed, existing fields are
not removed or reinterpreted incompatibly, and optional payload fields may be
added. Terminal-event meanings remain stable.

`model_retrying` is a non-terminal event. It means an attempt for the current
logical model request failed and the request will be attempted again. Its
payload includes the next `attempt`, `max_attempts`, `error_type`, and
`failure_category`; `step` remains in the stable top-level envelope.
`model_failed` continues to mean that the logical model request finally failed
and no recovery will continue.

`execution_budget_exhausted` is a terminal Run event emitted when the runtime
refuses to start a physical model attempt or an exposed tool-call batch because
its cumulative Run limit would be exceeded. It is followed by `agent_failed`
with end reason `execution_budget_exceeded`. Its payload contains `resource`,
`used`, and `limit`; tool-batch refusal also includes `requested` and
`remaining`. Because the refused action never starts, this path emits no
`model_failed`, fabricated ToolResult, or `context_build_failed` for the budget
refusal. The event is additive within JSONL wire schema version 1.

The three context-failure events have distinct meanings:

```text
context_build_failed
    = local context compilation failed before that context could be sent
context_window_exceeded
    = the provider rejected an already built and sent context
context_recovering
    = bounded local context recovery is being attempted
```

`context_window_exceeded` is an occurrence event, not inherently terminal. Its
payload contains the normalized `error_type`, whether
`context_recovery_available`, `max_context_recoveries`, and the optional
one-based `recovery_attempt` that will begin when recovery remains available.
It contains no raw provider error text.

`context_recovering` is likewise non-terminal. It means the overflow is being
handled by rebuilding smaller history within the same logical Agent step. Its
payload contains:

| Field | Meaning |
| --- | --- |
| `recovery_attempt` | One-based context rebuild attempt |
| `max_recoveries` | Independent configured rebuild limit |
| `error_type` | Normalized overflow exception type |
| `previous_history_tokens` | History estimate sent on the rejected attempt |
| `recovery_history_budget` | Emergency bounded-compilation budget |
| `recovered_history_tokens` | Successful rebuilt history estimate, when available |
| `previous_estimated_request_tokens` | Previous history + TaskState + schemas |
| `recovered_estimated_request_tokens` | Rebuilt history + unchanged TaskState + schemas, when available |

Successful recovery emits no terminal `model_failed` or
`context_build_failed`. If rebuilding is impossible, `context_recovering`
omits the unavailable recovered estimates and is followed by
`context_build_failed` because the local recovery compilation failed. If
recovery is disabled or the provider rejects the single recovered context
again, `context_window_exceeded` is followed by `agent_failed` without
`context_build_failed`. Both M18.4B events are additive within live-event JSONL
schema version 1.

When explicit context limits are configured, `context_built` adds these optional
payload fields:

| Field | Meaning |
| --- | --- |
| `context_window_tokens` | Explicit configured total context capacity |
| `reserved_output_tokens` | Capacity held back for model output |
| `usable_input_tokens` | Window minus the output reserve |
| `estimated_request_tokens` | Final history + TaskState + exposed-schema estimate |
| `context_pressure_detected` | Whether the normal candidate exceeded usable input |
| `available_history_tokens` | Usable input minus TaskState and exposed schemas |
| `bounded_history_applied` | Whether history was recompiled under that bound |

These fields describe the final context accepted for the model request. Token
values are deterministic provider-neutral estimates, not provider billing or
exact tokenizer counts. They are absent when limits are not configured. Adding
them is compatible with JSONL schema version 1 because payload fields are
optional and version 1 consumers must tolerate additive fields.

All current public event types have explicit mappings. Tool argument data uses
the same bounded, recursively redacted preview as human observability. The
serializer does not expose environment variables, credentials, arbitrary
objects, Python reprs, full ToolResults, or hidden model reasoning.

One-shot mode has no terminal ApprovalHandler. A `REQUIRE_APPROVAL` decision
therefore emits requested/denied evidence and fails closed without reading
stdin or printing an approval prompt.

## stdout, stderr, and exit codes

In JSONL mode stdout contains only JSONL events. Human responses, status lines,
and record-path notices are suppressed. CLI configuration, input, persistence,
and output-adapter errors use stderr. Exit codes continue to represent overall
command success or failure; consumers should not infer command success only
from the last event.

If event serialization or output fails, the Agent's listener isolation still
protects execution semantics. The CLI then reports an application/output error
and returns nonzero rather than silently treating the stream as complete.

## Persisted evidence interfaces

Human RunRecord inspection remains:

```bash
pureharness inspect run.json
```

The machine form writes one JSON document using the existing versioned
RunRecord persistence serializer:

```bash
pureharness inspect run.json --json
```

Durable Session discovery has a separate versioned summary document:

```bash
pureharness sessions --json
```

It contains newest-first Session summaries derived from durable metadata and
RunRecords. An empty store produces `{"schema_version":1,"sessions":[]}`.

## Live events, evidence, and replay

These surfaces are deliberately distinct:

```text
Live events = observations emitted while execution happens
RunRecord    = finalized persisted evidence for one Agent.run()
Replay       = read-only ordered reconstruction from RunRecord
```

`ExecutionUsage` is live per-run diagnostic state rather than persisted
evidence. RunRecord v1 retains its original six-value closed end-reason contract.
RunRecord v2 has the same persisted structure and adds only
`execution_budget_exceeded`; current writers emit v2 and current readers accept
and preserve v1 and v2. Older PureHarness versions are not expected to read v2.
In both versions, `model_call_count == len(model_invocations)` still counts
logical requests, while physical retry/context-recovery attempts and accepted
dispatch usage stay in `Agent.execution_usage` and runtime events.

The live-event wire schema remains version 1 independently of RunRecord
persistence version 2; they do not share a version or lifecycle.
Inspection and replay never call a model, policy, approval handler, tool, or
execution backend.

## Scope and limitations

M16 provides local JSON and JSONL output only. It does not provide an
interactive JSON protocol, RPC, remote telemetry, OpenTelemetry, Jaeger,
Prometheus, a metrics database, distributed tracing, or exact provider token
accounting.
