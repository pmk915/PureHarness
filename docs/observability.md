# MiniHarness observability

MiniHarness exposes one runtime through human-readable renderers and small,
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
miniharness run "fix the failing test" --output jsonl
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
miniharness inspect run.json
```

The machine form writes one JSON document using the existing versioned
RunRecord persistence serializer:

```bash
miniharness inspect run.json --json
```

Durable Session discovery has a separate versioned summary document:

```bash
miniharness sessions --json
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

The live-event wire schema and RunRecord persistence schema both currently use
version `1`, but they are not the same schema and do not share a lifecycle.
Inspection and replay never call a model, policy, approval handler, tool, or
execution backend.

## Scope and limitations

M16 provides local JSON and JSONL output only. It does not provide an
interactive JSON protocol, RPC, remote telemetry, OpenTelemetry, Jaeger,
Prometheus, a metrics database, distributed tracing, or exact provider token
accounting.
