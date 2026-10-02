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

`workspace_precondition_failed` is emitted when a standard structured file
mutation is blocked by read-before-edit or stale-file protection. It is an
occurrence event, not a terminal Run failure. `step` remains in the top-level
envelope; its payload is:

| Field | Meaning |
| --- | --- |
| `tool_name` | Blocked structured mutation tool |
| `call_id` | Provider call identity, when supplied |
| `path` | Canonical workspace-relative target; never a host absolute path |
| `reason` | `read_required` or `stale_observation` |
| `change` | Optional stale fact: `content_changed`, `target_missing`, or `target_appeared` |

Initial prepare failures occur after the exposed batch has passed
execution-budget preflight and progress has observed the attempted action, but
before ToolPolicy, approval, or underlying tool execution. Revalidation also
runs after policy and any approval; a stale failure there may follow approval
events but still precedes `tool_started`. The associated ToolResult has
`is_error=true`, so the model may recover by reading the file and trying again.
No fingerprint is exposed. The event adds no RunRecord end reason and is
additive within JSONL wire schema version 1.

`workspace_mutated` is emitted only after a structured mutation succeeds and
after its `tool_completed` event. Its payload is:

| Field | Meaning |
| --- | --- |
| `tool_name` | `write_file` or `apply_patch` |
| `call_id` | Provider call identity, when supplied |
| `path` | Canonical workspace-relative target |
| `operation` | `created`, `overwritten`, or `patched` |

It contains no content, patch text, fingerprint, or absolute host path. Failed,
blocked, denied, and merely requested mutations do not emit it. The event is
additive within JSONL wire schema version 1.

`progress_snapshot` reports deterministic facts for the current
`Agent.run()`. Its payload is:

| Field | Meaning |
| --- | --- |
| `logical_steps_completed` | Existing Agent steps fully appended to the Run trace |
| `model_attempts` | Physical attempts from authoritative `ExecutionUsage` |
| `tool_calls` | Batch-budget-admitted exposed calls from `ExecutionUsage` |
| `successful_tool_results` / `failed_tool_results` | Runtime-produced ToolResults grouped by `is_error` |
| `unique_tool_actions` | Distinct canonical tool-name + arguments identities dispatched |
| `repeated_tool_actions` | Occurrences after each observed identity's first occurrence |
| `max_identical_tool_action_count` | Largest occurrence count for one identity |
| `model_retries` | Same-context physical retry attempts admitted by all gates |
| `context_recoveries` | Successful smaller-context rebuilds after provider overflow |
| `context_pressure_count` | Logical requests where configured proactive pressure handling was applied |
| `context_window_exceeded_count` | Provider `ContextWindowExceededError` occurrences |
| `terminal` | Whether this is the Run's final progress snapshot |

`step` remains in the top-level event envelope. A non-terminal snapshot is
emitted after each completed tool step. Exactly one terminal snapshot is
emitted immediately before `agent_completed` or `agent_failed` for ordinary
completion/runtime-failure paths. Counters reset at the beginning of every Run,
not every Session.

Tool-action identity ignores `call_id` and hashes canonical JSON of the tool
name and arguments with sorted dictionary keys; list order remains meaningful.
Neither raw arguments nor fingerprints are exposed by this event. If an action
cannot be canonicalized, its repeat classification is skipped without failing
execution or merging unknown actions. These values are factual telemetry, not
a progress percentage, usefulness judgment, loop/stall detector, verification
guess, or stop policy. The event is additive within JSONL wire schema version
1.

`coding_evidence_snapshot` reports coding-specific factual evidence for the
current `Agent.run()`. It is deliberately separate from `ProgressSnapshot`,
which describes general runtime activity. Its payload is:

| Field | Meaning |
| --- | --- |
| `workspace_mutations` | Successful structured `write_file` / `apply_patch` mutations reported by Workspace Discipline |
| `command_executions` | `run_command` operations that reached `tool_started` |
| `command_tool_errors` | Started `run_command` operations whose ToolResult has `is_error=true` |
| `process_starts` / `process_polls` / `process_stops` | Corresponding process operations that reached `tool_started` |
| `process_tool_errors` | Started process operations whose ToolResult has `is_error=true` |
| `executions_since_last_mutation` | `run_command` and `start_process` operations observed after the latest structured mutation |
| `verification_attempts` | Native `run_command` operations explicitly declared `purpose="verification"` that reached `tool_started` |
| `verification_exit_zero` / `verification_exit_nonzero` | Started verification commands that produced the corresponding factual process exit status |
| `verification_tool_errors` | Started verification commands whose ToolResult has `is_error=true` and therefore produced no process exit status |
| `verifications_since_last_mutation` | Verification attempts observed after the latest structured mutation |
| `last_mutation_step` | Latest successful structured-mutation step, or `null` |
| `last_execution_step` | Latest started `run_command` / `start_process` step, or `null` |
| `last_verification_outcome` | Latest started verification's `exit_zero`, `exit_nonzero`, or `tool_error`; `null` while none exists or the latest attempt is pending |
| `last_verification_exit_code` | Latest completed verification process exit code, including negative values, or `null` for none/pending/Tool error |
| `last_verification_step` | Latest started verification step, or `null` |
| `terminal` | Whether this is the Run's final coding-evidence snapshot |

The tracker follows sequential tool-call order, including a mutation followed by
execution in the same logical batch. A later mutation resets
`executions_since_last_mutation`. Polling and stopping a process are recorded in
their own counters but are not new post-mutation executions. Calls rejected by
exposure, budget, argument validation, workspace preconditions, policy, or
approval never reach `tool_started` and do not increment execution counters.
The same actual ordering governs `verifications_since_last_mutation`, including
mutation followed by verification in one model tool batch; a later mutation
resets only the post-mutation count, not Run totals.

A non-terminal snapshot is emitted after each completed tool-call batch, after
the existing `progress_snapshot`. One terminal snapshot is emitted after the
terminal progress snapshot and before `agent_completed` or `agent_failed`; an
interrupted Run emits it immediately before `agent_interrupted`. The payload
contains no command arguments, output, file content, hashes, absolute paths, or
secrets. The event is additive within JSONL wire schema version 1.

M21.4A's verification fields are additive payload fields within that same wire
version. They remain run-scoped Agent API and live JSONL evidence; they are not
persisted in RunRecord v2, BenchmarkResult v1, or durable Session v1.

Verification purpose is explicit model-declared metadata on native
`run_command`, with exact values `general` and `verification`; it is never
inferred from command names, argv, output, cwd, timeout, user text, or Skill
text. A command that returns a non-zero exit code still has a successful
ToolResult, while a Tool exception records `tool_error` without inventing an
exit code. Neither zero nor non-zero exit status proves task correctness.
Background processes have no verification-purpose semantics in M21.4A. The
M21.3 CompletionPolicy originally did not inspect these fields; M21.4B adds the
narrow policy described below without changing the tracker.

### Evidence-aware completion

The coding profile enables one bounded completion recheck. This is a heuristic
runtime control, not proof of task correctness. A final candidate is
reconsidered when any of these rules applies:

- the latest structured mutation was followed by an explicitly
  verification-marked native command whose latest outcome is `exit_nonzero`
  or `tool_error`, with no later mutation;
- execution (`run_command` or `start_process`) was observed with no successful
  structured workspace mutation; or
- a successful structured mutation was observed with no execution after the
  latest mutation.

The verification-failure rule has highest priority. A verification before the
latest mutation is ignored; the later mutation is treated as a reaction even
without another execution. No activity is accepted directly, as are mutation
followed by general execution and mutation followed by exit-zero verification.
Structured verification outcome is inspected, but stdout, stderr, command
arguments, file content, final-answer text, TaskState wording, and task intent
are not. There is no shell-mutation detection, command-name classifier,
rendered-output parsing, or zero-work immediate-final guard. At most one
recheck is requested per Run.

`completion_recheck_requested` contains the stable `reason`, one-based
`recheck_number`, `max_rechecks`, and bounded factual counters:
`workspace_mutations`, `command_executions`, `process_starts`, and
`executions_since_last_mutation`. The rejected final is already in RunTrace but
is absent from Session. For `verification_failed_after_mutation` only, the
event also contains `verification_outcome` and nullable
`verification_exit_code`. Non-terminal progress and coding-evidence snapshots
follow the event.

`completion_recheck_skipped` contains `reason`, `skip_reason`,
`rechecks_used`, `max_rechecks`, and the same bounded counters. Skip reasons are
`context_capacity`, `step_budget`, `model_attempt_budget`, and
`recheck_limit`. Verification-failure skips carry the same optional factual
outcome fields. A skip accepts the original final and proceeds through the
ordinary single terminal completion sequence. Neither event includes commands,
stdout, stderr, paths, or file content.

The next logical request receives one ephemeral system message. Its
`context_built` event reports `completion_recheck_present=true` and a separate
positive `estimated_completion_recheck_tokens`; ordinary requests report
`false` and `0`. The guidance is pinned non-history context and survives
same-request retry and provider-overflow recovery. Its token cost is live
context telemetry only and is not persisted in RunRecord v2. Neither completion
event exposes final response content, commands, output, paths, or file content.
Both events and fields are additive within live-event JSONL schema version 1.

### Bounded stagnation advisory (opt-in)

`pureharness run ... --stagnation-advisory` enables the M24.3B runtime policy.
`runtime_advisory_emitted` is an additive, non-terminal JSONL v1 event at the
first budget-admitted model dispatch that actually includes guidance. It means
guidance was passed to the model interface, not that the provider accepted the
request or the task recovered. The envelope's `step` is the delivery step.
Its payload contains only `kind="stagnation"`, one-based `advisory_index`,
`detected_at_step`, `delivered_at_step`, `window_size`, `repeated_action_count`,
`unchanged_result_repeat_count`, `new_action_count`, `workspace_mutation_delta`,
and `verification_delta`. No arguments, results, paths, hashes, or advisory
text are emitted. Physical retry and context recovery do not emit it again.

Enabled runs also expose optional `context_built` fields
`stagnation_advisory_present` and `estimated_stagnation_advisory_tokens`.
Guidance is non-history context like completion recheck; its separate token
cost participates in ContextLimits and context-recovery request estimates but
is not persisted in RunRecord. Disabled runs emit neither field. Guidance that
cannot fit explicit context capacity is omitted without a delivery event or
intervention count. The existing history budget and atomic-unit rules remain
unchanged.

After delivery the policy remains latched even if the signal clears or new
actions/results appear. Only an increase in structured-mutation or marked
verification-attempt counters after delivery re-arms it, subject to an absolute
two-advisory cap. These counters do not prove correctness or audit shell writes.
Pending guidance, latch state, and intervention count are per-run live state;
they are not restored by observational replay or durable resume. Completion
events and policy remain distinct and unchanged.

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
| `previous_estimated_request_tokens` | Previous history + unchanged pinned non-history context + schemas |
| `recovered_estimated_request_tokens` | Rebuilt history + unchanged pinned non-history context + schemas, when available |

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

`ExecutionUsage`, `ProgressSnapshot`, `CodingEvidenceSnapshot`, and the
run-scoped workspace snapshot and mutation ledger are live diagnostic state
rather than persisted evidence. No progress, coding evidence, or workspace
discipline state is added to RunRecord v2 or
BenchmarkResult. RunRecord v1
retains its original six-value closed end-reason contract.
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
