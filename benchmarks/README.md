# MiniHarness M12 benchmark

This directory contains a small, deterministic benchmark for comparing
MiniHarness context and tool-exposure configurations. It is not a leaderboard
or a general benchmark platform.

## What is measured

Each case is one task and one named configuration in a fresh copied workspace.
The Agent produces the canonical per-run `RunRecord`. After the Agent stops,
the harness independently executes the task's argv verifier. A zero verifier
exit code means `task_success=True`; an Agent
`end_reason="completed"` does not by itself mean the task succeeded.

The four standard configurations use an explicit 8,000 estimated-history-token
budget where applicable:

| Config | History | ToolResult projection | TaskState injection | Trajectory compaction | Tool exposure |
| --- | --- | --- | --- | --- | --- |
| `raw_baseline` | FullHistory | identity | disabled | identity | all |
| `budget_only` | TokenBudget | identity | disabled | identity | all |
| `context_engineered` | TokenBudget | deterministic | enabled | deterministic | all |
| `full_miniharness` | TokenBudget | deterministic | enabled | deterministic | task-declared static names |

`raw_baseline` is the closest baseline supported by MiniHarness; it is still
the MiniHarness runtime, not a raw provider request. Static selective exposure
is curated task metadata, not semantic or intelligent tool selection.
`declared_required_tools` is descriptive metadata only and is not a success
oracle or proof that a tool is necessary.

## Tasks

The five curated fixtures cover a short arithmetic fix, multi-file navigation,
large verifier output, a longer two-defect repair, and a tool-exposure-sensitive
edit. Task metadata is static JSON. Verification uses argv lists without a
shell, with a centralized 30-second timeout.

The canonical fixture directory is never passed to coding tools. For every
task/config pair it is copied to a temporary workspace, tools are bound to that
copy, verification runs there, and the copy is removed afterward.

## Results

`BenchmarkResult` schema version 1 embeds the M11 RunRecord and bounded
verification stdout/stderr previews. JSONL output stores one complete result per
line under a caller-selected path; generated `benchmarks/results/*.jsonl`
files are ignored by Git.

Per-config summaries retain raw success counts/rates, end reasons, model/tool
calls, steps, estimated history/TaskState/tool-schema token sums, and compaction
counts. Estimated tokens are cumulative provider-neutral model-facing
estimates. They are neither unique-context size nor provider billing tokens.
No weighted composite score is calculated.

Unit tests use only deterministic scripted models and local temporary
workspaces. They require no API key or network access. A real-provider
entrypoint is intentionally deferred. Comparable experiments must supply a
model factory that creates a fresh, equivalent model for every case; M12 records
configuration identity but does not attempt to enforce provider determinism.
