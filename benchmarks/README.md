# PureHarness benchmark

This directory contains a small, deterministic benchmark for comparing
PureHarness context and tool-exposure configurations. It is not a leaderboard
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
| `full_pureharness` | TokenBudget | deterministic | enabled | deterministic | task-declared static names |

`raw_baseline` is the closest baseline supported by PureHarness; it is still
the PureHarness runtime, not a raw provider request. Static selective exposure
is curated task metadata, not semantic or intelligent tool selection.
`declared_required_tools` is descriptive metadata only and is not a success
oracle or proof that a tool is necessary.

## Tasks

The five curated fixtures cover a short arithmetic fix, multi-file navigation,
large verifier output, a longer two-defect repair, and a tool-exposure-sensitive
edit. Task metadata is static JSON. Verification uses argv lists without a
shell, with a centralized 30-second timeout.

The on-disk trust boundary is explicit:

```text
benchmarks/tasks/<task>/
    task.json       static task/configuration metadata
    workspace/      canonical input copied for every case
    verifier/       trusted oracle, never copied into Agent workspace
```

Each task keeps its trusted verifier outside the canonical workspace fixture.
For every task/config pair, only the fixture is copied to a writable temporary
workspace and coding tools are bound to that copy. The runner snapshots the
trusted verifier before Agent execution, checks that the canonical verifier did
not change, and materializes the snapshot at a separate random path only after
the Agent stops. Verification runs against the copied workspace, then all
temporary files are removed. Editing a workspace-local `verify.py` therefore
cannot forge success.

A verifier non-zero exit is an ordinary task failure. Failure to start the
verifier, verifier timeout, and trusted-verifier mutation are distinct
benchmark infrastructure errors. The trusted verifier path is not sent to the
model. The default local command backend is still not a hostile-code sandbox;
Docker or a stronger future isolation boundary is required if arbitrary Agent
commands must be prevented from exploring other readable host paths.

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
workspaces. They require no API key or network access. Comparable experiments
must supply a model factory that creates a fresh, equivalent model for every
case; result metadata records configuration identity but does not attempt to
enforce provider determinism.

M13 adds an optional controlled real-model runner and repeated-trial evidence
format. See [REAL_MODEL_EXPERIMENT.md](REAL_MODEL_EXPERIMENT.md). It remains
manual, API-backed, potentially costly, and outside deterministic pytest.

From a repository checkout, the installed CLI exposes the same benchmark and
experiment abstractions:

```bash
pureharness benchmark \
  --task simple_fix \
  --config raw_baseline \
  --repetitions 1 \
  --output benchmarks/results/first-run.jsonl
```

The default tasks root is relative to the current working directory, so run
this command from the repository root or pass `--tasks-root`. The harness is a
small controlled study: five curated tasks, approximate provider-neutral token
estimates, no provider usage accounting, no retries, no statistical analysis,
and no claim that success generalizes beyond these fixtures.
