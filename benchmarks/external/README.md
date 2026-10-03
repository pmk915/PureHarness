# External evaluation evidence

Third-party Harbor / Terminal-Bench results complement the small controlled
PureHarness benchmark. This directory contains reproducible, bounded summaries,
not raw benchmark data, task fixtures, or copies of Harbor jobs. M24.4A defines
the format and validates temporary receipts; reviewed receipt backfill is a
separate M24.4B step. `receipts/` is intentionally empty for this milestone.

## Offline extraction

From an installed development checkout, use the separate offline script:

```bash
.venv/bin/python scripts/external_evidence_receipt.py \
  jobs/2026-10-03__03-40-43 \
  --oracle-job jobs/2026-10-03__02-43-37 \
  --output /tmp/sqlite-external-evidence.json
```

Omit `--output` for JSON-only stdout. Output files must be new files with an
existing parent directory; the command refuses overwrite and any destination
inside either source job, including when the input is a trial directory.
Diagnostics use stderr and failures return exit code 1.

A job must contain exactly one relevant PureHarness trial. Supply `--trial
<directory-name>` or the trial directory itself when ambiguous. Oracle selection
likewise requires exactly one Oracle trial or `--oracle-trial <directory-name>`.
Other agent types are not used as PureHarness evidence. The extractor reads
structured result/lock/RunRecord/JSONL files only. It never runs Harbor, Git,
models, tools, shell commands, verifiers, or anything stored in a trajectory.
Unlike the older `scripts/benchmark_receipt.py`, it does not query the current
repository HEAD or installed Harbor version. Historical facts come from saved
`agent_info.version` and `lock.harbor.version`, not the extraction environment.

Required artifacts are the selected trial's `result.json`, exactly one
`pureharness-run-record.json`, and exactly one `pureharness-events.jsonl` under
its `agent/`, `artifacts/`, or `logs/` directories. Job input also requires the
job `result.json`; direct trial input may have no parent job metadata. Missing,
ambiguous, malformed, cross-run, or inconsistent required artifacts fail clearly.
JSONL requires its single terminal Agent event to match RunRecord. Symlink
escapes are rejected. Unknown additive JSONL v1 event types are ignored.

## Receipt schema v1

`receipt_type="pureharness.external_evidence"`, `schema_version=1` identifies
this artifact independently of old reproducibility receipts, RunRecord v1/v2,
and JSONL v1. The frozen model is in
`src/pureharness/evaluation/external_evidence.py`. It supports deterministic
sorted, indented JSON with a trailing newline and no extraction-time timestamp.

| Section | Source and meaning |
|---|---|
| `external` | Dataset, task/ref/checksum, recorded Harbor version, model/provider, full recorded PureHarness revision, reward, exception **type only**, job/trial identity |
| `runtime` | RunRecord run ID, end reason, protocol completion, steps, logical model calls, tool calls/executions/errors, trajectory/ToolResult compaction counts |
| `progress` | Last recorded progress snapshot's unique, repeated, and maximum-identical action counts |
| `coding` | Last recorded coding snapshot's structured mutations and verification attempt/zero/nonzero/tool-error counts |
| `context` | Recorded `context_built` count, maximum/final estimated history tokens, consistent reported history budget, compacted-context count |
| `stagnation_advisory` | Recorded delivered stagnation advisory count and first detected/delivered steps, not hypothetical interventions |
| `offline_evaluation` | Existing strict stagnation intervals with first/last window metrics; progress-gap anchors, longest-active-span evidence and final evidence |
| `oracle` | Optional separate selected Oracle trial's reward, exception occurrence count/type and task/job/trial identity |
| `unavailable_evidence` | Missing optional event evidence and incomplete post-step coding coverage |
| `sources` | SHA-256 digests of consumed artifacts under role labels; no raw artifact contents |

Raw tool arguments/results, prompts, file contents, environment, API keys,
exception messages/tracebacks and arbitrary config/payload fields are never
copied. This is a field allowlist, not a general secret-redaction service;
recorded identity metadata must itself be suitable for publication.

Absent optional data remains null, not invented zero. A zero advisory count is
the count of recorded delivery events, not proof the advisory option was off.
Context counts refer to recorded occurrences, including any recompilations,
and history estimates do not include skills, guidance, tool schemas or provider
billing. `configured_history_token_budget` is populated only when all context
events report the same non-null budget; varying effective budgets produce null.

Offline summaries reuse `stagnation_steps_from_run_record`, `StagnationEvaluator`
and `ProgressGapEvaluator` unchanged. Both require explicit same-run post-step
mutation/verification counters for every completed trace step; incomplete
coverage yields null offline evidence and an availability note, not a claim
of no stagnation. Terminal duplicate snapshots must agree. Computation does
not execute observations or call M22 scoring/diagnosis. No universal weighted
Agent score is introduced.

## Semantics and Oracle discipline

- **Reward** is the external evaluator's recorded numeric outcome, or null if
  unavailable. For these Terminal-Bench pilots, reward 1 means external success.
  The format does not impose that interpretation on every future dataset.
- **Harbor exception** describes infrastructure/adapter outcome independently
  of reward; only the recorded exception type is retained.
- **PureHarness end reason** describes runtime termination.
- **Protocol completion** means only `end_reason="completed"`. It never creates
  reward 1 or an inferred `task_success` field. Reward 0 does not invent a
  runtime exception/end reason. For example, a normal completed run may fail
  the external oracle; a max-steps run may have `NonZeroAgentExitCodeError`.
- **Stagnation/progress gaps/mutations/verification attempts** are observations,
  not judgments of task success, failed planning, health, or recovery.

Attach an Oracle health gate only via an explicit path. No unrelated jobs are
searched automatically. Task name and all mutually available dataset/ref/checksum
fields must match. Missing identity metadata remains null and cannot establish
full revision equivalence. Oracle `exceptions` is the selected trial's 0/1
exception occurrence count, not an aggregate across unrelated job trials.
Oracle reward, exceptions and identity stay separate from agent runtime metrics.
Record unhealthy gates faithfully; consumers must review them before interpreting
agent results. A healthy Oracle gate is a prerequisite for benchmark validity,
not evidence that the agent succeeded.

## Pilot validation and limits

M24.4A extracted the four existing pilots offline into temporary files:

| Trial | Reward | Runtime end / steps | Tool calls / executions | Mutations / verification | Advisory | Longest active gap |
|---|---:|---|---|---|---|---:|
| SQLite `Y44bw68` | 1 | completed / 113 | 116 / 116 | 8 / 31 | 0 | 28 |
| Regex `wZHN9Lz` | 1 | completed / 10 | 11 / 11 | 2 / 1 | 0 | 3 |
| MIPS baseline `WSR63QE` | 0 | max_steps_exceeded / 300 | 447 / 447 | 0 / 0 | 0 | 300 |
| MIPS advisory `jSVkhJi` | 0 | max_steps_exceeded / 300 | 399 / 398 | 1 / 0 | 1, detected 90 / delivered 91 | 163 |

All three separately supplied Oracle gates have reward 1 and exceptions 0.
The summaries match the [progress-gap audit](../../docs/progress_gap_audit.md).
These are n=1 pilots, not statistical evidence of causal advisory improvement
or a leaderboard. Task revisions/checksums, model, runtime revision, missing
evidence and Oracle health matter when comparing receipts. Hashes establish
artifact identity, not authenticity or benchmark validity. Full history is
read transiently for offline evaluation; receipts contain only bounded summaries.

After review, M24.4B can backfill selected receipts with task-matched Oracle
references, compare byte-identical re-extraction and source hashes, and publish
only allowlisted summaries. Do not add large `jobs/` trees or execute benchmarks
as part of that backfill.
