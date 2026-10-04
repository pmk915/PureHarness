# External evaluation evidence

Third-party Harbor / Terminal-Bench results complement the small controlled
PureHarness benchmark. This directory contains reproducible, bounded summaries,
not raw benchmark data, task fixtures, or copies of Harbor jobs. M24.4A defines
the format and offline extractor; M24.4B backfilled the initial four verified
pilot receipts. The final M24 closeout adds a second independent advisory-enabled
MIPS pilot. No benchmark execution is part of the backfill.

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

## Verified receipt index

Each receipt was generated with an explicit task-matched `--oracle-job`,
independently re-extracted to a temporary directory, and compared byte-for-byte.
During M24.4B, SHA-256 inventories of all files in the seven source jobs matched
before and after extraction. The closeout repeated this integrity check for
the new agent job and its explicitly supplied Oracle. Job/trial IDs, dataset,
task ref/checksum, model and full recorded PureHarness revision were checked
against the source artifacts.
Short revisions in filenames and this table are presentation only.

Oracle health below means recorded reward 1 and exceptions 0 for the separate
selected Oracle trial; it does not establish agent success.

<!-- external-evidence-index:start -->
| Task | Treatment / revision | Oracle health | Reward | Runtime end reason | Steps | Receipt |
|---|---|---|---:|---|---:|---|
| sqlite-db-truncate | post-M24.2 / `83abbf4` | 1 / 0 | 1 | completed | 113 | [JSON](receipts/terminal-bench-2.1_sqlite-db-truncate_83abbf4.json) |
| regex-log | post-M24.2 / `83abbf4` | 1 / 0 | 1 | completed | 10 | [JSON](receipts/terminal-bench-2.1_regex-log_83abbf4.json) |
| make-mips-interpreter | baseline / `83abbf4` | 1 / 0 | 0 | max_steps_exceeded | 300 | [JSON](receipts/terminal-bench-2.1_make-mips-interpreter_baseline_83abbf4.json) |
| make-mips-interpreter | bounded advisory / `afca7d2` | 1 / 0 | 0 | max_steps_exceeded | 300 | [JSON](receipts/terminal-bench-2.1_make-mips-interpreter_advisory_afca7d2.json) |
| make-mips-interpreter | bounded advisory, repeat2 / `afca7d2` | 1 / 0 | 0 | max_steps_exceeded | 300 | [JSON](receipts/terminal-bench-2.1_make-mips-interpreter_advisory-repeat2_afca7d2.json) |
<!-- external-evidence-index:end -->

The dataset is `terminal-bench/terminal-bench-2-1`; all indexed recorded model
identities are `deepseek-v4-flash` (provider `deepseek`), with Harbor `0.23.0`.
For re-extraction, use `jobs/<external.job_name>` and explicitly supply
`--oracle-job jobs/<oracle.job>` from each receipt, writing to a new file.
The initial four summaries match the
[progress-gap audit](../../docs/progress_gap_audit.md).
The closeout trial `make-mips-interpreter__CWgdNCX` comes from agent job
`2026-10-04__20-21-36`, paired with Oracle job `2026-10-03__04-13-48`.
It is a second independent advisory-enabled pilot, not a causal comparison:
reward 0, `max_steps_exceeded`, zero structured mutations/verification attempts,
one advisory detected at step 215 and delivered at 216, and a longest active
progress gap of 300. Its recorded maximum/final history estimates are
8000/7969 tokens. Runtime completion and structured anchors remain distinct
from external task success.

## Offline pack validation (M24.4C)

```bash
.venv/bin/python scripts/validate_external_evidence.py
# Optional: validate another checkout or a proposed fixture pack.
.venv/bin/python scripts/validate_external_evidence.py /path/to/external
```

Normal push/PR CI runs the same validator in the existing Python 3.11/3.12
test job before pytest. It checks committed receipt/index consistency and
fails on invalid evidence; it does not rerun Harbor, authenticate external
sources, or judge task correctness. The first command above is the local check.

The default target is this checkout's `benchmarks/external/`, independent of
the current working directory. Validation reads the README and all recursive
lowercase `*.json` receipt candidates in sorted order, including proposed new
files. `.gitkeep` and non-JSON files are ignored; directory symlinks are not
followed and paths escaping the pack are rejected. No Git or `jobs/` is needed.

The reader in `external_evidence.py` checks the existing v1 dataclass shape,
requiring explicit fields (including nullable ones) and rejecting unknown fields,
invalid types, negative counters and non-finite numbers; ambiguous duplicate
JSON field names are also rejected. Pack validation checks
counter/context/advisory relations, separate Oracle identity, duplicate source
trials/run IDs, filename SHA suffixes (a final `_` plus 7–40 hexadecimal digits),
and conflicting shared Oracle/task facts.
Known optional null evidence stays unavailable, not zero; a valid unhealthy
Oracle or reward 0 with a completed runtime is not rejected.

Only the table between the receipt-index markers is machine-validated, for
complete, unique and existing receipt links. Human-written table metric cells
are not parsed or authenticated. Success prints a deterministic receipt count;
failure reports sorted path/invariant diagnostics on stderr with exit code 1.
Nothing is rewritten. A malformed receipt reports its first structural error;
other receipts, semantic relations and index errors are still checked.

The workflow is: saved raw jobs -> `external_evidence_receipt.py` -> reviewed
committed receipts -> `validate_external_evidence.py`. Extraction needs original
artifacts; pack validation does not. Organization/internal consistency checks
are not benchmark-correctness or Agent-quality judgments, source authentication,
or a fresh source-hash comparison. Validation never reruns Harbor or verifiers.

## Pilot limits

These are individual pilot trajectories, not statistical evidence of causal
advisory improvement or a leaderboard. Task revisions/checksums, model, runtime revision, missing
evidence and Oracle health matter when comparing receipts. Hashes establish
artifact identity, not authenticity or benchmark validity. Full history is
read transiently for offline evaluation; receipts contain only bounded summaries.

The backfill contains only allowlisted summaries, not prompts, raw tool output,
environment values, task contents or verifier logs. A runtime max-steps end
reason is not a verifier-failure diagnosis; mutation and stagnation evidence
are not task-success judgments. Do not add large `jobs/` trees or execute
benchmarks as part of receipt maintenance.
