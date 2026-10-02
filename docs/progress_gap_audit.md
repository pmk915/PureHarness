# M24.3C: progress-gap evidence audit

This is a deterministic offline observation, not a detector or task-correctness
decision. No model, tool, backend, or Harbor execution is involved. Steps are
zero-based. External reward is read separately from the existing trial result;
Agent `completed` and a mutation/verification anchor are not task success.

## Semantics and reproduction

`ProgressGapEvaluator` consumes the same `StagnationStep` observations used by
strict stagnation. Positive successful structured-mutation or explicit
verification-attempt deltas anchor a span. Verification outcome does not affect
anchoring. An anchor step resets counters to zero and is excluded entirely from
the following span; no intra-batch tool ordering is inferred. Before any anchor,
age starts at the supplied prefix's first step. An inactive/final-message step
increments logical age but not `active_steps`. Novelty or changed observations
never reset age. An incomplete evidence span is not a confirmed anchor-free span.

Counts compare exact identities against the entire prefix, including anchor
steps. `unique_action_growth` equals newly seen identities since the anchor.
Repeat and unchanged-repeat ratios both use all span tool actions as denominator
(zero actions gives 0.0). Unknown identities are counted separately. Individual
mutation/verification distances are `None` if never observed or their evidence
became unavailable after the last observed anchor. A later anchor of that kind
restores that distance; a later known zero does not.

The following read-only Python example reproduces a per-step timeline for any
one of the trial directories below. It also covers the terminal message's
post-step snapshot without double-counting terminal duplicates.

```python
import json
from dataclasses import fields
from pathlib import Path
from pureharness.coding_evidence import CodingEvidenceSnapshot
from pureharness.evaluation import ProgressGapEvaluator, stagnation_steps_from_run_record
from pureharness.run_record import RunRecord
from pureharness.verification import VerificationOutcome

trial = Path("jobs/2026-10-03__03-40-43/sqlite-db-truncate__Y44bw68")
record = RunRecord.from_dict(json.loads((trial / "agent/pureharness-run-record.json").read_text()))
names = {field.name for field in fields(CodingEvidenceSnapshot)}
snapshots = {}
for line in (trial / "agent/pureharness-events.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event["event"] != "coding_evidence_snapshot":
        continue
    assert event["run_id"] == record.run_id
    values = {key: value for key, value in event["payload"].items() if key in names}
    if values.get("last_verification_outcome") is not None:
        values["last_verification_outcome"] = VerificationOutcome(values["last_verification_outcome"])
    snapshot = CodingEvidenceSnapshot(**values)
    if event["step"] in snapshots:
        assert snapshots[event["step"]] == snapshot
    snapshots[event["step"]] = snapshot
steps = stagnation_steps_from_run_record(record, coding_evidence=snapshots)
timeline = ProgressGapEvaluator().evaluate_steps(steps)
for evidence in timeline:
    print(json.dumps(evidence.to_dict(), sort_keys=True))
```

The evaluator retains O(unique prefix actions) hashes and scalar counters;
streaming does not retain history or old snapshots. Materializing `timeline`
explicitly costs O(steps). The existing record adapter materializes input pairs;
call `iter_evidence()` on an iterable of steps for streaming evaluation.

## Available controls and failures

| Trial under `jobs/` | Revision | Steps / end reason | Reward | Structured mutations / verification attempts |
|---|---|---|---|---|
| `2026-10-03__03-40-43/sqlite-db-truncate__Y44bw68` | `83abbf4` | 113 / completed | 1 | 8 / 31 |
| `2026-10-03__03-53-50/regex-log__wZHN9Lz` | `83abbf4` | 10 / completed | 1 | 2 / 1 |
| `2026-10-03__04-00-26/make-mips-interpreter__WSR63QE` | `83abbf4` | 300 / max_steps_exceeded | 0 | 0 / 0 |
| `2026-10-03__05-20-32/make-mips-interpreter__jSVkhJi` | `afca7d2` | 300 / max_steps_exceeded | 0 | 1 / 0 |

All snapshots cover every trace step and have matching run identity. Repeated
evaluation returns identical evidence and leaves the raw records unchanged.
No additional M24.3B trial with a PureHarness record was available at audit time.

## Anchors and spans over time

SQLite mutation anchors: **28, 40, 59, 61, 75, 103, 106, 109**.
Verification anchor steps: **29, 39, 41, 50, 60, 62, 76-81, 83-91, 93-96,
104, 107, 110, 111**. Steps 39 and 84 each contain two attempts. Outcomes:
22 exit-zero and 9 exit-nonzero, no verification tool errors. A failed attempt
still anchors evidence; it is not a health verdict.

Regex mutation anchors: **3, 6**; verification anchor: **7**, exit zero.
Baseline MIPS: no anchors. M24.3B MIPS: mutation **136**, no verification.
Its advisory delivery at 91 is **not** an anchor.

Every nonempty anchor-free span is listed below. Consecutive anchor steps have
empty spans and contribute no activity. Anchor-step actions are not included.

| Run / span | Active steps | Actions | New identities | Repeats | Unchanged repeats | Changed observations | Tool errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| SQLite 0-27 | 28 | 29 | 29 | 0 | 0 | 0 | 2 |
| SQLite 30-38 | 9 | 9 | 9 | 0 | 0 | 0 | 0 |
| SQLite 42-49 | 8 | 8 | 7 | 1 | 0 | 1 | 0 |
| SQLite 51-58 | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| SQLite 63-74 | 12 | 12 | 10 | 2 | 1 | 1 | 0 |
| SQLite 82 | 1 | 1 | 0 | 1 | 0 | 1 | 0 |
| SQLite 92 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| SQLite 97-102 | 6 | 6 | 5 | 1 | 1 | 0 | 0 |
| SQLite 105 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| SQLite 108 | 1 | 1 | 0 | 1 | 0 | 1 | 0 |
| SQLite 112 (final message) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Regex 0-2 | 3 | 5 | 5 | 0 | 0 | 0 | 0 |
| Regex 4-5 | 2 | 2 | 2 | 0 | 0 | 0 | 0 |
| Regex 8-9 (9 is final) | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| Baseline MIPS 0-299 | 300 | 447 | 155 | 292 | 292 | 0 | 2 |
| M24.3B MIPS 0-135 | 136 | 181 | 75 | 106 | 106 | 0 | 2 |
| M24.3B MIPS 137-299 | 163 | 217 | 119 | 98 | 97 | 1 | 1 |

Longest active spans: SQLite 28, regex 3, baseline MIPS 300, advisory MIPS 163.
The baseline's longest-span repeat/unchanged ratios are both **0.653244**.
The advisory run's late-span ratios are **0.451613 / 0.447005**. SQLite's
longest-span ratios are zero. Missing verification distances in MIPS are `None`
(never observed), not a fabricated 300-step distance from a verification.

Strict stagnation remains unchanged: SQLite/regex have no detections, baseline
MIPS detects at 233-251 and 277-289, advisory MIPS at 90-92 and 113-114.
After mutation 136, the advisory run accumulates 119 new identities and 97
unchanged repeats over 163 active steps. Every late 16-step window at 152-299
has at least 3 new identities, preventing strict detection but not resetting
the progress gap. This separates exact-repeat stagnation, novelty, and absence
of new structured anchors without assigning task meaning.

## Controls, interpretation, and limits

Successful runs can explore without anchors: SQLite succeeds after **28 active
steps of initial exploration**, longer than the strict detector's 16-step
horizon. A hypothetical "active gap >=16" failure flag would incorrectly flag
this control at step 15, even with a missing-verification condition. Long novel
activity is therefore not itself failure. Successful synthetic read-only tasks
can be observationally indistinguishable from unsuccessful exploration.

Within this small sample, MIPS differs by much longer spans, substantial
unchanged repetition, and absent marked verification, in combination. Its late
novelty grows despite external reward zero; this does not prove each new action
was useless. SQLite's nine failed verification attempts also rule out treating
verification failure or attempt count as task correctness. Thresholds that
separate these four artifacts are not calibrated controls across task families.

Only structured mutation is anchored: M24.3B's `dump_syms.py` is an exploratory
helper, and the indirect shell write of `syms.txt` at 152 is outside existing
structured counters. Neither a reset nor a short final gap proves recovery.
One trial per task/config, different task complexities, and exact argument
identity limit generalization; no causal advisory benefit is claimed.

Continuous factual progress-gap evidence is justified alongside strict-repeat
evidence. A second boolean/runtime signal, intervention, or threshold is **not**
justified yet. Collect more long successful/read-only exploration controls and
repeated failure trials offline before considering any separately approved
policy. This milestone leaves Agent, advisory/re-arm, RunRecord, JSONL, Harbor,
benchmark execution, and M22 diagnosis/report behavior unchanged.
