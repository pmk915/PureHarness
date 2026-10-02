# Internal fixture validity

These five project-authored fixtures are controlled micro-benchmarks, not public
leaderboard tasks or evidence of general coding-agent quality. Requirements must
be derivable from the prompt and Agent-visible workspace. The separate trusted
verifier checks behavior, not a preferred source-code implementation.

| Task | Intended construct | Agent-visible requirements | Hidden oracle | Expected mechanism/evidence |
| --- | --- | --- | --- | --- |
| `simple_fix` | Basic coding sanity/smoke | Arithmetic-fix prompt and `calculator.add` | Addition on representative inputs | Inspect and repair an obvious defect; no long-horizon claim |
| `exposure_sensitive` | Static selective exposure without losing completion ability | `SPEC.md`: trim surrounding whitespace and uppercase the heading | Multiple unseen headings and whitespace cases | `StaticNames` exposes the unchanged task-declared tool set; lower estimated tool-schema cost versus all-tools exposure |
| `multi_file` | Related-code/configuration inspection | `SPEC.md`, `pricing.py`, `settings.py`: subtotal plus configured tax surcharge | Multiple price lists and configured rates | Navigate implementation and configuration; do not hard-code the rate or a total |
| `large_output` | Large ToolResult handling | `SPEC.md` severity thresholds and `diagnose.py` report | Independent classification checks, including threshold boundaries | Execute the 400-row diagnostic; positive mechanism evidence is ToolResult projection/compaction observed in RunRecord, separately from task success |
| `long_horizon` | Iterative repair across two defects | `SPEC.md`, `pipeline.py`, runnable `check.py` | Parsing, direct averaging and composition on unseen signed, whitespace and varied-length inputs | Local parsing failure → repair → mean failure → repair → local pass; hidden verifier checks generalization |

## Visible feedback versus trusted verification

Only `workspace/` is copied for each task/configuration case. Prompts reference
visible specifications and local checks, never a runnable hidden verifier. The
runner retains its existing trusted-verifier snapshot/integrity boundary and
runs that verifier outside the writable workspace after Agent execution.
No trusted verifier implementation, solution file, or hidden test vectors are
added to a workspace. Public thresholds and arithmetic/parsing contracts are
requirements, not leaked hidden assertions.

`diagnose.py` intentionally prints all 400 sensor rows both before and after
repair. It is an informational report: its own exit code is not a correctness
oracle. `check.py` supplies fail-fast iterative feedback: its parsing check
precedes its mean check. Passing the visible example does not replace the
hidden generalization checks. Empty/malformed pipeline inputs are outside the
documented scored domain.

## Controls and interpretation

- Canonical fixtures remain deterministically broken. Tests run negative
  controls and known behavioral repairs in fresh temporary copies; they do not
  repair canonical inputs or put solutions in Agent-visible fixtures.
- `completed` is a protocol outcome, not task success. Only the independent
  trusted verifier exit code determines `task_success`.
- Exposure is static curated metadata, not intelligent selection or ToolPolicy.
  Compare `context_engineered` with `full_pureharness` to isolate exposure under
  otherwise equal standard settings. Other standard comparisons combine
  multiple mechanisms and are not single-factor attribution.
- A specification cannot force a particular navigation or repair trajectory.
  An Agent may fix both pipeline defects before running a check, or read the
  small related files in one step. Inspect trajectories to confirm the intended
  workflow actually occurred. `long_horizon` is a small two-defect feedback
  proxy, not a demonstration of large-scale long-horizon reliability.
- A report-producing script must actually be run to exercise large-output
  projection. Correctness alone does not prove that mechanism was triggered.
  These small fixtures may have ceiling effects and do not guarantee a score
  gap between configurations.
- One real-model run is a pilot, not statistical evidence. Use repeated trials
  under frozen fixture/configuration revisions, and retain raw evidence.
  Results from earlier prompts/oracles are not directly comparable to hardened
  fixtures; result schemas are unchanged and do not identify fixture revisions,
  so retain the Git revision separately.
- External benchmarks are required for external validity. No paid model run is
  part of the deterministic validity tests.
- Verifier separation is not hostile-code isolation. The local backend retains
  host privileges; this pass does not prevent arbitrary code from inspecting
  readable host paths or tampering with imported Python behavior.
