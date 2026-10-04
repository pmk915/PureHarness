# PureHarness interview guide

Use this as speaking notes, not a new specification or a benchmark report.
The [README architecture](../README.md#architecture) is the visual entry point;
the [architecture document](architecture.md) distinguishes implemented behavior
from targets. Timing below is a rehearsal target, not a measured recording.
The main README remains a project landing page.

## Project introductions

### 30-second version

PureHarness is a small execution-first runtime for long-running, tool-using AI
agents. It preserves raw history, bounds model context, controls tool effects,
and records what happened. Coding is the primary workload, not the kernel
identity. External pilots exposed a context-grouping defect and repeated
exploration without task success. SQLite and regex pilots recorded reward 1;
MIPS remained unsolved. The contribution is transparent execution and measurable
evidence, not state-of-the-art agent performance.

### 2-minute version

PureHarness addresses the engineering around a long-running agent, rather than
claiming to make the model a better reasoner. As tool results accumulate, context
can become unusable, effects need authorization, and a final answer can be
mistaken for task success. The project makes those boundaries explicit in a
small, inspectable runtime. Coding tasks provide its main workload and benchmark.
It is not another chat UI, SaaS, or a LangChain/LangGraph replacement.

One Session retains raw conversation history across runs. Each Agent run owns a
bounded synchronous loop and produces its own RunRecord. Every logical step
derives TaskState and compiles model context without rewriting Session.
ToolSelector chooses visible tool schemas before the model request; returned
calls go through ToolExecutor, workspace checks, ToolPolicy, and approval when
required. Results update history and activity evidence. Explicitly marked
verification commands can support one completion recheck, but the runtime does
not automatically test or decide correctness. Live JSONL and finalized records
support inspection and offline evaluation.

External Terminal-Bench pilots made the tradeoffs concrete. A context-grouping
defect led to a generic atomicity correction; a later unchanged SQLite task
completed with reward 1, as did a regex task. MIPS instead exhausted 300 steps
in the baseline and both advisory-enabled pilots while history stayed bounded.
That motivated exact-repeat and progress-gap evidence, not a success claim.
Task-matched Oracle trials, deterministic receipts, and CI validation keep
external outcomes separate from runtime facts. These five DeepSeek-specific
pilots are a small sample: advisory effectiveness and general performance
improvement are not established.

The engineering stories and receipt links below support both introductions.

### 8–10 minute technical walkthrough

#### 0:00–1:00 — Problem and positioning

Start with the difference between generating a tool call and operating a
long-running execution system. PureHarness aims for transparent, reliable,
measurable execution through explicit boundaries and benchmark-driven feedback,
not guaranteed recovery or correctness. Show the compact screenshot as a view
of effects and the verbose excerpt as a view of surrounding events. Disclose
that the [demo](cli_demo.md) scripts model responses; its tools and persistence
are real, but it is not autonomous LLM performance evidence.

#### 1:00–2:15 — Session, Run, and compiled context

Session is ordered raw messages and tool call/result history, not a file store.
The external CLI lifecycle owns durable persistence and can associate several
RunRecords with one Session. A run is one `Agent.run()` invocation, with fresh
per-run counters and an explicit end reason. Resume restores state and waits
for a new turn; observational replay never re-executes historical work.

At each logical step, TaskState is rebuilt from raw Session and context is
compiled as a derived model-facing view. Tool-result projection precedes
trajectory compaction and final budget selection. Closed independent tool
interactions can be separate units; overlapping calls/results stay indivisible.
Explain Story A below: the hard part was choosing the correct atomic boundary,
not merely shortening text. Structural integrity and preserved raw history do
not guarantee perfect model recall.

#### 2:15–3:45 — Model request and tool boundaries

The synchronous `Agent` owns the loop; `RuntimeController` classifies failures
and bounded lifecycle actions, not task plans. `Model.generate()` receives the
compiled items and selected Tool definitions, returning an assistant Message
or ToolCalls. ToolSelector runs before inference and controls visibility only;
the current selectors expose all tools or configured static names, not inferred
task intent. A non-exposed call is a protocol inconsistency, not a policy denial.

Admitted calls enter ToolExecutor: argument validation, optional workspace
preconditions, ToolPolicy, conditional approval, workspace revalidation, and
execution. Coding edits require observing existing files and checking freshness.
`REQUIRE_APPROVAL` needs an explicit approve decision; missing approval fails
closed. Command tools can delegate to an execution backend; file tools need
not. Neither visibility nor approval is isolation: local subprocess execution
inherits the host environment and is not a secure sandbox.

#### 3:45–5:15 — Results, completion, and recovery

Raw tool pairs go into Session. Tool failures generally become model-visible
error results rather than automatic retries. `ProgressSnapshot` counts activity
and exact repetition; `CodingEvidenceSnapshot` separately records successful
structured mutations, started executions, and explicitly marked verification.
A native `run_command` with `purpose="verification"` supplies exit/outcome
facts; arbitrary shell text is not parsed to infer intent or filesystem edits.

The optional coding completion policy can request one new logical step for
missing mutation/execution evidence or a failed marked verification after the
latest mutation. A later mutation can show reaction without proving a repair.
The recheck is bounded, may be skipped when capacity is unavailable, and cannot
certify correctness. Generic Agents need not configure this policy.

Separate this from model recovery: recoverable malformed output can retry, and
provider context overflow can rebuild a smaller context within the same logical
invocation. `model_call_count == len(model_invocations)` does not count physical
attempts. Step, physical-attempt, and tool-dispatch limits are distinct.
Interrupted tools can have uncertain external effects and are not replayed.
Offline recovery guidance is advice only; the opt-in bounded stagnation advisory
is a separate runtime policy, not offline evaluation taking over the loop.

#### 5:15–8:30 — Evaluation and the three engineering stories

Use the engineering notes below, emphasizing one decision and one limit per
story. A known context-grouping bug was corrected generically; bounded context
did not make MIPS succeed. Strict repetition and progress gaps describe
different facts; neither judges task correctness. Oracle health, external
reward, runtime termination, and advisory delivery remain separate evidence.

RunRecord is finalized evidence for one run; occurrence-time events can be
rendered as JSONL while the run proceeds. They are different schemas, not
interchangeable archives. Coding/progress counters remain live evidence, not
extra RunRecord fields. Post-run metrics, diagnosis, reports, and recovery
guidance consume trajectories without executing work. The completion-derived
metric named `task_success_score` is not an external success oracle.

#### 8:30–9:30 — What remains unsolved

Close with the small external sample, MIPS failures, limited mutation
observability, and unestablished advisory effectiveness. The next evidence
priority is more long successful/read-only controls and repeated failures under
matched conditions before calibrating any new intervention. Provider adaptation
has a narrow interface, but another provider's behavior still needs tests and
fresh evaluation. Do not promise a new model, larger budget, automatic repair,
or task-specific workaround as an already validated solution.

## Engineering stories

### A — Correct atomicity under a bounded context

**Problem.** An external failure exposed a deterministic grouping defect:
the old compiler treated an entire contiguous tool-only history as one atomic
unit. Many independent completed interactions could therefore become one
artificially oversized newest unit. Budget selection and old-trajectory
compaction could not operate at the intended boundaries.

**Decision.** The generic correction, revision `83abbf4` (M24.2), validates the
whole tool span, then splits only when all preceding calls have closed.
Independent pairs are separable; overlapping and incomplete batches remain
indivisible. Projection, compaction, and selection still leave Session raw.
A genuinely oversized newest unit still fails instead of being silently cut.
See the [compiler](../src/pureharness/context.py),
[context semantics](architecture.md#context-compiler), and
[atomicity regressions](../tests/test_context_atomicity.py).

**Evidence and tradeoff.** The unchanged external `sqlite-db-truncate` task
later completed with reward 1: 113 steps, 116 tool calls, 8 structured mutations,
and 31 marked verification attempts. The
[post-fix receipt](../benchmarks/external/receipts/terminal-bench-2.1_sqlite-db-truncate_83abbf4.json)
records the task identity and outcome. The committed pack does not contain a
pre-fix failing SQLite receipt; the failure-to-fix narrative is development
history, not a paired statistical experiment. Regression tests support the
generic structural correction, not general success-rate improvement.
Estimated-token bounds are not exact provider billing or proof that every
useful fact remains visible to the model.

### B — Bounded context does not imply useful progress

**Problem.** All three retained `make-mips-interpreter` pilots reached 300 steps
with `max_steps_exceeded` and reward 0. Their recorded maximum history estimate
was 8000 tokens, with active trajectory compaction. The bounded-context mechanism
remained operational; no context-budget termination explains these outcomes.
That does not prove compaction retained every useful fact.

**Decision.** First add deterministic exact-repeat evidence: the default strict
warning requires 16 consecutive active steps, previously seen actions,
unchanged observations, and known zero mutation/verification deltas. A separate
opt-in policy delivers at most two neutral advisories per run and re-arms only
after structured mutation or marked verification activity. It neither stops
the task nor automatically verifies or repairs anything. See
[stagnation semantics](architecture.md#agent) and
[advisory policy](../src/pureharness/stagnation_advisory.py).

**Observed outcomes.** The baseline had 447 calls and 292 repeats, with no
structured mutation or marked verification. The first advisory pilot delivered
guidance at step 91 after detection at 90; it recorded 399 calls, 204 repeats,
one structured mutation, and no marked verification. The second delivered at
216 after detection at 215, recording 444 calls, 295 repeats, and no structured
mutation or marked verification. Steps are zero-based. Both advisory pilots
still failed externally. These are observed trajectory differences, not causal
benefit or proof that the advisory was ignored.

**Follow-up evidence and tradeoff.** Progress-gap evidence measures activity
since successful structured mutations or explicit verification attempts.
Those are stronger structured anchors, not task-success proofs; even failed
verification anchors a span. New identities and changed observations do not
reset its age. After the first advisory run's mutation at step 136, another
163 active steps contained 119 new identities, 98 repeats, and 97 unchanged
repeats. Novelty continued despite eventual reward 0 and suppressed strict
repeat detection. Conversely, successful SQLite had 28 initial active steps
without an anchor. A simple 16-step no-anchor failure threshold would flag this
successful control. The [progress-gap audit](progress_gap_audit.md) supports
factual evidence, not a new automatic failure detector.

### C — Make evaluation claims independently inspectable

**Problem.** A model's final answer, command exit status, runtime end reason,
and external task reward answer different questions. Mixing them makes success
claims hard to audit and can hide infrastructure problems or failed pilots.

**Decision.** Internal cases use fresh workspace copies and separate trusted
verifiers. External receipts attach explicitly selected, task-matched Oracle
trials, keeping reward and exceptions apart from Agent facts. The offline
extractor allowlists bounded summaries, records source digests, and supports
byte-identical re-extraction from saved artifacts. Failed runs remain in the
same committed pack as successful ones.

**Evidence and limit.** The
[receipt index](../benchmarks/external/README.md#verified-receipt-index)
records five pilots with healthy task-matched Oracle trials. The
[pack validator](../scripts/validate_external_evidence.py) checks schema,
internal relations, shared identities, and index coverage without needing
`jobs/`. Normal push/PR [CI](../.github/workflows/ci.yml) runs it before pytest.
It does not rerun Harbor, compare absent raw sources, authenticate provenance,
or judge correctness. Stored hashes establish artifact identity, not independent
authenticity. Oracle health is a validity prerequisite, not Agent success.

## External evidence to cite

The recorded dataset is `terminal-bench/terminal-bench-2-1`, model
`deepseek-v4-flash` / provider `deepseek`, Harbor `0.23.0`. Short revisions below
are labels; receipts retain full revisions and task refs/checksums. Each pilot
has a separately recorded task-matched Oracle with reward 1 and exceptions 0;
the three MIPS pilots share the same Oracle trial, not three independent gates.

| Pilot / receipt | Reward | Runtime end reason | Steps |
|---|---:|---|---:|
| [SQLite, post-fix `83abbf4`](../benchmarks/external/receipts/terminal-bench-2.1_sqlite-db-truncate_83abbf4.json) | 1 | completed | 113 |
| [Regex, post-fix `83abbf4`](../benchmarks/external/receipts/terminal-bench-2.1_regex-log_83abbf4.json) | 1 | completed | 10 |
| [MIPS, baseline `83abbf4`](../benchmarks/external/receipts/terminal-bench-2.1_make-mips-interpreter_baseline_83abbf4.json) | 0 | max_steps_exceeded | 300 |
| [MIPS, advisory `afca7d2`](../benchmarks/external/receipts/terminal-bench-2.1_make-mips-interpreter_advisory_afca7d2.json) | 0 | max_steps_exceeded | 300 |
| [MIPS, advisory repeat2 `afca7d2`](../benchmarks/external/receipts/terminal-bench-2.1_make-mips-interpreter_advisory-repeat2_afca7d2.json) | 0 | max_steps_exceeded | 300 |

Say: these artifacts demonstrate recorded task-level successes, retained
failures, bounded history, and delivered advisories in specific pilots.
They do not establish a general success rate, model superiority, or advisory
causality. `max_steps_exceeded` describes runtime termination, not a
verifier-failure diagnosis; `NonZeroAgentExitCodeError` is a separate recorded
Harbor exception type. Full extraction/validation limits are in the
[external evidence guide](../benchmarks/external/README.md).

## Likely interview questions

### Why not use LangGraph?

The goal is to own and inspect a small synchronous execution loop and its state,
effect, and evidence contracts, not replace a framework. That is a deliberate
scope choice, not evidence that another framework cannot solve these problems
or that PureHarness is faster or better.

### Why distinguish Session and Run?

Conversation state survives turns; execution limits, counters, end reason, and
RunRecord belong to one run. Separating them prevents later turns from changing
earlier evidence. Durable resume restores history rather than executing it.

### Why is history not the same as model context?

Raw history is source truth. Compiled context is a replaceable, bounded view
that can project outputs and compact old interactions without destroying that
truth. This protects persistence and analysis, not perfect recall; omitted
content may still matter to the model.

### Why not terminate automatically on stagnation?

Exact unchanged repetition can be intentional polling or reinspection.
Successful SQLite also had 28 active no-anchor exploration steps. Neither
strict repetition nor progress-gap age proves failure. The current optional
advisory is bounded guidance; a calibrated stop policy needs broader controls.

### Why does completion not mean task success?

An accepted final Message ends the Agent protocol. Marked verification supplies
factual outcomes, but may be incomplete or unsuitable. Internal success comes
from a separate trusted verifier; external success comes from the recorded
evaluator reward. Even the CLI's `task_success_score` is completion-derived,
not an oracle result.

### How do you prevent benchmark overfitting?

Use generic changes with deterministic regression cases, unchanged external
tasks, fresh internal workspace copies, and verifiers outside the writable
workspace. Retain failures and task/revision/configuration identity. This
reduces leakage and selective reporting; five curated internal fixtures and
three external task identities cannot establish freedom from overfitting.
See the [benchmark boundary](../benchmarks/README.md).

### Why use Oracle health gates?

A task-matched Oracle with reward 1 and no recorded exception supplies a check
that the reference route worked in its recorded run. Missing or
unhealthy gates must stay visible. A healthy gate neither proves the Agent
succeeded nor establishes complete benchmark correctness or source authenticity.

### What did Terminal-Bench actually prove?

In these five recorded pilots, SQLite and regex obtained external reward 1;
all three MIPS trajectories exhausted 300 steps with reward 0. The artifacts
exercised bounded context and recorded advisory delivery. They do not prove
general reliability, a population success rate, or causal improvement.

### Why did make-mips-interpreter fail?

Confirmed: the runtime exhausted its step limit while exact repetition and
long anchor-free activity persisted; marked verification was absent and
external reward was 0. Bounded context was still operating. A planning or
phase-transition weakness is plausible, but these summaries do not isolate
model reasoning, derived-state limitations, or tool ergonomics as one cause.
Zero structured mutations is not proof of no indirect shell writes.

### What would you improve next?

First collect more matched successful/read-only long exploration controls and
repeated failures, then evaluate any proposed progress-aware policy offline
for false positives. Broader provider tests and explicit mutation-observability
design are candidates, not delivered features. Do not just increase steps,
change the history budget, or special-case MIPS to make one pilot pass.

### What was the hardest bug?

A concrete example is context atomicity: a valid safety rule became too coarse
when an entire tool-only span was treated as indivisible. The fix split only at
closed-call boundaries while preserving overlapping batches, matching rules,
raw history, and the hard budget. The atomicity regressions make that invariant
testable; the later SQLite success is supporting pilot evidence, not a causal
success-rate estimate.

### How would this scale to another provider/model?

Implement the existing provider-neutral `Model.generate()` contract and map
responses/errors into core Message, ToolCall, and error types. Test tool schemas,
argument decoding, overflow normalization, and request estimates. Core Agent
and execution backends need not depend on that provider. A narrow interface
does not validate another model's capability, token accounting, or performance;
the retained external pilots are DeepSeek-specific. See the
[Model boundary](architecture.md#model).

## Limitations to volunteer

- Five external pilots cover only three task identities and one recorded
  DeepSeek model. They are not a leaderboard, statistical study, or evidence
  of general success-rate improvement.
- MIPS remained unsolved, including the second independent advisory-enabled
  pilot. Delivery is established; advisory effectiveness is not.
- Structured mutations are not a full filesystem audit. Indirect command writes
  can be absent from counters; absent marked verification does not classify
  every shell command as unverified by intent.
- A mutation or verification attempt is an evidence anchor, not proof of useful
  task progress. A failed verification still anchors a span.
- The offline scripted demo demonstrates actual tools/rendering/persistence,
  not autonomous LLM performance. Capture provenance is not independently
  authenticated; see the [asset notes](assets/README.md).
- Token counts are estimates, not provider billing or exact total request size.
  Derived context may omit useful information even while its budget is healthy.
- Rule-based diagnosis is intentionally narrow; recovery guidance does not
  execute changes, and there is no automatic general self-repair.
- The loop is synchronous; concurrency and production isolation are not claimed.
  Local execution is not a sandbox. See the [security model](security.md).

## Final README presentation review

This is a static documentation review, not a completed browser acceptance test.
The existing [render checklist](assets/README.md#readme-render-review) remains
applicable; no screenshots, display widths, or architecture diagram changed.

- Desktop hierarchy remains positioning/disclosure, compact screenshot,
  capabilities and quick start, CLI, architecture, then evaluation evidence.
  The interview-guide link is a small follow-up, not a first-screen script.
- Compact remains capped at 800 px, verbose at 760 px with no fixed height.
  The tall verbose excerpt stays inside `details`/`summary`; both originals
  remain linked. Their offline-scripted disclosure stays visible.
- Both READMEs share one top-down Mermaid structure with short labels. This
  milestone adds no diagram and does not change the model/tool order.
- README tables remain two and five columns; this guide's evidence table has
  four. Long identifiers and command blocks may need horizontal scrolling.
- English/Chinese structure, code blocks, screenshots, and evaluation claims
  remain aligned. This deeper guide is English, like the architecture document,
  with an explicitly labeled link in the Chinese README.

Manual GitHub/browser checks still needed: desktop and narrow mobile layout,
image scaling and terminal-text legibility, verbose expansion, Mermaid
rendering/loop routing, table/code scrolling, and navigation anchors in both
languages. Local Markdown parsing cannot reproduce GitHub styling or execute
Mermaid. No responsive web code or new rendering dependency is introduced.
