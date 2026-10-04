# Controlled real-model experiment

## Purpose

This optional experiment measures whether PureHarness context and tool-exposure
configurations produce observable differences with the same real model. It does
not establish general model quality or statistical significance.

The frozen comparison dimensions are:

- the same curated task prompt and trusted verifier;
- a fresh copy of the same fixture for every case;
- the same DeepSeek model name;
- the same default ToolPolicy and local ExecutionBackend;
- the same `max_steps`;
- the four named M12 configs: `raw_baseline`, `budget_only`,
  `context_engineered`, and `full_pureharness`.

Only the declared context/projection/TaskState/compaction/exposure configuration
changes between comparable cases.

The internal fixtures are controlled micro-benchmarks, not leaderboard evidence;
see [VALIDITY.md](VALIDITY.md) for their visible contracts and hidden oracles.
Retain the fixture Git revision separately: results from different prompt/oracle
revisions must not be pooled as the same frozen experiment. A single run is a
pilot, and task success alone does not prove the intended mechanism occurred.

## Running

Real experiments are manual and require `DEEPSEEK_API_KEY`. Load it from the
environment or a local uncommitted `.env`, then start with one task, one config,
and one repetition to understand cost:

```bash
.venv/bin/python examples/run_real_benchmark.py \
  --task simple_fix \
  --config raw_baseline \
  --repetitions 1 \
  --output benchmarks/results/first-run.jsonl
```

Repeat `--task` or `--config` to select multiple values. Omitting them runs
all five tasks and all four configs. Total API-backed Agent runs are:

```text
selected tasks × selected configs × repetitions
```

This can incur material API cost. The runner executes cases serially and does
not automatically rerun failed cases; configured repetitions are separate
trials. Within each case, the Agent's bounded model-request retries and context
recovery still apply.

## Evidence and metrics

Each JSONL line is an `ExperimentResult` schema-version-1 object containing:

- model ID and one-based repetition;
- monotonic wall-clock duration for the complete benchmark case;
- the complete `BenchmarkResult`, including oracle outcome and bounded
  verifier evidence;
- the M11 RunRecord with end reason, steps, model/tool calls, tool-result
  errors, approximate context/TaskState/schema token sums, and compaction
  counts.

The terminal summary reports per config run/success counts, success rate,
average steps, model calls, tool calls, tool failures, and duration. The Python
summary also exposes cumulative estimated token and compaction metrics. No
weighted score is calculated.

Provider-exact input/output token usage is not currently available through the
Model protocol and is not added to RunRecord for this experiment.

## Interpretation and limitations

Model outputs are nondeterministic even under frozen harness conditions. Use
multiple repetitions and retain raw JSONL evidence. Small sample differences
must not be described as statistically significant or as proof that one config
is universally best.

Task success comes only from the trusted external verifier. Agent
`end_reason="completed"` remains a separate runtime outcome. Estimated tokens
are provider-neutral approximations, not billing usage. The default local
execution backend is not a security sandbox.

Ordinary pytest never runs this script, reads an API key, or makes network
requests. No real-model experiment result is committed or claimed by M13 unless
the command is deliberately run by a user.
