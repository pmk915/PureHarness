# MiniHarness contributor guide

## Project identity

MiniHarness is a small, inspectable, pluggable runtime for reliable long-horizon,
tool-using agents. Its priorities are transparent execution, recoverable state,
observable behavior, and controlled tool use. The coding agent is the primary
workload and benchmark; it is not the runtime kernel.

Do not grow MiniHarness into a general-purpose agent framework, a coding-agent
product clone, a multi-agent framework, or an unrelated collection of tools.

## Architecture invariants

1. The runtime kernel must not depend on concrete model providers such as
   DeepSeek.
2. Coding-specific tools remain outside the kernel.
3. Context strategies remain replaceable without changing the Agent loop.
4. Session is the source of truth; summaries, task state, and model context are
   derived views and must not silently replace durable history.
5. Tool definition and tool execution should evolve toward separate concerns;
   do not prematurely redesign the current `Tool` API.
6. Execution backends must not depend on model, context, or UI implementations.
7. UIs and renderers consume structured events and must not control execution.
   Do not add progress `print()` calls to the runtime kernel.
8. Observer or listener failure must not fail an Agent run or block other
   listeners.
9. Secrets must not enter untrusted execution environments by default. Local
   subprocess execution is not a secure sandbox.
10. Concrete plugins and adapters must not directly depend on other concrete
    plugins or adapters; communicate through core types and narrow interfaces.
11. Preserve existing tests and behavior unless an approved milestone explicitly
    changes semantics.
12. Prefer small, explicit Python abstractions over managers, factories,
    dependency-injection systems, metaprogramming, or plugin frameworks.
13. All Agent runtime tool execution must pass through `ToolExecutor` and
    `ToolPolicy`; do not invoke `Tool.function` directly from the Agent.
14. Context compilation must not mutate Session, orphan ToolCall/ToolResult
    groups, or silently truncate an indivisible semantic unit to fit a budget.
15. Model-facing ToolResult compaction is derived context state; preserve raw
    results in Session and estimate the projected representation sent to the
    model.
16. TaskState is derived from raw Session, is rebuildable, and must not become a
    second persistence source of truth.
17. Do not infer structured state from arbitrary shell text or natural-language
    semantics without an explicit architecture change.
18. Deterministic trajectory compaction is an ephemeral model-facing view;
    never write compacted blocks into Session or persistence.
19. M9 compaction may replace only complete old tool-execution units. Preserve
    User and Assistant messages and the token-selected recent window raw.
20. Apply ToolResult projection before trajectory compaction, estimate the
    actual representation at each boundary, and keep TokenBudget as the final
    hard constraint.
21. ToolSelector controls per-inference model visibility only; ToolPolicy
    remains the authorization boundary for exposed calls.
22. All production model-facing Tool definitions must pass through
    ToolSelector while ToolRegistry remains the complete capability source.
23. Reject calls to non-exposed tools as protocol inconsistency before
    execution; do not treat exposure as authorization or policy.
24. Each current, known Agent.run() termination path produces one RunRecord;
    the record describes that run and does not replace or duplicate Session.
25. RunRecord metrics are observational facts and must not influence context,
    selection, policy, execution, or task-success decisions.
26. Observational replay must never invoke models, tools, policies, execution
    backends, or other side effects.

## Development workflow

1. Inspect the repository, relevant implementation, tests, examples, and Git
   status before editing.
2. State which architectural boundary the change modifies.
3. Add or update tests for the intended behavior.
4. Make the smallest coherent change and preserve unrelated working-tree edits.
5. Run the relevant tests.
6. Run the full suite and `git diff --check`.
7. Summarize behavior changes, architecture impact, validation, and remaining
   risks.

## Verified commands

The project requires Python 3.11 or newer. It has no committed lock file or test
extra, so install the package and pytest explicitly:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e . pytest
```

Install the optional Rich terminal renderer when working on CLI presentation:

```bash
.venv/bin/python -m pip install -e '.[cli]'
```

Run the configured full test suite from the repository root:

```bash
.venv/bin/python -m pytest
```

The coding demo requires `DEEPSEEK_API_KEY`, calls the external DeepSeek API,
and operates on a temporary workspace:

```bash
.venv/bin/python examples/coding_agent_demo.py --terminal --locale zh-CN
```
