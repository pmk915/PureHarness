---
name: coding-task
version: 1
description: Complete coding changes through focused implementation and verification.
---

# Coding task procedure

For tasks that require changing code or workspace files:

1. Establish the current behavior, failure, or concrete requirement when practical. Use repository evidence rather than assumptions.

2. Inspect the smallest relevant set of files, symbols, and prior results needed to choose a change. Avoid open-ended investigation once the cause and edit are sufficiently clear.

3. Make a focused change that addresses the observed requirement. Preserve unrelated behavior and existing workspace changes.

4. After the latest change, execute a relevant concrete check appropriate to the repository and task. This may be a build, test, task-level run, smoke check, or another deterministic validation.

5. Observe the actual result, including failures, exit status, useful output, and expected artifacts. Tool execution alone does not establish correctness.

6. If the check fails, diagnose that evidence, revise the implementation, and run an appropriate check again. Base the next action on the latest workspace state.

7. Report completion only with evidence about the latest change and validation result. If verification is unavailable or impractical, state that limitation instead of inventing success.

For analysis or read-only tasks, inspect and report without modifying files merely to satisfy this procedure.
