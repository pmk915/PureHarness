# MiniHarness security model

MiniHarness v0.1 provides explicit policy and execution boundaries, but it is
not a hardened production sandbox or a hostile multi-tenant security boundary.

## Policy, approval, and isolation are separate

`ToolPolicy` makes an authorization decision for a model-requested capability.
Every production Agent tool call passes through `ToolExecutor`, which consults
the policy before invoking the tool. The default policy allows read, write, and
execute capabilities and denies the reserved destructive risk level.

This is capability-level policy, not an argument-level security proof. An
allowed command tool can still receive dangerous arguments. Tool exposure is
also not authorization: `ToolSelector` controls which definitions a model sees,
while `ToolPolicy` remains the decision boundary for exposed calls.

For `REQUIRE_APPROVAL`, an `ApprovalHandler` obtains a one-time `APPROVE` or
`DENY` decision. No handler means deny. The terminal adapter approves only an
explicit `y` or `yes`; empty, invalid, EOF, and Ctrl+C input deny. Policy denial
does not invoke the handler, and approval never bypasses ToolPolicy or
ToolExecutor.

`ExecutionBackend` controls where and how an authorized command runs:

- `LocalExecutionBackend` launches a host subprocess, inherits the host
  environment, and provides no isolation. Use it only for trusted local work.
- `DockerExecutionBackend` applies a constrained container configuration and
  does not expose network or secrets by default. It reduces host exposure, but
  Docker shares the host kernel and its daemon is privileged infrastructure.

Neither backend turns arbitrary Agent-generated commands into proven-safe
operations. Human approval does not make arbitrary code safe, sandboxed,
reversible, or idempotent. Approval is an authorization boundary, not execution
isolation.

Approval decisions are allow-once only. They are not remembered as trust rules.
Durable resume never reopens historical approval prompts or retries their tool
calls. A crash after approval but during execution still has unknown side-effect
state and must not cause automatic retry.

## Secrets

The DeepSeek adapter reads `DEEPSEEK_API_KEY` in the client process. Secrets
must not be copied into Agent workspaces or untrusted execution environments.
The local backend inherits the host environment, so it should not run hostile
commands in a process containing sensitive environment variables.

The Docker backend's default configuration does not forward host environment
variables, credentials, network access, or arbitrary host paths. Changing
those controls changes the threat model and must be reviewed explicitly.

## Benchmark trust boundary

Benchmark task fixtures have two different trust levels:

```text
canonical task directory
    |-- workspace/   copied to a fresh writable Agent workspace
    `-- verifier/    trusted, never copied into that workspace
```

Before Agent execution, the runner snapshots the trusted verifier. After the
Agent stops, it checks that the canonical verifier is unchanged and writes the
snapshot to a separate random temporary path for verification. A zero verifier
exit code is the only source of `task_success=True`.

This prevents workspace-local test edits from replacing the oracle and detects
canonical verifier mutation. It does not make the local backend safe against a
hostile process that searches readable host paths, exfiltrates data, or attacks
the operating system. Use stronger external isolation when the Agent or task is
untrusted.

## Current limitations

- ToolPolicy does not parse or classify command arguments.
- Local execution is not sandboxed and inherits the process environment.
- Docker is a practical isolation adapter, not a perfect security boundary.
- Approval supports only allow-once or deny; there are no remembered trust
  rules, remote approval, durable pending queue, RBAC, or multi-tenant
  isolation.
- RunRecord and Session persistence are evidence/state mechanisms, not tamper-
  evident security logs.
