# CLI demo assets

Two user-supplied, manually captured screenshots of actual offline scripted
CLI terminal sessions are available. This integration preserves their original
bytes: neither image was regenerated, cropped, retouched, or given synthetic
terminal contents.

| Asset | Status | Native dimensions | Role |
|---|---|---|---|
| [cli-compact.png](cli-compact.png) | Captured / real terminal screenshot | 1813 × 899 px | Complete concise read → patch → verify → completed flow |
| [cli-verbose.png](cli-verbose.png) | Captured / real terminal screenshot | 1195 × 1527 px | Early context/model/policy/tool cycles, not final completion |
| `pureharness-demo.gif` | Optional / not captured | — | No file or placeholder link |

The PNGs show English output and the `offline-scripted-demo` model identity.
Their contents match the [reproducible demo flow](../cli_demo.md). Exact capture
commands, capture revision/date, terminal dimensions in columns/rows, and tool
versions were not supplied; pixel dimensions are read from the PNG headers.
The screenshots' visible contents and the user's account of manual capture are
the available provenance, not an independently authenticated recording.

## Privacy and interpretation review

Visual inspection found no visible API keys, credentials, `.env` contents,
unrelated command history, or misleading benchmark claims. The shown `/tmp`
workspace/record paths and random session/run IDs are demo metadata. Compact
also shows `/home/pmk/projects/pureharness/.venv/bin/python`: this discloses a
local username/project path, not a secret, and is acceptable for this public
project under the requested presentation review. It has not been removed.

Compact shows the final answer's explicit benchmark disclaimer and protocol
completion, not externally verified task success. Its trailing
`Interrupted. Use /exit to leave.` cancels prompt input after the run completed;
it does not mean the completed run was interrupted. Verbose is a partial
observability view and must not be captioned as a completed trajectory.

The landing pages display compact at 900 px wide (about 446 px tall) and verbose
at 850 px wide (about 1086 px tall), both below native width. The tall verbose
image is in a collapsed section so it does not dominate the README. Links open
the originals; HTML display sizing does not rewrite image pixels. Two static
PNGs are sufficient for current presentation; a GIF is not required.

## Future capture and publication

Use a 110-column × 32-row terminal where possible. Retain a visible
**offline scripted demo** disclosure; only model responses are scripted, while
tools and evidence generation are real. Do not imply an LLM solved the task or
that the demo is external benchmark evidence.

Before publishing, inspect the entire image/recording for secrets, unrelated
terminal content, and identifying local paths. Record the repository revision,
capture command, locale, dimensions, and tool versions here alongside the asset.
Cropping for framing is acceptable for future captures, but must not conceal the
scripted-model limitation or change the meaning of the flow. Terminal output and
outcomes must never be fabricated or modified. Review secrets before every
publication; the current review does not authenticate raw source artifacts or
guarantee the absence of non-visible metadata.
Add matching README embeds only when genuine files exist.
Generated workspaces, session/RunRecord/JSONL
artifacts, and raw terminal recordings belong in the temporary demo directory,
not Git or the external-evidence receipt pack.
