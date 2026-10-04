# CLI demo assets

**Capture pending:** no screenshot, terminal recording, or GIF is committed.
This directory contains documentation only, not empty binary placeholders.

Follow the [reproducible demo flow](../cli_demo.md) to capture actual CLI output.
Intended filenames after review:

- `cli-compact.png`: the real default compact interactive view.
- `cli-verbose.png`: the equivalent task with verbose observability.
- `pureharness-demo.gif`: optional real terminal recording.

Use a 110-column × 32-row terminal where possible. Retain a visible
**offline scripted demo** disclosure; only model responses are scripted, while
tools and evidence generation are real. Do not imply an LLM solved the task or
that the demo is external benchmark evidence.

Before publishing, inspect the entire image/recording for secrets, unrelated
terminal content, and identifying local paths. Record the repository revision,
capture command, locale, dimensions, and tool versions here alongside the asset.
Do not insert synthetic UI text or edit outcomes. Add matching README embeds
only when the genuine files exist. Generated workspaces, session/RunRecord/JSONL
artifacts, and raw terminal recordings belong in the temporary demo directory,
not Git or the external-evidence receipt pack.
