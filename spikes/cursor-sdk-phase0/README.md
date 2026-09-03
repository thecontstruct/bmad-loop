# Cursor SDK — Phase-0 spike (throwaway)

Purpose: settle the go/no-go gate from the research doc
(`_bmad-output/planning-artifacts/research/technical-cursor-sdk-bmad-loop-provider-integration-research-2026-07-07.md`)
**before** any adapter code is written.

## The gate

> Does a **headless Cursor local agent** run unattended and write a
> bmad-loop-style `result.json` to an exact path on the local working tree?

Everything in the SDK-local design depends on "yes".

## Files

- `cursor-sidecar.mjs` — the process a future Python `CursorSdkAdapter` would spawn.
  Drives a `@cursor/sdk` **local** agent, re-emits `run.stream()` as NDJSON on
  stdout, then a `__sidecar_result__` sentinel line with terminal status + token usage.
- `gate-check.mjs` — spawns the sidecar (as the adapter would), tells the agent to
  write `tasks/spike-task/result.json`, and verifies the file + schema. Writes
  `gate-result.json`.

## Run

```bash
cd spikes/cursor-sdk-phase0
export CURSOR_API_KEY=...        # required
npm install                      # installs @cursor/sdk (Node >= 22.13)

# A: plumbing only (auth + stream + usage), no file-write contract
npm run sidecar -- --cwd "$PWD" --prompt "List the files here and summarize the README in one line."

# B: the actual gate (GO/NO-GO)
npm run gate                     # add --keep to inspect .gate-tmp, --model <id> to override
```

Exit 0 = GO. This directory is disposable; delete it once the result is recorded.
