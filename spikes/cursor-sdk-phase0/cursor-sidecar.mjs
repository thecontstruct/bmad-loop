// Phase-0 spike sidecar: the shape a bmad-loop CursorSdkAdapter would spawn.
//
// Drives a Cursor SDK *local* agent against a working tree, re-emitting the
// run's stream as newline-delimited JSON (NDJSON) on stdout, then a final
// sentinel line carrying terminal status + token usage. Human-readable logs go
// to stderr so stdout stays a clean machine stream.
//
//   node cursor-sidecar.mjs --cwd <dir> --model <id> --prompt "<text>"
//   node cursor-sidecar.mjs --cwd <dir> --prompt-file <path>
//
// Exit code: 0 on run status "finished", 2 otherwise, 1 on setup/transport error.

import { readFileSync } from "node:fs";

const SENTINEL = "__sidecar_result__";
const MAX_FIELD = 4000; // truncate oversized tool args/results in the echoed stream

function parseArgs(argv) {
  const out = { cwd: process.cwd(), model: "composer-2.5", prompt: null, timeoutMs: 600000 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--cwd") out.cwd = next();
    else if (a === "--model") out.model = next();
    else if (a === "--prompt") out.prompt = next();
    else if (a === "--prompt-file") out.prompt = readFileSync(next(), "utf8");
    else if (a === "--timeout-ms") out.timeoutMs = Number(next());
    else if (!a.startsWith("--") && out.prompt == null) out.prompt = a;
  }
  return out;
}

function emit(obj) {
  let line;
  try {
    line = JSON.stringify(obj, (_k, v) => {
      if (typeof v === "string" && v.length > MAX_FIELD) return v.slice(0, MAX_FIELD) + "…[truncated]";
      return v;
    });
  } catch {
    line = JSON.stringify({ type: "__unserializable", keys: Object.keys(obj ?? {}) });
  }
  process.stdout.write(line + "\n");
}

function log(...args) {
  process.stderr.write(`[sidecar] ${args.join(" ")}\n`);
}

const opts = parseArgs(process.argv.slice(2));

if (!process.env.CURSOR_API_KEY) {
  log("FATAL: CURSOR_API_KEY is not set");
  process.exit(1);
}
if (!opts.prompt) {
  log("FATAL: no prompt (use --prompt, --prompt-file, or a positional arg)");
  process.exit(1);
}

let Agent;
try {
  ({ Agent } = await import("@cursor/sdk"));
} catch (err) {
  log("FATAL: cannot import @cursor/sdk — run `npm install` in this dir first.", String(err));
  process.exit(1);
}

log(`node ${process.version} · model ${opts.model} · cwd ${opts.cwd}`);

let agent;
let timer;
try {
  agent = await Agent.create({
    apiKey: process.env.CURSOR_API_KEY,
    model: { id: opts.model },
    local: { cwd: opts.cwd },
  });
  log(`agent created: ${agent.agentId}`);

  const run = await agent.send(opts.prompt);
  log(`run started: ${run.id ?? "(no id)"}`);

  timer = setTimeout(() => {
    log(`FATAL: timeout after ${opts.timeoutMs}ms — cancelling run`);
    run.cancel?.().catch(() => {});
  }, opts.timeoutMs);

  for await (const event of run.stream()) {
    emit(event);
  }

  const result = await run.wait();
  clearTimeout(timer);

  emit({
    type: SENTINEL,
    status: result.status,
    agentId: agent.agentId,
    runId: result.id,
    durationMs: result.durationMs ?? null,
    usage: result.usage ?? null,
    resultText: typeof result.result === "string" ? result.result.slice(0, MAX_FIELD) : null,
    error: result.error ?? null,
  });

  log(`run finished: status=${result.status} usage=${result.usage ? result.usage.totalTokens + " tok" : "n/a"}`);
  process.exitCode = result.status === "finished" ? 0 : 2;
} catch (err) {
  clearTimeout(timer);
  log("FATAL: run failed:", err?.stack || String(err));
  emit({ type: SENTINEL, status: "error", error: { message: String(err?.message || err) } });
  process.exitCode = 1;
} finally {
  try {
    await agent?.[Symbol.asyncDispose]?.();
  } catch {
    /* best-effort dispose */
  }
}
