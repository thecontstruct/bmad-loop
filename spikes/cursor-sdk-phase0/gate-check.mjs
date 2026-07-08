// Phase-0 GO/NO-GO gate.
//
// The load-bearing question from the research: will a headless Cursor *local*
// agent run unattended and write a bmad-loop-style result.json to an exact
// path on the local working tree? This drives the sidecar exactly as a Python
// CursorSdkAdapter would (spawn a child, read NDJSON), then verifies the file.
//
//   node gate-check.mjs [--model <id>] [--keep]
//
// Exit 0 = PASS (result.json written + valid), 1 = FAIL.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const model = (() => {
  const i = argv.indexOf("--model");
  return i >= 0 ? argv[i + 1] : "composer-2.5";
})();
const keep = argv.includes("--keep");

const tmp = join(here, ".gate-tmp");
const resultRel = join("tasks", "spike-task", "result.json");
const resultAbs = join(tmp, resultRel);
const promptFile = join(tmp, "prompt.txt");

// Fresh working tree with one real file so the agent has genuine context.
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp, { recursive: true });
writeFileSync(join(tmp, "README.md"), "# Gate scratch repo\nUsed by the Cursor SDK Phase-0 gate.\n");

const expected = {
  status: "completed",
  task_id: "spike-task",
  summary: "Phase-0 gate: prove headless local agent writes result.json",
  spike: true,
};

const prompt = [
  "You are running fully headless in an automation harness. No human will approve anything.",
  `Create a file at the relative path \`${resultRel}\` (create parent directories as needed).`,
  "Its entire contents must be EXACTLY this JSON and nothing else (no markdown, no fences):",
  "",
  JSON.stringify(expected, null, 2),
  "",
  "Do not create, edit, or delete any other file. After the file is written, stop.",
].join("\n");
writeFileSync(promptFile, prompt);

console.log(`[gate] model=${model} cwd=${tmp}`);
console.log(`[gate] expecting result at: ${resultAbs}`);

const child = spawn(
  process.execPath,
  [join(here, "cursor-sidecar.mjs"), "--cwd", tmp, "--model", model, "--prompt-file", promptFile],
  { env: process.env },
);

let sentinel = null;
let stdoutBuf = "";
let eventCount = 0;

child.stdout.on("data", (chunk) => {
  stdoutBuf += chunk.toString();
  let nl;
  while ((nl = stdoutBuf.indexOf("\n")) >= 0) {
    const line = stdoutBuf.slice(0, nl).trim();
    stdoutBuf = stdoutBuf.slice(nl + 1);
    if (!line) continue;
    eventCount++;
    try {
      const obj = JSON.parse(line);
      if (obj.type === "__sidecar_result__") sentinel = obj;
    } catch {
      /* ignore non-JSON noise */
    }
  }
});
child.stderr.on("data", (c) => process.stderr.write(c));

child.on("close", (code) => {
  const checks = [];
  const record = (name, ok, detail = "") => {
    checks.push({ name, ok, detail });
    console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
  };

  console.log("\n[gate] results:");
  record("sidecar streamed events", eventCount > 0, `${eventCount} lines`);
  record("sidecar emitted terminal sentinel", !!sentinel, sentinel ? `status=${sentinel.status}` : "missing");
  record("run status is finished", sentinel?.status === "finished", sentinel?.status ?? "n/a");
  record("token usage reported", !!sentinel?.usage, sentinel?.usage ? `${sentinel.usage.totalTokens} tok` : "none");

  let fileOk = false;
  let parsed = null;
  let schemaOk = false;
  if (existsSync(resultAbs)) {
    fileOk = true;
    try {
      parsed = JSON.parse(readFileSync(resultAbs, "utf8"));
      schemaOk = parsed && parsed.status === "completed" && parsed.task_id === "spike-task" && parsed.spike === true;
    } catch (e) {
      parsed = { __parse_error: String(e) };
    }
  }
  record("result.json written to exact path", fileOk, resultAbs);
  record("result.json is valid + matches contract", schemaOk, parsed ? JSON.stringify(parsed).slice(0, 200) : "no file");

  // The GATE itself is the file write; streaming/usage are secondary evidence.
  const gatePass = fileOk && schemaOk;
  const summary = {
    gate: gatePass ? "PASS" : "FAIL",
    childExitCode: code,
    eventCount,
    runStatus: sentinel?.status ?? null,
    usage: sentinel?.usage ?? null,
    resultWritten: fileOk,
    resultValid: schemaOk,
    model,
    timestamp: new Date().toISOString(),
  };
  writeFileSync(join(here, "gate-result.json"), JSON.stringify(summary, null, 2) + "\n");

  console.log(`\n[gate] ${gatePass ? "✅ GO — SDK-local can back a bmad-loop provider" : "❌ NO-GO — see failures above"}`);
  console.log(`[gate] summary written to gate-result.json`);
  if (!keep) rmSync(tmp, { recursive: true, force: true });
  process.exit(gatePass ? 0 : 1);
});
