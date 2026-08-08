// Headless transport for Cursor SDK; stdout is NDJSON plus one final sentinel.
import { readFileSync } from 'node:fs';

const SENTINEL = '__sidecar_result__';
function parseArgs(argv) {
  const out = { cwd: process.cwd(), model: 'composer-2.5', prompt: null, timeoutMs: 600000 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]; const next = () => argv[++i];
    if (arg === '--cwd') out.cwd = next();
    else if (arg === '--model') out.model = next();
    else if (arg === '--prompt') out.prompt = next();
    else if (arg === '--prompt-file') out.prompt = readFileSync(next(), 'utf8');
    else if (arg === '--timeout-ms') out.timeoutMs = Number(next());
  }
  return out;
}
const emit = (event) => process.stdout.write(JSON.stringify(event) + '\n');
const options = parseArgs(process.argv.slice(2));
if (!process.env.CURSOR_API_KEY || !options.prompt) process.exit(1);
let agent; let timer;
try {
  const { Agent } = await import('@cursor/sdk');
  agent = await Agent.create({ apiKey: process.env.CURSOR_API_KEY, model: { id: options.model }, local: { cwd: options.cwd } });
  const run = await agent.send(options.prompt);
  timer = setTimeout(() => run.cancel?.().catch(() => {}), options.timeoutMs);
  for await (const event of run.stream()) emit(event);
  const result = await run.wait(); clearTimeout(timer);
  emit({ type: SENTINEL, status: result.status, agentId: agent.agentId, runId: result.id, usage: result.usage ?? null });
  process.exitCode = result.status === 'finished' ? 0 : 2;
} catch (error) {
  clearTimeout(timer);
  emit({ type: SENTINEL, status: 'error', error: { message: String(error?.message || error) } });
  process.exitCode = 1;
} finally {
  try { await agent?.[Symbol.asyncDispose]?.(); } catch { /* best effort */ }
}
