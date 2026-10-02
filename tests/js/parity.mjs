// Reads a JSON job on stdin and prints the browser engine's answers, so
// tests/test_web_parity.py can compare them with the Python reference.
import { readFileSync } from "node:fs";
import { evidence, documentFrequencies, reason, ridgeFit } from "../../web/js/engine.js";
import { phraseInText } from "../../web/js/text.js";
import { loadRules, signals } from "../../web/js/quality.js";

const job = JSON.parse(readFileSync(0, "utf8"));
const rules = JSON.parse(readFileSync(new URL("../../triage/quality_rules.json", import.meta.url)));
globalThis.fetch = async () => ({ json: async () => rules });
await loadRules("rules");
const df = documentFrequencies(job.papers);
const out = {
  phrases: job.phrases.map(([phrase, text]) => phraseInText(phrase, text)),
  signals: job.papers.map(p => signals(p)),
  reasons: job.papers.map((p, i) => reason(job.labels[i], evidence(p, job.profile, df), job.features[i], job.profile.focus)),
  ridge: ridgeFit(job.ridge.X, job.ridge.y, job.ridge.w),
};
process.stdout.write(JSON.stringify(out));
