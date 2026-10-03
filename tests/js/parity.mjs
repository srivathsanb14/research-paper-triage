// Reads a JSON job on stdin and prints the browser's answers, so tests/test_web_parity.py can compare
// them with the Python signals and scikit-learn's ridge regression.
import { readFileSync } from "node:fs";
import { ridgeFit } from "../../web/js/engine.js";
import { loadRules, signals } from "../../web/js/quality.js";

const job = JSON.parse(readFileSync(0, "utf8"));
const rules = JSON.parse(readFileSync(new URL("../../triage/quality_rules.json", import.meta.url)));
globalThis.fetch = async () => ({ json: async () => rules });
await loadRules("rules");
const out = {
  signals: job.papers.map(p => signals(p)),
  ridge: ridgeFit(job.ridge.X, job.ridge.y, job.ridge.w),
};
process.stdout.write(JSON.stringify(out));
