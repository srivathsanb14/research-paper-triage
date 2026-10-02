// Launches the Python server (site + accounts) from the repo root: node scripts/run_server.cjs [port]
const { spawn } = require("node:child_process");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const py = path.join(root, ".venv", "bin", "python");
const child = spawn(py, ["-m", "server", "--port", process.argv[2] || "8000"], { cwd: root, stdio: "inherit", env: { ...process.env, PYTHONPATH: root } });
child.on("exit", code => process.exit(code ?? 1));
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => child.kill(sig));
