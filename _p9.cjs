const { chromium } = require("playwright");
(async () => {
  const b = await chromium.launch({ channel: "chrome" }); const p = await b.newPage();
  p.on("requestfailed", r => console.log("FAILED", r.url().slice(0, 120), r.failure()?.errorText));
  p.on("console", m => console.log("console", m.text().slice(0, 200)));
  await p.goto("http://127.0.0.1:8765/");
  console.log(await p.evaluate(async () => { try { const m = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.web.min.js"); return "page ok " + Object.keys(m).length; } catch (e) { return "page " + e; } }));
  console.log(await p.evaluate(() => new Promise(res => {
    const src = `self.onmessage = async () => { try { const m = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.web.min.js"); postMessage("worker ok " + Object.keys(m).length); } catch (e) { postMessage("worker " + e); } };`;
    const w = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })), { type: "module" }); w.onmessage = e => res(e.data); w.postMessage(1); })));
  await b.close();
})();
