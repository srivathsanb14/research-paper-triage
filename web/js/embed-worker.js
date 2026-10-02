// Runs the off-the-shelf MiniLM sentence-embedding model off the main thread.
// Model weights (~23 MB, quantized) come from the Hugging Face Hub and are
// cached by the browser; the same model and quantization embedded the catalog
// (scripts/embed_catalog.mjs), so profile and paper vectors share one space.
// The browser build, pinned; a second CDN covers an outage of the first.
const LIBS = [
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js",
  "https://unpkg.com/@huggingface/transformers@3.8.1/dist/transformers.min.js",
];
const MODEL = "Xenova/all-MiniLM-L6-v2";

let extractor = null;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function retry(fn, tries = 3) {
  for (let i = 0; ; i++) {
    try { return await fn(); } catch (e) { if (i + 1 >= tries) throw e; await sleep(1500 * (i + 1)); }
  }
}
let loading = null;

async function load() {
  loading ??= (async () => {
    // Flaky connections drop requests; retry with backoff before giving up.
    let lib, lastError;
    for (let attempt = 0; attempt < 3 && !lib; attempt++) {
      if (attempt) await sleep(1500 * attempt);
      for (const url of LIBS) {
        try { lib = await import(url); break; } catch (e) { lastError = e; }
      }
    }
    if (!lib) throw lastError;
    const { pipeline, env } = lib;
    env.allowLocalModels = false;
    extractor = await retry(() => pipeline("feature-extraction", MODEL, {
      dtype: "q8",
      progress_callback: p => {
        if (p.status === "progress" && p.total) self.postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total });
      },
    }));
    self.postMessage({ type: "ready" });
  })().catch(error => {
    loading = null;
    self.postMessage({ type: "error", message: String(error?.message || error) });
    throw error;
  });
  return loading;
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") { load().catch(() => {}); return; }
  if (data.type === "embed") {
    try {
      await load();
      const out = [];
      for (let i = 0; i < data.texts.length; i += 16) {
        const res = await extractor(data.texts.slice(i, i + 16), { pooling: "mean", normalize: true });
        const dim = res.dims[1];
        for (let j = 0; j < res.dims[0]; j++) out.push(res.data.slice(j * dim, (j + 1) * dim));
      }
      self.postMessage({ type: "result", id: data.id, vectors: out }, out.map(v => v.buffer));
    } catch (error) {
      self.postMessage({ type: "result", id: data.id, error: String(error?.message || error) });
    }
  }
};
