// Main-thread handle on the embedding worker. Status: idle → loading → ready | error.
export const paperText = p => `${p.title}. ${p.abstract || ""}`.slice(0, 2000); // must match scripts/embed_catalog.mjs

export class Embedder extends EventTarget {
  status = "idle";
  progress = 0;
  error = "";
  #worker = null;
  #files = new Map();
  #calls = new Map();
  #next = 0;
  #ready = null;

  start() {
    if (this.#worker) return this.#ready;
    try {
      this.#worker = new Worker(new URL("./embed-worker.js", import.meta.url), { type: "module" });
    } catch (e) {
      this.#set("error", String(e));
      return Promise.reject(e);
    }
    this.#ready = new Promise((resolve, reject) => {
      this.#worker.onmessage = ({ data }) => {
        if (data.type === "progress") {
          this.#files.set(data.file, [data.loaded, data.total]);
          let l = 0, t = 0;
          for (const [a, b] of this.#files.values()) { l += a; t += b; }
          this.progress = t ? l / t : 0;
          this.#set("loading");
        } else if (data.type === "ready") {
          this.progress = 1;
          this.#set("ready");
          resolve();
        } else if (data.type === "error") {
          this.#set("error", data.message);
          reject(new Error(data.message));
        } else if (data.type === "result") {
          const call = this.#calls.get(data.id);
          this.#calls.delete(data.id);
          if (data.error) call?.reject(new Error(data.error));
          else call?.resolve(data.vectors);
        }
      };
      this.#worker.onerror = e => { this.#set("error", e.message || "Worker failed"); reject(e); };
    });
    this.#ready.catch(() => {});
    this.#set("loading");
    this.#worker.postMessage({ type: "load" });
    return this.#ready;
  }

  /** Float32Array[] for each text; waits for the model. */
  async embed(texts) {
    await this.start();
    if (!texts.length) return [];
    const id = ++this.#next;
    return new Promise((resolve, reject) => {
      this.#calls.set(id, { resolve, reject });
      this.#worker.postMessage({ type: "embed", id, texts });
    });
  }

  #set(status, error = "") {
    this.status = status;
    this.error = error;
    this.dispatchEvent(new Event("change"));
  }
}
