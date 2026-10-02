// Local persistence: everything stays in this browser (IndexedDB).
// "state" holds profiles, feedback and labels; "papers" holds papers fetched or
// imported by the visitor (with their embeddings). Falls back to memory when
// storage is blocked (private windows, previews), so the app still works.
const DB_NAME = "paper-triage";
const DB_VERSION = 1;
let dbPromise = null;
let memory = { state: null, papers: new Map() };

function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise(resolve => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains("state")) db.createObjectStore("state");
        if (!db.objectStoreNames.contains("papers")) db.createObjectStore("papers", { keyPath: "id" });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch { resolve(null); }
  });
  return dbPromise;
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const result = fn(t.objectStore(store));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export async function persistent() {
  return !!(await open());
}

export async function loadState() {
  const db = await open();
  if (!db) return memory.state;
  try {
    return await new Promise((resolve, reject) => {
      const req = db.transaction("state").objectStore("state").get("app");
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
  } catch { return memory.state; }
}

let pending = null;
let writing = null;
/**
 * Write-through: the write starts immediately, and calls made while one is in
 * flight are coalesced into a single follow-up write of the latest state. (A
 * debounce timer could still be pending when the page unloads, losing data.)
 */
export function saveState(state) {
  memory.state = state;
  pending = state;
  writing ??= (async () => {
    const db = await open();
    while (pending) {
      const next = pending;
      pending = null;
      if (!db) continue;
      try { await tx(db, "state", "readwrite", s => s.put(next, "app")); } catch (e) { console.warn("Could not save state", e); }
    }
  })().finally(() => {
    writing = null;
    if (pending) saveState(pending); // arrived after the loop's last check
  });
  return writing;
}

/** Resolves once every queued write has been committed. */
export async function flush() {
  while (writing) await writing;
}

export async function loadPapers() {
  const db = await open();
  if (!db) return [...memory.papers.values()];
  try {
    return await new Promise((resolve, reject) => {
      const req = db.transaction("papers").objectStore("papers").getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  } catch { return [...memory.papers.values()]; }
}

export async function putPapers(papers) {
  for (const p of papers) memory.papers.set(p.id, p);
  const db = await open();
  if (!db) return;
  try { await tx(db, "papers", "readwrite", s => { for (const p of papers) s.put(p); }); } catch (e) { console.warn("Could not save papers", e); }
}

export async function clearAll() {
  memory = { state: null, papers: new Map() };
  const db = await open();
  if (!db) return;
  await tx(db, "state", "readwrite", s => s.clear());
  await tx(db, "papers", "readwrite", s => s.clear());
}
