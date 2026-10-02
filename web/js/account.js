// Optional accounts. The static site has no server, so everything here is a no-op
// unless the Paper Triage server (python -m server) answers at ./api/. When it does,
// a signed-in user's profiles, ratings and labels are synced to their account.
const API = new URL("../api/", import.meta.url);

export const acct = { available: false, user: null, updated: null, conflict: false };

async function call(path, { method = "GET", body, keepalive = false } = {}) {
  const res = await fetch(new URL(path, API), {
    method, keepalive, credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = (res.headers.get("content-type") || "").includes("json") ? await res.json().catch(() => null) : null;
  if (!res.ok) throw Object.assign(new Error(json?.detail ? (Array.isArray(json.detail) ? json.detail[0]?.msg : json.detail) : `Request failed (${res.status})`), { status: res.status });
  return json;
}

/** Detects the backend and any existing session. Never throws. */
export async function init() {
  try {
    const me = await call("me");
    acct.available = me?.backend === true;
    acct.user = me?.user ?? null;
  } catch { acct.available = false; acct.user = null; }
  return acct;
}

export async function signIn(username, password) {
  acct.user = (await call("login", { method: "POST", body: { username, password } })).user;
  return acct.user;
}

export async function register(username, password) {
  acct.user = (await call("register", { method: "POST", body: { username, password } })).user;
  return acct.user;
}

export async function signOut() {
  await flush();
  await call("logout", { method: "POST", body: {} }).catch(() => {});
  Object.assign(acct, { user: null, updated: null, conflict: false });
}

export async function pullState() {
  const r = await call("state");
  acct.updated = r.updated;
  acct.conflict = false;
  return r.state;
}

let latest = null;
let timer = null;
let inflight = null;

async function send(keepalive = false) {
  while (latest && acct.user && !acct.conflict) {
    const state = latest;
    latest = null;
    try {
      acct.updated = (await call("state", { method: "PUT", body: { state, base: acct.updated }, keepalive })).updated;
    } catch (e) {
      if (e.status === 409) { acct.conflict = true; onConflict?.(); } else console.warn("Could not sync to your account", e);
      return;
    }
  }
}

let onConflict = null;
export const onSyncConflict = fn => { onConflict = fn; };

/** One upload at a time: `base` must be the stamp from the previous response, or the server reports a conflict. */
function run(keepalive = false) {
  const prev = inflight ?? Promise.resolve();
  const mine = prev.then(() => send(keepalive)).finally(() => { if (inflight === mine) inflight = null; });
  inflight = mine;
  return mine;
}

/** Queue the latest state for upload (coalesced; one request at a time). No-op when signed out. */
export function queue(state) {
  if (!acct.user) return;
  latest = state;
  clearTimeout(timer);
  timer = setTimeout(() => { timer = null; run(); }, 600);
}

/** Send anything still waiting; used when the page is hidden or the user signs out. */
export async function flush(keepalive = false) {
  clearTimeout(timer);
  timer = null;
  if (latest) await run(keepalive);
  else if (inflight) await inflight;
}

export const seedSets = () => call("seed-sets").then(r => r.sets);
export const seedSet = slug => call(`seed-sets/${encodeURIComponent(slug)}`);
