const loading = document.getElementById("loading");
const status = document.getElementById("loading-status");
const retry = document.getElementById("retry");
retry.addEventListener("click", () => location.reload());

function showError(message) {
  status.textContent = message;
  retry.hidden = false;
}

window.addEventListener("error", () => {
  if (!loading.hidden) showError("Couldn’t start Paper Triage. Check your connection and try again.");
});

async function start() {
  // One writer per origin/path prevents two tabs from overwriting the same
  // IndexedDB filesystem with independently loaded SQLite snapshots.
  if (!navigator.locks || !window.indexedDB) {
    showError("This app needs a current browser with local storage enabled. Try Chrome, Firefox, Edge or Safari.");
    return;
  }
  const appPath = new URL("./", import.meta.url).pathname;
  const storagePath = `/paper-triage-${encodeURIComponent(appPath)}`;
  await navigator.locks.request(`paper-triage:${appPath}`, { ifAvailable: true }, async (lock) => {
    if (!lock) {
      showError("Paper Triage is already open in another tab. Use that tab, or close it and try again here.");
      return;
    }
    try {
      const response = await fetch(new URL("./app-config.json", import.meta.url));
      if (!response.ok) throw new Error("The app configuration could not be loaded.");
      const streamlitConfig = await response.json();
      const { mount } = await import("https://cdn.jsdelivr.net/npm/@stlite/browser@1.9.2/build/stlite.js");
      const root = document.getElementById("root");
      const observer = new MutationObserver(() => {
        if (root.querySelector('.page-title, [data-testid="stException"]')) {
          loading.hidden = true;
          observer.disconnect();
        }
      });
      observer.observe(root, { childList: true, subtree: true });
      mount({
        entrypoint: "app.py",
        archives: [{ url: new URL("./app.zip", import.meta.url).href, format: "zip" }],
        requirements: ["numpy", "pandas", "scipy", "scikit-learn", "requests", "sqlite3", "toml"],
        env: {
          TRIAGE_BROWSER_MODE: "1",
          TRIAGE_EMBEDDING_BACKEND: "tfidf",
          TRIAGE_DB_PATH: `${storagePath}/triage.db`,
        },
        idbfsMountpoints: [storagePath],
        streamlitConfig,
      }, root);
      // The browser releases this lock when the page closes or navigates away.
      await new Promise(() => {});
    } catch (error) {
      console.error(error);
      showError("Couldn’t load Paper Triage. Check your connection and try again.");
    }
  });
}

start().catch(() => showError("Browser storage is unavailable. Allow site storage and try again."));
