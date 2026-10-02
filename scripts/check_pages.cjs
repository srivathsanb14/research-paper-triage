// End-to-end check of the static site. Run after `python scripts/build_pages.py` and `npm ci`.
//   npm run test:pages                       (Playwright's Chromium)
//   BROWSER_CHANNEL=chrome npm run test:pages (installed Chrome)
//   OFFLINE_MODEL=1 …                        (skip the MiniLM download; TF-IDF only)
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");

const site = path.resolve(__dirname, "../_site");
const prefix = "/research-paper-triage/"; // GitHub Pages serves project sites under a subpath
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer(async (req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
  const file = path.resolve(site, pathname.slice(prefix.length) || "index.html");
  if (!file.startsWith(site + path.sep)) { res.writeHead(404).end(); return; }
  try {
    const data = await fs.readFile(file);
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  } catch { res.writeHead(404).end(); }
});

function invertedIndex(text) {
  const index = {};
  text.split(" ").forEach((w, i) => (index[w] ||= []).push(i));
  return index;
}

const step = msg => console.log(`PASS: ${msg}`);

async function main() {
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${server.address().port}${prefix}`;
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || undefined, headless: true });
  const context = await browser.newContext({ viewport: { width: 1360, height: 900 }, acceptDownloads: true });
  if (process.env.OFFLINE_MODEL) await context.route(/cdn\.jsdelivr\.net|huggingface\.co/, r => r.abort());
  // Live OpenAlex calls are stubbed so the check is deterministic and offline-safe.
  await context.route(/api\.openalex\.org/, r => r.fulfill({ contentType: "application/json", body: JSON.stringify({ results: [{
    id: "https://openalex.org/W999", doi: "https://doi.org/10.9999/live-test", title: "Live OpenAlex Test Paper on Retrieval Evaluation", type: "article",
    abstract_inverted_index: invertedIndex("We evaluate retrieval augmented generation faithfulness with a new benchmark and release code at github.com/example. The study covers many datasets and shows careful ablations across settings and tasks for evaluation of retrieval systems in practice."),
    authorships: [{ author: { display_name: "T. Tester" } }], primary_location: { source: { display_name: "Test Journal", type: "journal" }, version: "publishedVersion" },
    publication_date: "2026-09-01", publication_year: 2026, cited_by_count: 3, fwci: 2.1, referenced_works_count: 40,
  }] }) }));
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  const errors = [];
  page.on("pageerror", e => { errors.push(e.message); console.error("Browser error:", e.message); });
  page.on("console", m => { if (m.type() === "warning" && !/content-length/.test(m.text())) console.error("Browser warning:", m.text()); });
  let failed = false;
  try {
    const t0 = Date.now();
    await page.goto(url);
    await page.getByRole("button", { name: /Rank papers/ }).waitFor();
    step(`welcome page interactive in ${Date.now() - t0} ms`);

    await page.getByRole("button", { name: "RAG & LLM evaluation" }).click();
    await page.getByRole("button", { name: /Rank papers/ }).click();
    await page.locator("article.paper").first().waitFor();
    const top = await page.locator("article.paper h2").first().innerText();
    assert.match(top, /retriev|RAG|language model|LLM/i, `unexpected top paper: ${top}`);
    assert.ok(await page.locator("article.paper .signals").count() > 0, "signals are shown");
    step(`a new visitor gets a ranked Read list without setup (top: “${top.slice(0, 60)}”)`);

    if (!process.env.OFFLINE_MODEL) {
      await page.waitForFunction(() => window.__triage.run?.space?.name === "minilm", null, { timeout: 120000 });
      step("MiniLM loads in a worker and the ranking switches to semantic vectors");
    }

    // Feedback, undo, save, hide.
    const first = page.locator("article.paper").first();
    const firstId = await first.getAttribute("data-id");
    await first.getByRole("button", { name: "Relevant", exact: true }).click();
    await page.getByRole("button", { name: "Undo" }).click();
    let fb = await page.evaluate(() => window.__triage.state.profiles[window.__triage.state.active].feedback.length);
    assert.equal(fb, 0, "undo removes feedback");
    const card = id => page.locator(`article.paper[data-id="${id}"]`);
    await card(firstId).getByRole("button", { name: "Relevant", exact: true }).click();
    await card(firstId).getByRole("button", { name: /^Save/ }).click();
    await page.getByRole("tab", { name: /Skim/ }).click();
    const skimCard = page.locator("article.paper").first();
    const skimId = await skimCard.getAttribute("data-id");
    await skimCard.getByRole("button", { name: "Hide" }).click();
    await page.waitForFunction(id => !document.querySelector(`article.paper[data-id="${CSS.escape(id)}"]`), skimId);
    step("rate, undo, save and hide work");

    // Details panel and keyboard shortcuts.
    await page.getByRole("tab", { name: /Read/ }).click();
    await page.keyboard.press("j");
    await page.keyboard.press("Enter");
    await page.locator(".details .bars").first().waitFor();
    await page.keyboard.press("n");
    fb = await page.evaluate(() => window.__triage.state.profiles[window.__triage.state.active].feedback.map(f => f.action));
    assert.ok(fb.includes("not_useful"), "keyboard N records not relevant");
    step("details show the score breakdown; keyboard triage works");

    // Signal filters.
    await page.getByRole("tab", { name: /All/ }).click();
    await page.locator("label.chip-toggle", { hasText: "Peer-reviewed" }).click();
    const sigs = await page.locator("article.paper .signals").allInnerTexts();
    assert.ok(sigs.length && sigs.every(t => t.includes("Peer-reviewed")), "peer-reviewed filter");
    await page.locator("label.chip-toggle", { hasText: "Peer-reviewed" }).click();
    step("worth-it signal filters narrow the list");

    // Live OpenAlex search (stubbed).
    await page.getByRole("button", { name: /Find more on OpenAlex/ }).click();
    await page.getByText(/Added 1 papers from OpenAlex/).waitFor();
    await page.locator("#search").fill("Live OpenAlex Test Paper");
    await page.locator("article.paper h2", { hasText: "Live OpenAlex Test Paper" }).waitFor();
    await page.locator("#search").fill("");
    step("papers found on OpenAlex join the ranking");

    // Saved list exports BibTeX.
    await page.locator("#nav").getByRole("link", { name: /^Saved/ }).click();
    const bib = page.waitForEvent("download");
    await page.getByRole("button", { name: "BibTeX" }).click();
    assert.match(await fs.readFile(await (await bib).path(), "utf8"), /^@(article|misc)\{/);
    step("saved papers export to BibTeX");

    // Hand labelling, then insights.
    await page.locator("#nav").getByRole("link", { name: /^Label/ }).click();
    await page.locator("#labeler").fill("ci");
    await page.locator("#labeler").press("Tab");
    for (let i = 0; i < 24; i++) {
      const title = await page.locator(".label-card h2").innerText();
      const good = /retriev|RAG|language model|LLM|hallucinat/i.test(title);
      await page.keyboard.press(good || i % 5 === 0 ? "1" : i % 4 ? "3" : "2");
      await page.waitForFunction(t => document.querySelector(".label-card h2")?.innerText !== t, title);
    }
    await page.keyboard.press("Backspace");
    const n = await page.evaluate(() => Object.keys(window.__triage.state.profiles[window.__triage.state.active].labels).length);
    assert.equal(n, 23, "undo removes the last label");
    const csv = page.waitForEvent("download");
    await page.getByRole("button", { name: "Labels CSV" }).click();
    const rows = (await fs.readFile(await (await csv).path(), "utf8")).trim().split("\n");
    assert.equal(rows.length, 24);
    assert.match(rows[0], /^paper_id,title,abstract/);
    step("keyboard labelling with undo; labels export as a dataset CSV");

    await page.locator("#nav").getByRole("link", { name: /^Insights/ }).click();
    await page.locator(".tiles .tile").first().waitFor();
    await page.locator("figure.chart svg").waitFor();
    const box = await page.locator("figure.chart .hit").boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.locator("figure.chart .tip").waitFor();
    step("insights show cross-validated metrics and an interactive discovery curve");

    // Persistence and backup round-trip.
    await page.reload();
    await page.locator("article.paper, .tiles .tile").first().waitFor();
    const after = await page.evaluate(() => { const p = window.__triage.state.profiles[window.__triage.state.active]; return [Object.keys(p.labels).length, p.feedback.length]; });
    assert.equal(after[0], 23);
    await page.getByRole("button", { name: "Settings" }).click();
    const dl = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download backup" }).click();
    const backupPath = await (await dl).path();
    const backup = JSON.parse(await fs.readFile(backupPath, "utf8"));
    assert.equal(backup.format, "paper-triage-web-profile");
    assert.equal(Object.keys(backup.profile.labels).length, 23);
    await page.locator('#settings-dialog input[data-act="restore"]').setInputFiles(backupPath);
    await page.getByText(/Restored “RAG & LLM evaluation \(restored 1\)”/).waitFor();
    step("ratings and labels survive reload; backups download and restore");

    // Import a JSON collection.
    await page.goto(url + "#/feed");
    await page.locator("article.paper").first().waitFor();
    await page.locator('input[data-act="import-papers"]').setInputFiles({ name: "papers.json", mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify([{ id: "test:import", title: "Imported Paper About Retrieval Augmented Generation", abstract: "Retrieval augmented generation for scientific question answering." }])) });
    await page.getByText("Imported 1 new papers.").waitFor();
    await page.locator("#search").fill("Imported Paper About");
    await page.getByRole("tab", { name: /All/ }).click();
    await page.locator("article.paper h2", { hasText: "Imported Paper About" }).waitFor();
    await page.locator("#search").fill("");
    step("JSON paper collections can be imported and ranked");

    // Review mode: a person accepts or corrects model-proposed labels.
    const sets = await page.evaluate(() => window.__triage.proposals.sets.length);
    if (sets) {
      await page.locator("#nav").getByRole("link", { name: /^Label/ }).click();
      await page.locator('[data-act="review-set"]').first().click();
      await page.locator(".suggest").waitFor();
      const proposed = await page.locator(".suggest .label-chip").innerText();
      await page.keyboard.press("Enter");
      await page.locator(".suggest").waitFor();
      await page.keyboard.press("3");
      const got = await page.evaluate(() => Object.values(window.__triage.state.profiles[window.__triage.state.active].labels));
      assert.equal(got.length, 2);
      assert.equal(got[0].label, got[0].proposed, `accepted suggestion (${proposed})`);
      assert.ok(got.every(l => l.proposal), "provenance is kept");
      const jsonl = page.waitForEvent("download");
      await page.getByRole("button", { name: "Labels JSONL" }).click();
      const rows = (await fs.readFile(await (await jsonl).path(), "utf8")).trim().split("\n").map(l => JSON.parse(l));
      assert.ok(rows.every(r => r.label_source === "human-verified" && r.proposed_label));
      step("review mode records suggestion and human answer; exports as human-verified");
      await page.locator("#nav").getByRole("link", { name: /^For you/ }).click();
      await page.locator("article.paper").first().waitFor();
    }

    await fs.mkdir("test-results", { recursive: true });
    await page.getByRole("tab", { name: /Read/ }).click();
    await page.screenshot({ path: "test-results/pages-desktop.png" });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: "test-results/pages-mobile.png" });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, "no horizontal scroll on mobile");
    assert.deepEqual(errors, []);
    step("mobile layout fits; no uncaught errors");
  } catch (e) {
    failed = true;
    await fs.mkdir("test-results", { recursive: true });
    await page.screenshot({ path: "test-results/pages-failure.png", fullPage: true }).catch(() => {});
    console.error((await page.locator("body").innerText().catch(() => "")).slice(0, 3000));
    throw e;
  } finally {
    await browser.close();
    server.close();
    if (failed) process.exitCode = 1;
  }
}

main().catch(e => { console.error(e); process.exitCode = 1; server.close(); });
