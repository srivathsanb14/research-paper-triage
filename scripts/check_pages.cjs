// Run after `python3.12 scripts/build_pages.py` and `npm ci`.
// `BROWSER_CHANNEL=chrome node scripts/check_pages.cjs` uses installed Chrome.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const http = require("node:http");
const fs = require("node:fs/promises");
const path = require("node:path");

const site = path.resolve(__dirname, "../_site");
const prefix = "/research-paper-triage/";
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".zip": "application/zip" };
const server = http.createServer(async (request, response) => {
  const pathname = new URL(request.url, "http://localhost").pathname;
  if (!pathname.startsWith(prefix)) { response.writeHead(404).end(); return; }
  const relative = pathname.slice(prefix.length) || "index.html";
  const file = path.resolve(site, relative);
  if (!file.startsWith(site + path.sep)) { response.writeHead(404).end(); return; }
  try {
    const data = await fs.readFile(file);
    response.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
    response.end(data);
  } catch { response.writeHead(404).end(); }
});

async function main() {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}${prefix}`;
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || undefined, headless: true });
  let page;
  let progress;
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
    page = await context.newPage();
    page.setDefaultTimeout(180000);
    const errors = [];
    page.on("pageerror", error => { errors.push(error.message); console.error("Browser error:", error.message); });
    page.on("requestfailed", request => console.error("Request failed:", request.url(), request.failure()?.errorText));
    await page.goto(url + "?view=READ");
    console.log("Loading browser app at a GitHub Pages-style subpath…");
    progress = setInterval(async () => {
      try { console.log("Loading:", (await page.locator("body").innerText({ timeout: 5000 })).slice(-700)); }
      catch { console.log("Waiting for browser response…"); }
    }, 20000);
    await page.getByRole("tab", { name: "Reading list", exact: true }).waitFor();
    clearInterval(progress);
    console.log("App interface loaded.");
    await page.getByRole("button", { name: /Relevant$/ }).first().waitFor();
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    assert.equal(await page.locator("#loading").isVisible(), false);
    console.log("PASS: sample collection and rankings load.");

    const second = await context.newPage();
    await second.goto(url);
    await second.getByText("Paper Triage is already open", { exact: false }).waitFor();
    await second.close();
    console.log("PASS: a second tab cannot overwrite local data.");

    await page.getByRole("button", { name: /Relevant$/ }).first().click();
    await page.getByRole("tab", { name: "Settings", exact: true }).click();
    await page.getByRole("button", { name: /Download profile backup$/ }).waitFor();
    const beforeDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: /Download profile backup$/ }).click();
    const before = JSON.parse(await fs.readFile(await (await beforeDownload).path(), "utf8"));
    const newest = before.feedback.at(-1);
    assert.equal(newest.action, "useful");
    await page.reload();
    await page.getByRole("button", { name: /Download profile backup$/ }).waitFor();
    const afterDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: /Download profile backup$/ }).click();
    const after = JSON.parse(await fs.readFile(await (await afterDownload).path(), "utf8"));
    assert.deepEqual(after.feedback, before.feedback);
    assert.equal(after.profile.name, before.profile.name);
    console.log("PASS: feedback and profile persist after reload; backups download.");

    await page.getByText("Add papers", { exact: true }).click();
    const collection = [{ id: "test:browser-import", title: "Browser Persistence Test Paper", abstract: "Retrieval augmented generation for scientific question answering." }];
    await page.getByLabel("Paper collection (.json)", { exact: true }).setInputFiles({ name: "papers.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(collection)) });
    await page.getByRole("button", { name: "Import papers", exact: true }).click();
    await page.getByRole("tab", { name: "Reading list", exact: true }).click();
    await page.getByText(/^All \(/).click();
    await page.getByPlaceholder("Search titles and abstracts").fill("Browser Persistence Test Paper");
    await page.getByPlaceholder("Search titles and abstracts").press("Enter");
    await page.locator(".paper-title").filter({ hasText: "Browser Persistence Test Paper" }).waitFor();
    console.log("PASS: imported papers can be ranked and searched.");

    await page.getByRole("tab", { name: "Labeling", exact: true }).click();
    await page.getByText("Sampling", { exact: true }).waitFor({ state: "attached" });
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    console.log("PASS: labeling view works.");
    await page.getByRole("tab", { name: "Evaluation", exact: true }).click();
    await page.getByText("Ranking quality", { exact: true }).waitFor();
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    console.log("PASS: evaluation view works.");
    await page.getByRole("tab", { name: "Reading list", exact: true }).click();
    await page.getByPlaceholder("Search titles and abstracts").fill("");
    await page.getByPlaceholder("Search titles and abstracts").press("Enter");
    await page.locator(".paper-title").first().waitFor();
    await fs.mkdir("test-results", { recursive: true });
    await page.screenshot({ path: "test-results/pages-desktop.png", fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: "test-results/pages-mobile.png", fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log("PASS: no uncaught browser errors; mobile layout fits.");
  } catch (error) {
    if (page) {
      await fs.mkdir("test-results", { recursive: true });
      await page.screenshot({ path: "test-results/pages-failure.png", fullPage: true });
      console.error((await page.locator("body").innerText()).slice(0, 5000));
    }
    throw error;
  } finally { clearInterval(progress); await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => server.close());
