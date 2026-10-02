// Discovery curve: share of good papers found while reading down a ranked list.
// Inline SVG with a crosshair tooltip; colours come from CSS custom properties.
import { esc } from "./ui.js";

const W = 640, H = 300, M = { l: 44, r: 16, t: 14, b: 38 };

/** series: [{name, values (0..1, one per rank), cls: "s1"|"s2"|"ref"|"ideal"}] */
export function discoveryChart(series, { title = "Good papers found as you read down the list" } = {}) {
  const n = series[0].values.length;
  const x = k => M.l + ((k - 1) / Math.max(1, n - 1)) * (W - M.l - M.r);
  const y = v => M.t + (1 - v) * (H - M.t - M.b);
  const path = vals => vals.map((v, i) => `${i ? "L" : "M"}${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticksX = [...new Set([1, Math.round(n / 4), Math.round(n / 2), Math.round((3 * n) / 4), n])].filter(k => k >= 1);
  const grid = [0, 0.25, 0.5, 0.75, 1].map(v => `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("");
  const xt = ticksX.map(k => `<text class="tick" x="${x(k)}" y="${H - M.b + 18}" text-anchor="middle">${k}</text>`).join("");
  const lines = series.map(s => `<path class="line ${s.cls}" d="${path(s.values)}"/>`).join("");
  const legend = series.map(s => `<span class="lg"><i class="sw ${s.cls}"></i>${esc(s.name)}</span>`).join("");
  const data = esc(JSON.stringify(series.map(s => ({ name: s.name, cls: s.cls, v: s.values.map(v => Math.round(v * 1000) / 10) }))));
  return `<figure class="chart" data-series="${data}" data-n="${n}">
    <figcaption>${esc(title)}</figcaption>
    <div class="chart-wrap">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
        ${grid}<line class="axis" x1="${M.l}" x2="${W - M.r}" y1="${y(0)}" y2="${y(0)}"/>${xt}
        <text class="axis-title" x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">Papers reviewed, in ranked order</text>
        ${lines}
        <line class="cross" x1="0" x2="0" y1="${M.t}" y2="${y(0)}" visibility="hidden"/>
        <g class="dots"></g>
        <rect class="hit" x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="transparent"/>
      </svg>
      <div class="tip" hidden></div>
    </div>
    <div class="legend">${legend}</div>
  </figure>`;
}

/** Wire hover behaviour for every chart inside `root`. */
export function bindCharts(root) {
  for (const fig of root.querySelectorAll("figure.chart")) {
    const series = JSON.parse(fig.dataset.series);
    const n = +fig.dataset.n;
    const svg = fig.querySelector("svg");
    const hit = fig.querySelector(".hit");
    const cross = fig.querySelector(".cross");
    const dots = fig.querySelector(".dots");
    const tip = fig.querySelector(".tip");
    const x = k => M.l + ((k - 1) / Math.max(1, n - 1)) * (W - M.l - M.r);
    const y = v => M.t + (1 - v / 100) * (H - M.t - M.b);
    const show = evt => {
      const pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      const k = Math.min(n, Math.max(1, Math.round(1 + ((loc.x - M.l) / (W - M.l - M.r)) * (n - 1))));
      cross.setAttribute("x1", x(k)); cross.setAttribute("x2", x(k)); cross.setAttribute("visibility", "visible");
      dots.innerHTML = series.map(s => `<circle class="dot ${s.cls}" cx="${x(k)}" cy="${y(s.v[k - 1])}" r="4.5"/>`).join("");
      tip.innerHTML = `<b>After ${k} paper${k > 1 ? "s" : ""}</b>` + series.map(s => `<div><i class="sw ${s.cls}"></i>${esc(s.name)}<span>${Math.round(s.v[k - 1])}%</span></div>`).join("");
      tip.hidden = false;
      const box = fig.querySelector(".chart-wrap").getBoundingClientRect();
      const left = evt.clientX - box.left;
      tip.style.left = `${Math.min(left + 14, box.width - tip.offsetWidth - 4)}px`;
      tip.style.top = `8px`;
    };
    const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.innerHTML = ""; };
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerleave", hide);
  }
}
