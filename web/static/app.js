/* 县域比率分层估计与总量校准 —— 前端逻辑（无依赖，SVG 直绘） */
"use strict";

const state = {
  scenario: "baseline",
  locked: new Set(),
  selectedId: null,
  data: null,        // /api/state 响应
  validation: null,  // /api/validation 响应
};

const SERIES = [
  { key: "raw",    name: "原始率",   color: "var(--series-1)" },
  { key: "shrunk", name: "收缩率",   color: "var(--series-2)" },
  { key: "cal",    name: "校准后率", color: "var(--series-3)" },
];

const $ = (sel) => document.querySelector(sel);
const pct = (x, d = 1) => (x == null ? "—" : (x * 100).toFixed(d) + "%");
const num = (x) => (x == null ? "—" : Number(x).toLocaleString("zh-CN"));

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
function textEl(x, y, str, attrs = {}) {
  const t = svgEl("text", { x, y, ...attrs });
  t.textContent = str;
  return t;
}

/* ---------- 工具提示 ---------- */
const tooltip = $("#tooltip");
function showTooltip(x, y, buildFn) {
  tooltip.textContent = "";
  buildFn(tooltip);
  tooltip.hidden = false;
  const r = tooltip.getBoundingClientRect();
  tooltip.style.left = Math.min(x + 14, window.innerWidth - r.width - 8) + "px";
  tooltip.style.top = Math.max(8, y - r.height - 10) + "px";
}
function hideTooltip() { tooltip.hidden = true; }
function ttRow(parent, color, label, value) {
  const row = document.createElement("div");
  row.className = "tt-row";
  const key = document.createElement("span");
  key.className = "tt-key";
  key.style.background = color;
  const lab = document.createElement("span");
  lab.textContent = label;
  const val = document.createElement("span");
  val.className = "tt-val";
  val.textContent = value;
  row.append(key, lab, val);
  parent.appendChild(row);
}

/* ---------- 数据获取 ---------- */
async function refresh() {
  const main = $("#main");
  main.classList.add("loading");
  const lockedParam = [...state.locked].join(",");
  const q = `scenario=${state.scenario}&locked=${encodeURIComponent(lockedParam)}`;
  const resp = await fetch("/api/state?" + q);
  const data = await resp.json();
  if (data.error) { alert("计算失败: " + data.error); main.classList.remove("loading"); return; }
  state.data = data;
  $("#export-csv").href = "/api/export/estimates.csv?" + q;
  $("#export-json").href = "/api/export/summary.json?" + q;
  renderAll();
  main.classList.remove("loading");
}

/* ---------- 渲染：横幅与磁贴 ---------- */
function renderBanner() {
  const cal = state.data.estimates.calibration;
  const banner = $("#banner");
  banner.className = "banner show " + (cal.feasible ? "ok" : "critical");
  banner.textContent = "";
  const icon = document.createElement("span");
  icon.className = "icon";
  icon.textContent = cal.feasible ? "✓" : "⚠";
  banner.appendChild(icon);
  banner.appendChild(document.createTextNode(cal.message));
  if (!cal.feasible) {
    const d = document.createElement("div");
    d.className = "muted";
    d.textContent = "已确认锁定项保持不变；可解锁部分确认项后重算，或修正全市总量。";
    banner.appendChild(d);
  }
}

function renderTiles() {
  const e = state.data.estimates;
  const ov = state.data.validation_overall;
  const tiles = [
    { label: "全市确认总量 T", value: num(e.scenario.city_total),
      sub: e.calibration.feasible ? `校准后总量 ${num(Math.round(e.calibration.achieved_total))}` : "校准不可行" },
    { label: "校准前模型隐含总量", value: num(Math.round(e.model_total_before_calibration)),
      sub: `λ = ${e.calibration.feasible ? e.calibration.lambda.toExponential(2) : "—"}` },
    { label: "收缩强度 τ̂²", value: e.model.tau2.toFixed(3),
      sub: `由 ${e.model.n_train} 个有样本县估计（未用人口量）` },
    { label: "留出验证覆盖率", value: pct(ov.coverage_shrunk, 0),
      sub: `外推 ${pct(ov.coverage_extrapolated, 0)} · 名义 90%` },
  ];
  const box = $("#tiles");
  box.textContent = "";
  for (const t of tiles) {
    const el = document.createElement("div");
    el.className = "tile";
    const l = document.createElement("div"); l.className = "label"; l.textContent = t.label;
    const v = document.createElement("div"); v.className = "value"; v.textContent = t.value;
    const s = document.createElement("div"); s.className = "sub"; s.textContent = t.sub;
    el.append(l, v, s);
    box.appendChild(el);
  }
}

/* ---------- 图 1：三率区间图（横向 caterpillar） ---------- */
function renderRatesChart() {
  const counties = state.data.estimates.counties;
  const feasible = state.data.estimates.calibration.feasible;

  // 按地区分组（保持数据顺序）
  const groups = [];
  for (const c of counties) {
    const g = groups.find((g) => g.region === c.region);
    if (g) g.rows.push(c); else groups.push({ region: c.region, rows: [c] });
  }

  const rowH = 24, headerH = 18;
  const margin = { t: 8, r: 16, b: 26, l: 76 };
  const W = 880;
  const plotH = counties.length * rowH + groups.length * headerH;
  const H = margin.t + plotH + margin.b;
  const plotW = W - margin.l - margin.r;

  let maxV = 0;
  for (const c of counties) {
    maxV = Math.max(maxV, c.raw_rate || 0, c.shrunk_hi, feasible ? c.cal_hi || 0 : 0);
  }
  const xMax = Math.ceil(maxV * 20) / 20; // 5% 步进取整
  const x = (v) => margin.l + (v / xMax) * plotW;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "各县原始率、收缩率、校准后率及90%区间" });

  // 网格与 x 轴
  for (let v = 0; v <= xMax + 1e-9; v += 0.05) {
    svg.appendChild(svgEl("line", { x1: x(v), y1: margin.t, x2: x(v), y2: margin.t + plotH,
      stroke: "var(--gridline)", "stroke-width": 1 }));
    svg.appendChild(textEl(x(v), H - 8, pct(v, 0),
      { "text-anchor": "middle", "font-size": 10, fill: "var(--text-muted)" }));
  }
  svg.appendChild(svgEl("line", { x1: margin.l, y1: margin.t + plotH, x2: margin.l + plotW,
    y2: margin.t + plotH, stroke: "var(--baseline)", "stroke-width": 1 }));

  // 行位置索引
  let yCursor = margin.t;
  const rowY = new Map();
  for (const g of groups) {
    svg.appendChild(textEl(margin.l - 6, yCursor + 12, g.region,
      { "text-anchor": "end", "font-size": 11, fill: "var(--text-muted)", "font-weight": 600 }));
    svg.appendChild(svgEl("line", { x1: margin.l, y1: yCursor + headerH - 4, x2: margin.l + plotW,
      y2: yCursor + headerH - 4, stroke: "var(--gridline)", "stroke-width": 1 }));
    yCursor += headerH;
    for (const c of g.rows) {
      rowY.set(c.id, yCursor + rowH / 2);
      yCursor += rowH;
    }
  }

  for (const c of counties) {
    const y = rowY.get(c.id);
    // 县名
    const nameAttrs = { "text-anchor": "end", "font-size": 11,
      fill: c.extrapolated ? "var(--text-muted)" : "var(--text-primary)" };
    svg.appendChild(textEl(margin.l - 6, y + 4, c.name, nameAttrs));
    // 选中高亮
    if (c.id === state.selectedId) {
      svg.appendChild(svgEl("rect", { x: margin.l, y: y - rowH / 2, width: plotW, height: rowH,
        fill: "var(--series-1)", opacity: 0.08 }));
    }
    // 收缩率：区间线 + 点（y - 3）
    const ys = y - 3;
    svg.appendChild(svgEl("line", { x1: x(c.shrunk_lo), y1: ys, x2: x(c.shrunk_hi), y2: ys,
      stroke: "var(--series-2)", "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55 }));
    svg.appendChild(svgEl("circle", { cx: x(c.shrunk_rate), cy: ys, r: 4.5,
      fill: "var(--series-2)", stroke: "var(--surface-1)", "stroke-width": 2 }));
    // 校准后率：区间线 + 点（y + 4）
    if (feasible && c.cal_rate != null) {
      const yc = y + 4;
      svg.appendChild(svgEl("line", { x1: x(c.cal_lo), y1: yc, x2: x(c.cal_hi), y2: yc,
        stroke: "var(--series-3)", "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55 }));
      svg.appendChild(svgEl("circle", { cx: x(c.cal_rate), cy: yc, r: 4.5,
        fill: "var(--series-3)", stroke: "var(--surface-1)", "stroke-width": 2 }));
    }
    // 原始率：仅点（y - 10）
    if (c.raw_rate != null) {
      svg.appendChild(svgEl("circle", { cx: x(c.raw_rate), cy: y - 10, r: 4,
        fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 }));
    }
    // 命中区域（整行，含键盘可达）
    const hit = svgEl("rect", { x: margin.l, y: y - rowH / 2, width: plotW, height: rowH,
      fill: "transparent", tabindex: 0, role: "button",
      "aria-label": `${c.name} 原始率 ${pct(c.raw_rate)} 收缩率 ${pct(c.shrunk_rate)} 校准后率 ${pct(c.cal_rate)}` });
    hit.addEventListener("pointermove", (ev) => showTooltip(ev.clientX, ev.clientY, (tt) => {
      const t = document.createElement("div"); t.className = "tt-title"; t.textContent = `${c.name}（${c.region}）`;
      tt.appendChild(t);
      ttRow(tt, "var(--series-1)", "原始率", pct(c.raw_rate));
      ttRow(tt, "var(--series-2)", "收缩率", `${pct(c.shrunk_rate)} [${pct(c.shrunk_lo)}, ${pct(c.shrunk_hi)}]`);
      if (c.cal_rate != null) ttRow(tt, "var(--series-3)", "校准后率", `${pct(c.cal_rate)} [${pct(c.cal_lo)}, ${pct(c.cal_hi)}]`);
      const n = document.createElement("div"); n.className = "tt-note";
      n.textContent = `样本 ${c.n_sample} · 有效 ${c.n_eff}` + (c.extrapolated ? " · 无样本·模型外推" : "") + (c.locked ? " · 已锁定" : "");
      tt.appendChild(n);
    }));
    hit.addEventListener("pointerleave", hideTooltip);
    hit.addEventListener("click", () => selectCounty(c.id));
    hit.addEventListener("focus", () => selectCounty(c.id));
    svg.appendChild(hit);
  }

  const host = $("#chart-rates");
  host.textContent = "";
  host.appendChild(makeLegend(feasible ? SERIES : SERIES.slice(0, 2)));
  host.appendChild(svg);
}

function makeLegend(series) {
  const leg = document.createElement("div");
  leg.className = "legend";
  for (const s of series) {
    const k = document.createElement("span");
    k.className = "key";
    const sw = document.createElement("span");
    sw.className = "swatch dot";
    sw.style.background = s.color;
    k.appendChild(sw);
    k.appendChild(document.createTextNode(s.name));
    leg.appendChild(k);
  }
  return leg;
}

/* ---------- 图 2：样本量柱状图 ---------- */
function renderSamplesChart() {
  const counties = state.data.estimates.counties;
  const W = 880, H = 240;
  const margin = { t: 10, r: 16, b: 34, l: 40 };
  const plotW = W - margin.l - margin.r, plotH = H - margin.t - margin.b;
  const nMax = Math.max(...counties.map((c) => c.n_sample), 10);
  const yMax = Math.ceil(nMax / 20) * 20;
  const y = (v) => margin.t + plotH - (v / yMax) * plotH;
  const band = plotW / counties.length;
  const barW = Math.min(24, band - 6);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "各县抽样样本量" });
  for (let v = 0; v <= yMax; v += 20) {
    svg.appendChild(svgEl("line", { x1: margin.l, y1: y(v), x2: margin.l + plotW, y2: y(v),
      stroke: "var(--gridline)", "stroke-width": 1 }));
    svg.appendChild(textEl(margin.l - 6, y(v) + 3, String(v),
      { "text-anchor": "end", "font-size": 10, fill: "var(--text-muted)" }));
  }
  svg.appendChild(svgEl("line", { x1: margin.l, y1: y(0), x2: margin.l + plotW, y2: y(0),
    stroke: "var(--baseline)", "stroke-width": 1 }));

  counties.forEach((c, i) => {
    const cx = margin.l + i * band + band / 2;
    if (c.n_sample > 0) {
      const h = y(0) - y(c.n_sample);
      // 4px 圆角数据端（顶部），底部直角贴基线
      const r = Math.min(4, h);
      const bar = svgEl("path", {
        d: `M ${cx - barW / 2} ${y(0)} L ${cx - barW / 2} ${y(c.n_sample) + r}
            Q ${cx - barW / 2} ${y(c.n_sample)} ${cx - barW / 2 + r} ${y(c.n_sample)}
            L ${cx + barW / 2 - r} ${y(c.n_sample)}
            Q ${cx + barW / 2} ${y(c.n_sample)} ${cx + barW / 2} ${y(c.n_sample) + r}
            L ${cx + barW / 2} ${y(0)} Z`,
        fill: "var(--series-1)",
      });
      svg.appendChild(bar);
    } else {
      svg.appendChild(textEl(cx, y(0) - 4, "无样本",
        { "text-anchor": "middle", "font-size": 9, fill: "var(--text-muted)" }));
    }
    svg.appendChild(textEl(cx, H - 20, c.id,
      { "text-anchor": "middle", "font-size": 9, fill: "var(--text-muted)" }));
    const hit = svgEl("rect", { x: margin.l + i * band, y: margin.t, width: band, height: plotH,
      fill: "transparent", tabindex: 0,
      "aria-label": `${c.name} 样本量 ${c.n_sample}` });
    hit.addEventListener("pointermove", (ev) => showTooltip(ev.clientX, ev.clientY, (tt) => {
      const t = document.createElement("div"); t.className = "tt-title"; t.textContent = c.name;
      tt.appendChild(t);
      ttRow(tt, "var(--series-1)", "样本量 n", String(c.n_sample));
      ttRow(tt, "var(--series-1)", "有效样本量", String(c.n_eff));
      ttRow(tt, "var(--series-1)", "设计效应 deff", c.deff == null ? "—" : String(c.deff));
      if (c.extrapolated) {
        const n = document.createElement("div"); n.className = "tt-note";
        n.textContent = "无样本：估计完全依赖模型外推";
        tt.appendChild(n);
      }
    }));
    hit.addEventListener("pointerleave", hideTooltip);
    hit.addEventListener("click", () => selectCounty(c.id));
    svg.appendChild(hit);
  });

  const host = $("#chart-samples");
  host.textContent = "";
  host.appendChild(svg);
}

/* ---------- 图 3：留出验证覆盖率 ---------- */
function renderCoverageChart() {
  const folds = state.validation.folds;
  const W = 880, H = 230;
  const margin = { t: 12, r: 16, b: 30, l: 44 };
  const plotW = W - margin.l - margin.r, plotH = H - margin.t - margin.b;
  const y = (v) => margin.t + plotH - v * plotH; // v ∈ [0,1]

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "各留出折的90%区间覆盖率" });
  for (let v = 0; v <= 1.001; v += 0.25) {
    svg.appendChild(svgEl("line", { x1: margin.l, y1: y(v), x2: margin.l + plotW, y2: y(v),
      stroke: "var(--gridline)", "stroke-width": 1 }));
    svg.appendChild(textEl(margin.l - 6, y(v) + 3, pct(v, 0),
      { "text-anchor": "end", "font-size": 10, fill: "var(--text-muted)" }));
  }
  // 90% 参考线
  svg.appendChild(svgEl("line", { x1: margin.l, y1: y(0.9), x2: margin.l + plotW, y2: y(0.9),
    stroke: "var(--text-muted)", "stroke-width": 2 }));
  svg.appendChild(textEl(margin.l + plotW, y(0.9) - 5, "名义 90%",
    { "text-anchor": "end", "font-size": 10, fill: "var(--text-muted)" }));

  const band = plotW / folds.length;
  const barW = Math.min(24, band / 3);
  const series = [
    { key: "coverage_extrapolated", name: "外推预测", color: "var(--series-1)" },
    { key: "coverage_shrunk", name: "收缩预测", color: "var(--series-2)" },
  ];
  folds.forEach((f, i) => {
    const cx = margin.l + i * band + band / 2;
    series.forEach((s, j) => {
      const v = f[s.key];
      const bx = cx + (j - 0.5) * (barW + 2) - barW / 2;
      const h = y(0) - y(v);
      const r = Math.min(4, h);
      svg.appendChild(svgEl("path", {
        d: `M ${bx} ${y(0)} L ${bx} ${y(v) + r} Q ${bx} ${y(v)} ${bx + r} ${y(v)}
            L ${bx + barW - r} ${y(v)} Q ${bx + barW} ${y(v)} ${bx + barW} ${y(v) + r}
            L ${bx + barW} ${y(0)} Z`,
        fill: s.color,
      }));
      const hit = svgEl("rect", { x: bx - 2, y: margin.t, width: barW + 4, height: plotH,
        fill: "transparent", tabindex: 0,
        "aria-label": `${f.region} ${s.name}覆盖率 ${pct(v, 0)}` });
      hit.addEventListener("pointermove", (ev) => showTooltip(ev.clientX, ev.clientY, (tt) => {
        const t = document.createElement("div"); t.className = "tt-title";
        t.textContent = `留出地区：${f.region}`;
        tt.appendChild(t);
        ttRow(tt, s.color, s.name + "覆盖率", pct(v, 0));
        ttRow(tt, "var(--text-muted)", "训练县数", String(f.n_train));
        ttRow(tt, "var(--text-muted)", "τ̂²", f.tau2.toFixed(3));
      }));
      hit.addEventListener("pointerleave", hideTooltip);
      svg.appendChild(hit);
    });
    svg.appendChild(textEl(cx, H - 12, f.region,
      { "text-anchor": "middle", "font-size": 11, fill: "var(--text-secondary)" }));
  });

  const host = $("#chart-coverage");
  host.textContent = "";
  host.appendChild(makeLegend(series));
  host.appendChild(svg);
}

/* ---------- 验证表 ---------- */
function renderValidationTable() {
  const v = state.validation;
  const host = $("#validation-table");
  host.textContent = "";
  const tbl = document.createElement("table");
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of ["留出地区", "训练县数", "τ̂²", "覆盖率(外推)", "覆盖率(收缩)",
                   "MAE(收缩)", "加权MAE(收缩)", "加权MAE(原始率)"]) {
    const th = document.createElement("th"); th.textContent = h; hr.appendChild(th);
  }
  thead.appendChild(hr);
  tbl.appendChild(thead);
  const tb = document.createElement("tbody");
  for (const f of v.folds) {
    const tr = document.createElement("tr");
    for (const val of [f.region, f.n_train, f.tau2.toFixed(3),
                       pct(f.coverage_extrapolated, 0), pct(f.coverage_shrunk, 0),
                       f.mae_shrunk.toFixed(4),
                       f.wmae_shrunk == null ? "—" : f.wmae_shrunk.toFixed(4),
                       f.wmae_raw == null ? "—" : f.wmae_raw.toFixed(4)]) {
      const td = document.createElement("td");
      td.className = "num"; td.textContent = val;
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  const ov = v.overall;
  const tr = document.createElement("tr");
  tr.style.fontWeight = "600";
  for (const val of ["整体", "—", "—", pct(ov.coverage_extrapolated, 0), pct(ov.coverage_shrunk, 0),
                     ov.mae_shrunk.toFixed(4), ov.wmae_shrunk.toFixed(4), ov.wmae_raw.toFixed(4)]) {
    const td = document.createElement("td");
    td.className = "num"; td.textContent = val;
    tr.appendChild(td);
  }
  tb.appendChild(tr);
  tbl.appendChild(tb);
  host.appendChild(tbl);
}

/* ---------- 明细表 ---------- */
function renderDataTable() {
  const e = state.data.estimates;
  const feasible = e.calibration.feasible;
  const host = $("#data-table");
  host.textContent = "";
  const tbl = document.createElement("table");
  const cols = ["县", "地区", "人口", "n", "n_eff", "deff", "原始率",
                "收缩率", "90%区间", ...(feasible ? ["校准后率", "90%区间"] : []),
                "收缩B", "确认数", "锁定", "模拟真值"];
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const h of cols) { const th = document.createElement("th"); th.textContent = h; hr.appendChild(th); }
  thead.appendChild(hr);
  tbl.appendChild(thead);
  const tb = document.createElement("tbody");
  for (const c of e.counties) {
    const tr = document.createElement("tr");
    if (c.id === state.selectedId) tr.classList.add("selected");
    const cells = [
      c.name + (c.extrapolated ? " ⚠" : ""), c.region, num(c.population),
      c.n_sample, c.n_eff, c.deff == null ? "—" : c.deff,
      pct(c.raw_rate), pct(c.shrunk_rate),
      `[${pct(c.shrunk_lo)}, ${pct(c.shrunk_hi)}]`,
      ...(feasible ? [pct(c.cal_rate), `[${pct(c.cal_lo)}, ${pct(c.cal_hi)}]`] : []),
      c.shrink_weight_B.toFixed(2),
      c.confirmed_count == null ? "—" : num(c.confirmed_count),
    ];
    for (const val of cells) {
      const td = document.createElement("td");
      td.className = "num"; td.textContent = val;
      tr.appendChild(td);
    }
    // 锁定复选框
    const tdLock = document.createElement("td");
    if (c.confirmed_count != null) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = state.locked.has(c.id);
      cb.setAttribute("aria-label", `锁定 ${c.name} 已确认值`);
      cb.addEventListener("click", (ev) => ev.stopPropagation());
      cb.addEventListener("change", () => {
        if (cb.checked) state.locked.add(c.id); else state.locked.delete(c.id);
        refresh();
      });
      tdLock.appendChild(cb);
    }
    tr.appendChild(tdLock);
    const tdTrue = document.createElement("td");
    tdTrue.className = "num"; tdTrue.textContent = pct(c.true_rate);
    tr.appendChild(tdTrue);
    tr.addEventListener("click", () => selectCounty(c.id));
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  host.appendChild(tbl);
}

/* ---------- 县区明细面板 ---------- */
function renderDetail() {
  const host = $("#detail-body");
  host.textContent = "";
  const c = state.data.estimates.counties.find((x) => x.id === state.selectedId);
  if (!c) {
    const p = document.createElement("p"); p.className = "muted";
    p.textContent = "点击图表或表格中的县查看详情。";
    host.appendChild(p);
    return;
  }
  const feasible = state.data.estimates.calibration.feasible;
  const cal = state.data.estimates.calibration;

  const h3 = document.createElement("h3");
  h3.textContent = `${c.name}（${c.id} · ${c.region}）`;
  host.appendChild(h3);

  const flags = document.createElement("div");
  flags.className = "detail-flags";
  const addFlag = (text, cls) => {
    const f = document.createElement("span");
    f.className = "flag " + cls; f.textContent = text;
    flags.appendChild(f);
  };
  if (c.extrapolated) addFlag("无样本 · 模型外推", "warn");
  if (c.n_sample > 0 && c.n_sample <= 3) addFlag(`小样本 n=${c.n_sample}`, "warn");
  if (c.locked) addFlag("已锁定确认值", "lock");
  if (c.clamped === "lower") addFlag("校准贴下界 0", "warn");
  if (c.clamped === "upper") addFlag("校准贴上界 1", "warn");
  host.appendChild(flags);

  const sec1 = document.createElement("div");
  sec1.className = "detail-section";
  sec1.innerHTML = "<h4>三种比率</h4>";
  host.appendChild(sec1);
  const kvs = [
    ["原始率（设计加权）", pct(c.raw_rate)],
    ["收缩率", `${pct(c.shrunk_rate)}  [${pct(c.shrunk_lo)}, ${pct(c.shrunk_hi)}]`],
    ["校准后率", feasible && c.cal_rate != null
      ? `${pct(c.cal_rate)}  [${pct(c.cal_lo)}, ${pct(c.cal_hi)}]` : "—（校准不可行）"],
    ["模拟真值（仅核查用）", pct(c.true_rate)],
    ["人口", num(c.population)],
    ["样本 n / 有效 n_eff", `${c.n_sample} / ${c.n_eff}`],
    ["设计效应 deff", c.deff == null ? "—" : c.deff],
  ];
  for (const [k, v] of kvs) {
    const row = document.createElement("div"); row.className = "kv";
    const kk = document.createElement("span"); kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("span"); vv.className = "v"; vv.textContent = v;
    row.append(kk, vv);
    sec1.appendChild(row);
  }

  // 借用信息分解
  const sec2 = document.createElement("div");
  sec2.className = "detail-section";
  const h4 = document.createElement("h4");
  h4.textContent = "借用了哪些信息";
  sec2.appendChild(h4);
  if (c.extrapolated) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "该县无抽样数据，估计 100% 来自模型：其他地区样本训练的系数 × 本县地区特征。";
    sec2.appendChild(p);
  }
  const bar = document.createElement("div");
  bar.className = "borrow-bar";
  const own = document.createElement("div");
  own.className = "seg-own"; own.style.width = pct(c.own_weight, 1);
  const mod = document.createElement("div");
  mod.className = "seg-model"; mod.style.width = pct(c.shrink_weight_B, 1);
  bar.append(own, mod);
  sec2.appendChild(bar);
  const bl = document.createElement("div");
  bl.className = "borrow-legend";
  const l1 = document.createElement("span"); l1.textContent = `本县数据 ${pct(c.own_weight, 0)}`;
  const l2 = document.createElement("span"); l2.textContent = `模型（其他地区+特征）${pct(c.shrink_weight_B, 0)}`;
  bl.append(l1, l2);
  sec2.appendChild(bl);

  const prior = document.createElement("div");
  prior.className = "kv";
  const pk = document.createElement("span"); pk.className = "k"; pk.textContent = "模型先验率 x'β̂";
  const pv = document.createElement("span"); pv.className = "v"; pv.textContent = pct(c.prior_rate);
  prior.append(pk, pv);
  sec2.appendChild(prior);
  for (const ct of c.covariate_contribs) {
    const row = document.createElement("div"); row.className = "kv";
    const kk = document.createElement("span"); kk.className = "k";
    kk.textContent = `${ct.name} ${ct.x.toFixed(2)} × β ${ct.beta.toFixed(2)}`;
    const vv = document.createElement("span"); vv.className = "v";
    vv.textContent = (ct.contrib >= 0 ? "+" : "") + ct.contrib.toFixed(3);
    row.append(kk, vv);
    sec2.appendChild(row);
  }
  host.appendChild(sec2);

  // 校准
  const sec3 = document.createElement("div");
  sec3.className = "detail-section";
  const h4b = document.createElement("h4");
  h4b.textContent = "总量校准";
  sec3.appendChild(h4b);
  const calRows = feasible
    ? [["校准位移（校准后 − 收缩）", c.cal_shift == null ? "—" : (c.cal_shift >= 0 ? "+" : "") + pct(c.cal_shift)],
       ["拉格朗日乘子 λ", cal.lambda.toExponential(3)],
       ["状态", c.locked ? "锁定（未参与调整）" : c.clamped ? `贴${c.clamped === "lower" ? "下" : "上"}界` : "自由调整"]]
    : [["状态", "校准不可行：全市总量与已确认锁定项冲突"]];
  for (const [k, v] of calRows) {
    const row = document.createElement("div"); row.className = "kv";
    const kk = document.createElement("span"); kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("span"); vv.className = "v"; vv.textContent = v;
    row.append(kk, vv);
    sec3.appendChild(row);
  }
  // 锁定开关
  if (c.confirmed_count != null) {
    const lab = document.createElement("label");
    lab.style.display = "flex";
    lab.style.gap = "6px";
    lab.style.marginTop = "8px";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.locked.has(c.id);
    cb.addEventListener("change", () => {
      if (cb.checked) state.locked.add(c.id); else state.locked.delete(c.id);
      refresh();
    });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(`锁定已确认值（${num(c.confirmed_count)}，率 ${pct(c.confirmed_count / c.population)}）`));
    sec3.appendChild(lab);
  }
  host.appendChild(sec3);
}

function selectCounty(id) {
  state.selectedId = id;
  renderDetail();
  renderRatesChart();
  renderDataTable();
}

/* ---------- 汇总渲染 ---------- */
function renderAll() {
  renderBanner();
  renderTiles();
  renderRatesChart();
  renderSamplesChart();
  renderCoverageChart();
  renderValidationTable();
  renderDataTable();
  renderDetail();
  const m = state.data.estimates.model;
  $("#model-info").textContent =
    `模型: logit(θ) = β'x + u, τ̂²=${m.tau2.toFixed(4)}, ` +
    m.feature_names.map((n, i) => `β[${n}]=${m.beta[i].toFixed(3)}`).join(" ");
}

/* ---------- 初始化 ---------- */
async function init() {
  // 主题
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const cur = root.getAttribute("data-theme");
    root.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  });

  const vResp = await fetch("/api/validation");
  state.validation = await vResp.json();

  // 先取一次默认状态以填充场景与默认锁定
  const first = await fetch("/api/state?scenario=baseline");
  const firstData = await first.json();
  state.locked = new Set(firstData.default_locked);
  const sel = $("#scenario");
  for (const [key, sc] of Object.entries(firstData.scenarios)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = `${sc.label}（T=${num(sc.city_total)}）`;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => { state.scenario = sel.value; refresh(); });

  $("#lock-all").addEventListener("click", () => {
    state.locked = new Set(state.data.default_locked);
    refresh();
  });
  $("#unlock-all").addEventListener("click", () => {
    state.locked = new Set();
    refresh();
  });

  state.data = firstData;
  $("#export-csv").href = "/api/export/estimates.csv?scenario=baseline&locked=" + [...state.locked].join(",");
  $("#export-json").href = "/api/export/summary.json?scenario=baseline&locked=" + [...state.locked].join(",");
  renderAll();
}

init();
