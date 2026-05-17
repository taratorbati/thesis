"use strict";
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Tara Torbati";
pres.title = "Modern Control Methods for Agricultural Irrigation";

// ─── PALETTE ────────────────────────────────────────────────
// Deep navy (dominant) + teal accent + off-white body
const C = {
  navy: "0D1B3E",   // dominant background
  teal: "0B9B8A",   // accent / charts
  teal2: "14B8A6",   // lighter teal
  ice: "C9E8E4",   // light teal tint for cards
  white: "FFFFFF",
  offwht: "F0F4F8",
  muted: "8EA8C3",   // muted label color
  red: "E05252",   // MPC highlight
  amber: "F59E0B",   // SAC highlight
  green: "22C55E",   // positive
  slate: "1E3A5F",   // card bg
  bodyTxt: "CBD5E1",   // body text on dark bg
  darkTxt: "1E293B",   // body text on light bg
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

// helper: section-header card
function sectionCard(slide, text, sub) {
  slide.background = { color: C.navy };
  // Big teal accent block left
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });
  slide.addText(text, { x: 0.4, y: 1.8, w: 9.2, h: 1.2, fontSize: 40, bold: true, color: C.white, fontFace: "Calibri" });
  if (sub) slide.addText(sub, { x: 0.4, y: 3.1, w: 9.2, h: 0.7, fontSize: 20, color: C.teal2, fontFace: "Calibri", italic: true });
}

// helper: content slide shell
function contentSlide(titleText, dark = true) {
  const slide = pres.addSlide();
  slide.background = { color: dark ? C.navy : C.offwht };
  const txt = dark ? C.white : C.darkTxt;
  // title
  slide.addText(titleText, {
    x: 0.45, y: 0.18, w: 9.1, h: 0.58,
    fontSize: 24, bold: true, color: txt, fontFace: "Calibri",
  });
  // thin teal rule under title
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 0.82, w: 9.1, h: 0.025,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  return slide;
}

// helper: metric card
function metricCard(slide, x, y, w, h, label, value, unit, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.slate }, line: { color: color || C.teal }, shadow: makeShadow() });
  slide.addText(value, { x, y: y + 0.08, w, h: h * 0.55, fontSize: 30, bold: true, color: color || C.teal, fontFace: "Calibri", align: "center", valign: "bottom" });
  slide.addText(unit, { x, y: y + h * 0.58, w, h: h * 0.2, fontSize: 11, color: C.bodyTxt, fontFace: "Calibri", align: "center" });
  slide.addText(label, { x, y: y + h * 0.78, w, h: h * 0.2, fontSize: 11, color: C.muted, fontFace: "Calibri", align: "center" });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 1 — TITLE
// ─────────────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };

  // Left accent stripe
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });

  // Decorative teal circle top-right
  slide.addShape(pres.shapes.OVAL, { x: 7.8, y: -0.9, w: 3.0, h: 3.0, fill: { color: C.teal, transparency: 80 }, line: { color: C.teal, transparency: 60 } });

  slide.addText("Modern Control Methods", {
    x: 0.55, y: 0.7, w: 9.0, h: 0.9, fontSize: 36, bold: true,
    color: C.white, fontFace: "Calibri",
  });
  slide.addText("for Agricultural Irrigation", {
    x: 0.55, y: 1.55, w: 9.0, h: 0.9, fontSize: 36, bold: true,
    color: C.teal2, fontFace: "Calibri",
  });
  slide.addText("MPC vs. Reinforcement Learning in Water-Constrained Crop Systems", {
    x: 0.55, y: 2.55, w: 9.0, h: 0.55, fontSize: 17, color: C.bodyTxt, fontFace: "Calibri", italic: true,
  });

  // Divider
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 3.22, w: 4.5, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  slide.addText([
    { text: "Tara Torbati", options: { bold: true, color: C.white } },
    { text: "   |   ITMO University, R4237c", options: { color: C.bodyTxt } },
  ], { x: 0.55, y: 3.38, w: 9.0, h: 0.38, fontSize: 14, fontFace: "Calibri" });

  slide.addText("Supervisor: Peregudin A. A.   |   MSc Defence, 2026", {
    x: 0.55, y: 3.78, w: 9.0, h: 0.38, fontSize: 13, color: C.muted, fontFace: "Calibri",
  });

  // Study site label
  slide.addText("Study site: Gilan Province, Iran  •  6 ha Hashemi rice field  •  38.298°N, 48.847°E", {
    x: 0.55, y: 4.9, w: 9.0, h: 0.38, fontSize: 11, color: C.muted, fontFace: "Calibri",
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 2 — OUTLINE
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Presentation Outline");
  const items = [
    ["01", "Motivation & Problem Statement"],
    ["02", "Simulation Environment (ABM)"],
    ["03", "Controller Designs — MPC & SAC"],
    ["04", "Experimental Setup"],
    ["05", "Results: Baselines & MPC"],
    ["06", "Results: SAC Training & Evaluation"],
    ["07", "Head-to-Head Comparison"],
    ["08", "Conclusions & Future Work"],
  ];
  const cols = [[0, 4], [1, 4]];
  items.forEach((item, i) => {
    const col = i < 4 ? 0 : 1;
    const row = i % 4;
    const x = col === 0 ? 0.45 : 5.2;
    const y = 1.05 + row * 1.08;
    // number circle
    slide.addShape(pres.shapes.OVAL, { x, y: y + 0.02, w: 0.46, h: 0.46, fill: { color: C.teal }, line: { color: C.teal } });
    slide.addText(item[0], { x, y: y + 0.02, w: 0.46, h: 0.46, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(item[1], { x: x + 0.56, y, w: 4.2, h: 0.5, fontSize: 15, color: C.white, fontFace: "Calibri", valign: "middle" });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 3 — MOTIVATION
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Motivation: The Anthropogenic Drought Crisis");
  // 3 stat cards
  const cards = [
    { val: "70%", unit: "of global freshwater", lbl: "consumed by agriculture" },
    { val: "40%", unit: "of world population", lbl: "under water stress by 2030" },
    { val: "238k", unit: "hectares of paddy", lbl: "in Gilan Province alone" },
  ];
  cards.forEach((c, i) => {
    metricCard(slide, 0.45 + i * 3.2, 1.05, 2.9, 1.55, c.lbl, c.val, c.unit, C.teal);
  });

  slide.addText("The Control Engineering Gap", {
    x: 0.45, y: 2.85, w: 9.1, h: 0.42, fontSize: 17, bold: true, color: C.teal2, fontFace: "Calibri",
  });
  const bullets = [
    "Traditional irrigation is reactive and open-loop — no forecast, no constraint awareness",
    "MPC can enforce strict seasonal water budgets but requires expensive online optimisation",
    "RL offers millisecond-latency decisions after training — but constraint satisfaction is not guaranteed",
    "This thesis rigorously benchmarks both on a high-fidelity ABM of a real Iranian rice field",
  ];
  slide.addText(
    bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < bullets.length - 1 } })),
    { x: 0.45, y: 3.3, w: 9.1, h: 2.0, fontSize: 14, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 6 }
  );
}

// ─────────────────────────────────────────────────────────────
// SLIDE 4 — ABM ENVIRONMENT
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("The Simulation Environment: Agent-Based Model");
  // Left column: bullets
  slide.addText([
    { text: "130 crop-soil agents on 10×13 DEM grid", options: { bullet: true, breakLine: true } },
    { text: "Elevation range 74–181 m (Talish Mountains)", options: { bullet: true, breakLine: true } },
    { text: "25-year NASA POWER climate dataset (2000–2025)", options: { bullet: true, breakLine: true } },
    { text: "Cascade water routing — eliminates 'bathtub' mass-balance errors", options: { bullet: true, breakLine: true } },
    { text: "Decoupled surface & subsurface hydrology (corrects original framework)", options: { bullet: true, breakLine: true } },
    { text: "Drought stress integrated into biomass eq. (FAO AquaCrop approach)", options: { bullet: true, breakLine: true } },
    { text: "Validated against NASA MERRA-2 GWETROOT   r = 0.74", options: { bullet: true } },
  ], { x: 0.45, y: 0.95, w: 5.2, h: 4.3, fontSize: 13, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 5 });

  // Right column: 5-state diagram
  const states = [
    { lbl: "x₁ Root moisture (mm)", col: C.teal },
    { lbl: "x₂ Thermal time (GDD)", col: C.teal2 },
    { lbl: "x₃ Maturation stress", col: C.amber },
    { lbl: "x₄ Biomass (g/m²)", col: C.green },
    { lbl: "x₅ Surface ponding (mm)", col: C.red },
  ];
  slide.addText("Per-agent state vector", { x: 5.85, y: 0.95, w: 3.8, h: 0.42, fontSize: 14, bold: true, color: C.teal2, fontFace: "Calibri" });
  states.forEach((s, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.85, y: 1.45 + i * 0.73, w: 3.8, h: 0.6, fill: { color: C.slate }, line: { color: s.col, width: 1.5 } });
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.85, y: 1.45 + i * 0.73, w: 0.08, h: 0.6, fill: { color: s.col }, line: { color: s.col } });
    slide.addText(s.lbl, { x: 6.05, y: 1.45 + i * 0.73, w: 3.6, h: 0.6, fontSize: 13, color: C.white, fontFace: "Calibri", valign: "middle" });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 5 — CROP PARAMETERS
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Crop & Scenario Parameterisation");
  // Two crop cards
  const riceProps = [
    ["Cultivar", "Hashemi rice"],
    ["Season", "DOY 141–233 (93 days)"],
    ["Kc", "1.15  |  Depletion p = 0.20"],
    ["Root depth", "400 mm  |  HI = 0.45"],
    ["Drought sens. θ₁₄", "0.80  (high)"],
    ["Maturity GDD", "1 250 °C·day"],
  ];
  const tobProps = [
    ["Cultivar", "Nicotiana tabacum"],
    ["Season", "DOY 146–249 (104 days)"],
    ["Kc", "0.90  |  Depletion p = 0.50"],
    ["Root depth", "700 mm  |  HI = 0.55"],
    ["Drought sens. θ₁₄", "0.60  (moderate)"],
    ["Maturity GDD", "1 200 °C·day"],
  ];

  [riceProps, tobProps].forEach((props, ci) => {
    const bx = 0.45 + ci * 4.9;
    const label = ci === 0 ? "Hashemi Rice" : "Tobacco";
    const col = ci === 0 ? C.teal : C.amber;
    slide.addShape(pres.shapes.RECTANGLE, { x: bx, y: 0.95, w: 4.5, h: 3.5, fill: { color: C.slate }, line: { color: col }, shadow: makeShadow() });
    slide.addShape(pres.shapes.RECTANGLE, { x: bx, y: 0.95, w: 4.5, h: 0.42, fill: { color: col }, line: { color: col } });
    slide.addText(label, { x: bx, y: 0.95, w: 4.5, h: 0.42, fontSize: 15, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    props.forEach((p, pi) => {
      slide.addText(p[0], { x: bx + 0.1, y: 1.44 + pi * 0.5, w: 1.8, h: 0.44, fontSize: 12, color: C.muted, fontFace: "Calibri", valign: "middle" });
      slide.addText(p[1], { x: bx + 1.95, y: 1.44 + pi * 0.5, w: 2.45, h: 0.44, fontSize: 12, color: C.white, fontFace: "Calibri", valign: "middle" });
    });
  });

  // Scenario strip bottom
  slide.addText("Test Scenarios (held-out years, never seen during training)", {
    x: 0.45, y: 4.6, w: 9.1, h: 0.38, fontSize: 14, bold: true, color: C.teal2, fontFace: "Calibri",
  });
  const scens = [
    { lbl: "Dry 2022", val: "39.7 mm", col: C.red },
    { lbl: "Moderate 2018", val: "108.8 mm", col: C.amber },
    { lbl: "Wet 2024", val: "176.8 mm", col: C.teal },
  ];
  scens.forEach((s, i) => {
    metricCard(slide, 0.45 + i * 3.2, 5.05, 2.9, 0.35, s.lbl, s.val, "seasonal rainfall", s.col);
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 6 — MPC DESIGN
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("MPC Design: Cost Function & Calibration");
  slide.addText("Five-term cost function   J  (recommended operating point α*)", {
    x: 0.45, y: 0.95, w: 9.1, h: 0.42, fontSize: 15, bold: true, color: C.teal2, fontFace: "Calibri",
  });

  const terms = [
    { sym: "α₁ = 1.0", lbl: "Terminal biomass reward", note: "anchor term", col: C.green },
    { sym: "α₂ = 0.016", lbl: "Water cost (Iranian domestic-base tariff)", note: "7 000 toman/m³", col: C.teal },
    { sym: "α₃ = 0.1", lbl: "Drought stress regulariser", note: "<1% yield sensitivity", col: C.amber },
    { sym: "α₅ = 0.005", lbl: "Actuator smoothing ΔU²", note: "tie-breaking", col: C.muted },
    { sym: "α₆ = 8.0", lbl: "FC-overshoot soft penalty  (x₁ > FC)", note: "eliminates waterlogging", col: C.red },
  ];
  terms.forEach((t, i) => {
    const y = 1.48 + i * 0.63;
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 0.08, h: 0.5, fill: { color: t.col }, line: { color: t.col } });
    slide.addText(t.sym, { x: 0.65, y, w: 1.5, h: 0.5, fontSize: 13, bold: true, color: t.col, fontFace: "Calibri", valign: "middle" });
    slide.addText(t.lbl, { x: 2.25, y, w: 4.5, h: 0.5, fontSize: 13, color: C.white, fontFace: "Calibri", valign: "middle" });
    slide.addText(t.note, { x: 6.85, y, w: 2.7, h: 0.5, fontSize: 11, color: C.muted, fontFace: "Calibri", valign: "middle", italic: true });
  });

  // Key stats
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 4.72, w: 9.1, h: 0.025, fill: { color: C.slate }, line: { color: C.slate } });
  slide.addText([
    { text: "Hp* = 8 days   ", options: { bold: true, color: C.teal } },
    { text: "|   3 120 decision vars, 2 081 constraints (per step)   ", options: { color: C.bodyTxt } },
    { text: "|   warm-start: 2.5× solver speedup   ", options: { color: C.bodyTxt } },
    { text: "|   solver: CasADi + IPOPT", options: { color: C.bodyTxt } },
  ], { x: 0.45, y: 4.8, w: 9.1, h: 0.5, fontSize: 12, fontFace: "Calibri" });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 7 — SAC DESIGN
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("SAC Design: CTDE Architecture");

  // Actor box
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 0.98, w: 4.3, h: 2.5, fill: { color: C.slate }, line: { color: C.teal }, shadow: makeShadow() });
  slide.addText("Actor  (deployed online)", { x: 0.45, y: 0.98, w: 4.3, h: 0.42, fontSize: 14, bold: true, color: C.teal, fontFace: "Calibri", align: "center", valign: "middle" });
  slide.addText([
    { text: "Input:  62-dim per-agent obs", options: { breakLine: true } },
    { text: "        (5 local + 57 global features)", options: { breakLine: true } },
    { text: "Arch:   62 → 128 → 128 → 1", options: { breakLine: true } },
    { text: "Shared weights across all 130 agents", options: { breakLine: true } },
    { text: "Output: 130 actions in ~1 ms (CPU)", options: {} },
  ], { x: 0.55, y: 1.45, w: 4.1, h: 1.95, fontSize: 12.5, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 3 });

  // Critic box
  slide.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 0.98, w: 4.25, h: 2.5, fill: { color: C.slate }, line: { color: C.amber }, shadow: makeShadow() });
  slide.addText("Critic  (training only)", { x: 5.3, y: 0.98, w: 4.25, h: 0.42, fontSize: 14, bold: true, color: C.amber, fontFace: "Calibri", align: "center", valign: "middle" });
  slide.addText([
    { text: "Input:  837-dim (707 obs + 130 actions)", options: { breakLine: true } },
    { text: "Arch:   837 → 256 → 256 → 1 (×2 twin)", options: { breakLine: true } },
    { text: "Clipped double-Q (reduces overestimation)", options: { breakLine: true } },
    { text: "Never used at deployment time", options: {} },
  ], { x: 5.4, y: 1.45, w: 4.05, h: 1.95, fontSize: 12.5, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 3 });

  // Arrow CTDE label
  slide.addText("CTDE", { x: 4.42, y: 2.05, w: 0.8, h: 0.45, fontSize: 12, bold: true, color: C.teal2, fontFace: "Calibri", align: "center" });
  slide.addShape(pres.shapes.LINE, { x: 4.75, y: 2.28, w: 0.55, h: 0, line: { color: C.teal2, width: 1.5 } });

  // Observation space breakdown
  slide.addText("707-dimensional observation space", {
    x: 0.45, y: 3.65, w: 9.1, h: 0.38, fontSize: 14, bold: true, color: C.teal2, fontFace: "Calibri",
  });
  const obsParts = [
    { lbl: "650-dim", sub: "Per-agent block\n5 states × 130 agents" },
    { lbl: "9-dim", sub: "Global scalars\n(day, budget, weather)" },
    { lbl: "48-dim", sub: "8-day forecast\n(6 vars × 8 days)" },
  ];
  obsParts.forEach((o, i) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.45 + i * 3.2, y: 4.1, w: 2.9, h: 1.22, fill: { color: C.slate }, line: { color: C.teal } });
    slide.addText(o.lbl, { x: 0.45 + i * 3.2, y: 4.1, w: 2.9, h: 0.55, fontSize: 22, bold: true, color: C.teal, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(o.sub, { x: 0.45 + i * 3.2, y: 4.65, w: 2.9, h: 0.62, fontSize: 11, color: C.muted, fontFace: "Calibri", align: "center" });
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 8 — EXPERIMENTAL SETUP
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Experimental Setup: 9-Cell Evaluation Grid");

  // Grid
  const scenarios = ["Dry (2022)", "Moderate (2018)", "Wet (2024)"];
  const budgets = ["100%  484 mm", "85%  411 mm", "70%  339 mm"];
  const colW = [2.7, 2.7, 2.7];
  const startX = 0.95, startY = 1.05;
  const rowH = 0.9;

  // Header row
  slide.addText("Scenario \\ Budget", { x: startX, y: startY, w: 1.6, h: 0.5, fontSize: 12, color: C.muted, fontFace: "Calibri", align: "center", valign: "middle" });
  budgets.forEach((b, j) => {
    slide.addShape(pres.shapes.RECTANGLE, { x: startX + 1.7 + j * 2.75, y: startY, w: 2.55, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
    slide.addText(b, { x: startX + 1.7 + j * 2.75, y: startY, w: 2.55, h: 0.5, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
  });

  // Data rows
  const scColors = [C.red, C.amber, C.teal];
  scenarios.forEach((s, i) => {
    const ry = startY + 0.55 + i * rowH;
    slide.addShape(pres.shapes.RECTANGLE, { x: startX, y: ry, w: 1.6, h: 0.8, fill: { color: scColors[i] }, line: { color: scColors[i] } });
    slide.addText(s, { x: startX, y: ry, w: 1.6, h: 0.8, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    budgets.forEach((_, j) => {
      slide.addShape(pres.shapes.RECTANGLE, { x: startX + 1.7 + j * 2.75, y: ry, w: 2.55, h: 0.8, fill: { color: C.slate }, line: { color: "2A4A6E" } });
      slide.addText("1×MPC  +  5×SAC\n+ 1 fixed-schedule", { x: startX + 1.7 + j * 2.75, y: ry, w: 2.55, h: 0.8, fontSize: 11, color: C.bodyTxt, fontFace: "Calibri", align: "center", valign: "middle" });
    });
  });

  slide.addText("Data split:  20 training  |  3 dev  |  3 test years   (stratified by rainfall tercile)", {
    x: 0.45, y: 4.5, w: 9.1, h: 0.38, fontSize: 13, color: C.teal2, fontFace: "Calibri", italic: true, align: "center",
  });
  slide.addText("Test years never touched during training or hyperparameter selection.", {
    x: 0.45, y: 4.9, w: 9.1, h: 0.38, fontSize: 12, color: C.muted, fontFace: "Calibri", align: "center",
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 9 — BASELINE RESULTS
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Baseline Results: No-Irrigation & Fixed Schedule");

  // Bar chart: yield comparison across scenarios
  const chartData = [
    { name: "No Irrigation", labels: ["Dry", "Moderate", "Wet"], values: [1462, 1478, 2243] },
    { name: "Fixed 100%", labels: ["Dry", "Moderate", "Wet"], values: [3607, 3302, 2790] },
    { name: "Fixed 85%", labels: ["Dry", "Moderate", "Wet"], values: [3394, 3089, 2901] },
    { name: "Fixed 70%", labels: ["Dry", "Moderate", "Wet"], values: [3118, 2841, 3024] },
  ];
  slide.addChart(pres.charts.BAR, chartData, {
    x: 0.45, y: 0.95, w: 6.0, h: 3.5, barDir: "col",
    chartColors: ["455A64", "0B9B8A", "14B8A6", "64B5A0"],
    chartArea: { fill: { color: C.slate } },
    catAxisLabelColor: C.muted,
    valAxisLabelColor: C.muted,
    valGridLine: { color: "2A4A6E", size: 0.5 },
    catGridLine: { style: "none" },
    showTitle: true, title: "Yield (kg/ha) by Scenario",
    titleColor: C.teal2, titleFontSize: 13,
    showLegend: true, legendPos: "b", legendFontColor: C.bodyTxt, legendFontSize: 11,
    showValue: false,
  });

  // Key findings box
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.65, y: 0.95, w: 2.9, h: 3.5, fill: { color: C.slate }, line: { color: C.teal } });
  slide.addText("Key Findings", { x: 6.65, y: 0.95, w: 2.9, h: 0.42, fontSize: 14, bold: true, color: C.teal, fontFace: "Calibri", align: "center", valign: "middle" });
  slide.addText([
    { text: "Wet 100% fixed: 83.7 waterlog-days", options: { bullet: true, breakLine: true, color: C.red } },
    { text: "Reducing budget to 70% in wet year INCREASES yield (2790 → 3024 kg/ha)", options: { bullet: true, breakLine: true, color: C.amber } },
    { text: "Fixed schedule cannot respond to dry spells", options: { bullet: true, breakLine: true, color: C.bodyTxt } },
    { text: "Open-loop scheduling is fundamentally inadequate", options: { bullet: true, color: C.bodyTxt } },
  ], { x: 6.75, y: 1.42, w: 2.7, h: 2.95, fontSize: 11.5, fontFace: "Calibri", paraSpaceAfter: 8 });

  // Waterlog callout
  slide.addText("⚠  Wet year: fixed-schedule 83.7 waterlog-days  vs.  MPC 19.0 days", {
    x: 0.45, y: 4.65, w: 9.1, h: 0.55, fontSize: 13, bold: true, color: C.amber, fontFace: "Calibri", align: "center",
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 10 — MPC RESULTS
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("MPC Results: Perfect Forecast, Hp = 8");

  // Table
  const hdr = { fill: { color: C.teal }, color: C.white, bold: true, fontSize: 12 };
  const tData = [
    [{ text: "Cell", options: hdr }, { text: "Yield (kg/ha)", options: hdr }, { text: "Water (mm)", options: hdr }, { text: "Wlog-d", options: hdr }, { text: "Wall-min", options: hdr }],
    ["Dry / 100%", "4 145", "469.1", "0.8", "44.1"],
    ["Dry / 85%", "4 069", "409.9", "0.3", "32.7"],
    ["Dry / 70%", "3 766", "338.7", "1.4", "22.0"],
    ["Moderate / 100%", "—", "—", "—", "—"],
    ["Moderate / 85%", "—", "—", "—", "—"],
    ["Moderate / 70%", "—", "—", "—", "—"],
    ["Wet / 100%", "3 759", "310.2", "19.0", "43.1"],
    ["Wet / 85%", "3 743", "308.3", "16.9", "48.3"],
    ["Wet / 70%", "3 754", "307.8", "18.2", "51.0"],
  ];
  // style moderate rows
  [4, 5, 6].forEach(r => {
    tData[r] = tData[r].map(c => ({ text: c, options: { color: C.muted, fontSize: 12 } }));
  });

  slide.addTable(tData, {
    x: 0.45, y: 0.95, w: 7.2, h: 4.3,
    colW: [1.8, 1.5, 1.3, 1.0, 1.6],
    border: { pt: 0.5, color: "2A4A6E" },
    fill: { color: C.slate },
    color: C.white, fontSize: 12, fontFace: "Calibri",
  });

  // Annotation
  slide.addText([
    { text: "Moderate Hp=8 runs pending Kaggle CPU completion", options: { color: C.muted, italic: true } },
  ], { x: 0.45, y: 5.3, w: 7.2, h: 0.3, fontSize: 11, fontFace: "Calibri" });

  // Side callouts
  const calls = [
    { val: "+14.9%", sub: "vs fixed-schedule\ndry/100%", col: C.green },
    { val: "+34.7%", sub: "vs fixed-schedule\nwet/100%", col: C.teal },
    { val: "64%", sub: "budget used\nwet year", col: C.amber },
  ];
  calls.forEach((c, i) => {
    metricCard(slide, 7.8, 0.95 + i * 1.55, 1.75, 1.35, c.sub, c.val, "", c.col);
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 11 — SAC TRAINING
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("SAC Training: Three-Phase Trajectory");

  // Phase timeline
  const phases = [
    { range: "0 – 25k steps", lbl: "Phase 1\nHealthy Convergence", note: "Critic loss 2713 → 0.67\nFull-season completion ✓", col: C.green },
    { range: "25k – 96k steps", lbl: "Phase 2\nProductive Plateau", note: "Best eval R = −0.37\nbest_model.zip captured ✓", col: C.teal },
    { range: "96k – 200k steps", lbl: "Phase 3\nQ-value Divergence", note: "Critic loss → 84 693\nent_coef spike 0.03 → 1.29", col: C.red },
  ];
  phases.forEach((p, i) => {
    const x = 0.45 + i * 3.2;
    slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.98, w: 3.0, h: 2.55, fill: { color: C.slate }, line: { color: p.col }, shadow: makeShadow() });
    slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.98, w: 3.0, h: 0.38, fill: { color: p.col }, line: { color: p.col } });
    slide.addText(p.range, { x, y: 0.98, w: 3.0, h: 0.38, fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(p.lbl, { x: x + 0.1, y: 1.42, w: 2.8, h: 0.72, fontSize: 13, bold: true, color: p.col, fontFace: "Calibri", align: "center" });
    slide.addText(p.note, { x: x + 0.1, y: 2.2, w: 2.8, h: 1.2, fontSize: 12, color: C.bodyTxt, fontFace: "Calibri", align: "center" });
  });

  slide.addText("Root Causes of Divergence (Identified)", {
    x: 0.45, y: 3.68, w: 9.1, h: 0.38, fontSize: 15, bold: true, color: C.teal2, fontFace: "Calibri",
  });
  const causes = [
    "c_term = 5: terminal bonus 300× larger than per-step rewards → Bellman error accumulation",
    "130-dim joint action: critic Q(s,a) overfits; double-Q overestimation accumulates",
    "target_entropy = −13: near-deterministic policy starves critic of action-space coverage",
    "No gradient clipping: large critic updates → larger errors → positive feedback loop",
  ];
  slide.addText(
    causes.map((c, i) => ({ text: c, options: { bullet: true, breakLine: i < causes.length - 1 } })),
    { x: 0.45, y: 4.1, w: 9.1, h: 1.3, fontSize: 13, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 4 }
  );
}

// ─────────────────────────────────────────────────────────────
// SLIDE 12 — SAC EVALUATION RESULTS
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("SAC Evaluation: Seed 0, Best Model (~step 96k)");

  // Bar chart yield comparison SAC vs MPC
  const chartData = [
    { name: "No Irrigation", labels: ["Dry", "Moderate", "Wet"], values: [1462, 1478, 2243] },
    { name: "Fixed Schedule", labels: ["Dry", "Moderate", "Wet"], values: [3607, 3302, 2790] },
    { name: "MPC Hp=8", labels: ["Dry", "Moderate", "Wet"], values: [4145, 0, 3759] },
    { name: "SAC", labels: ["Dry", "Moderate", "Wet"], values: [2477, 2376, 3145] },
  ];
  slide.addChart(pres.charts.BAR, chartData, {
    x: 0.45, y: 0.95, w: 6.0, h: 3.5, barDir: "col",
    chartColors: ["455A64", "0B9B8A", "E05252", "F59E0B"],
    chartArea: { fill: { color: C.slate } },
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted,
    valGridLine: { color: "2A4A6E", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: true, title: "Yield at 100% Budget (kg/ha)",
    titleColor: C.teal2, titleFontSize: 13,
    showLegend: true, legendPos: "b", legendFontColor: C.bodyTxt, legendFontSize: 11,
    showValue: false,
  });

  // Key SAC findings right
  slide.addShape(pres.shapes.RECTANGLE, { x: 6.65, y: 0.95, w: 2.9, h: 3.5, fill: { color: C.slate }, line: { color: C.amber } });
  slide.addText("SAC Findings", { x: 6.65, y: 0.95, w: 2.9, h: 0.42, fontSize: 14, bold: true, color: C.amber, fontFace: "Calibri", align: "center", valign: "middle" });
  slide.addText([
    { text: "39–70% budget utilised (never hits constraint)", options: { bullet: true, breakLine: true } },
    { text: "~2–3 mm/day constant schedule — not scenario-adaptive", options: { bullet: true, breakLine: true } },
    { text: "Applies MORE water in wetter years (opposite of optimal)", options: { bullet: true, breakLine: true } },
    { text: "Inference: ~1 ms  (25 000× faster than MPC)", options: { bullet: true, color: C.teal } },
  ], { x: 6.75, y: 1.42, w: 2.7, h: 2.95, fontSize: 11.5, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 8 });

  // Bottom note
  slide.addText([
    { text: "Wet-year outperformance over fixed-schedule is coincidental, not learned behaviour.", options: { italic: true, color: C.muted } },
  ], { x: 0.45, y: 4.65, w: 9.1, h: 0.38, fontSize: 12, fontFace: "Calibri" });

  slide.addText("Training pathology — not an inherent RL limitation. Known fixes specified for next run.", {
    x: 0.45, y: 5.05, w: 9.1, h: 0.35, fontSize: 12, bold: true, color: C.amber, fontFace: "Calibri",
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 13 — HEAD-TO-HEAD
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Head-to-Head: The Core Comparison");

  // Scatter-style bar chart showing gaps
  const gapData = [
    { name: "MPC vs Fixed", labels: ["Dry/100%", "Dry/85%", "Dry/70%", "Wet/100%", "Wet/85%", "Wet/70%"], values: [538, 675, 648, 969, 842, 730] },
    { name: "SAC vs Fixed", labels: ["Dry/100%", "Dry/85%", "Dry/70%", "Wet/100%", "Wet/85%", "Wet/70%"], values: [-1130, -910, -617, 355, 247, 131] },
  ];
  slide.addChart(pres.charts.BAR, gapData, {
    x: 0.45, y: 0.95, w: 9.1, h: 3.6, barDir: "col",
    chartColors: [C.teal, C.amber],
    chartArea: { fill: { color: C.slate } },
    catAxisLabelColor: C.muted, valAxisLabelColor: C.muted,
    valGridLine: { color: "2A4A6E", size: 0.5 }, catGridLine: { style: "none" },
    showTitle: true, title: "Yield Delta vs. Fixed Schedule (kg/ha)   [positive = better]",
    titleColor: C.teal2, titleFontSize: 13,
    showLegend: true, legendPos: "r", legendFontColor: C.bodyTxt, legendFontSize: 12,
  });

  slide.addText([
    { text: "MPC Hp=8:   ", options: { bold: true, color: C.teal } },
    { text: "+492 to +969 kg/ha  over fixed schedule in all completed cells", options: { color: C.bodyTxt } },
  ], { x: 0.45, y: 4.65, w: 9.1, h: 0.35, fontSize: 13, fontFace: "Calibri" });
  slide.addText([
    { text: "SAC seed 0: ", options: { bold: true, color: C.amber } },
    { text: "−617 to −1130 kg/ha (dry/mod)  |  +131 to +355 kg/ha (wet, coincidental)", options: { color: C.bodyTxt } },
  ], { x: 0.45, y: 5.05, w: 9.1, h: 0.35, fontSize: 13, fontFace: "Calibri" });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 14 — LATENCY COMPARISON
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Computational Cost: The Deployment Argument");

  // Big number callouts
  metricCard(slide, 0.45, 1.0, 4.2, 1.9, "MPC mean per-decision", "25.9 min", "wall-clock per day", C.red);
  metricCard(slide, 5.35, 1.0, 4.2, 1.9, "SAC mean per-decision", "~1 ms", "CPU inference", C.teal);

  // Speedup callout
  slide.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: 2.05, w: 3.4, h: 0.9, fill: { color: C.teal }, line: { color: C.teal } });
  slide.addText("25 000× faster", { x: 3.3, y: 2.05, w: 3.4, h: 0.9, fontSize: 22, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });

  slide.addText([
    { text: "MPC:", options: { bold: true, color: C.red } },
    { text: " 22–51 wall-min per season  |  worst-case single call: 274 s  |  requires high-performance CPU", options: { color: C.bodyTxt } },
    { text: "\nSAC:", options: { bold: true, color: C.teal, breakLine: true } },
    { text: " 0.3 s total season  |  cold start 41 ms  |  compatible with edge MCU at 3 ms budget", options: { color: C.bodyTxt } },
    { text: "\nEven a corrected SAC with MPC-level yield would provide a genuinely advantageous combination.", options: { italic: true, color: C.teal2, breakLine: true } },
  ], { x: 0.45, y: 3.2, w: 9.1, h: 2.1, fontSize: 13.5, fontFace: "Calibri", paraSpaceAfter: 6 });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 15 — RECOMMENDED FIXES
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Recommended Fixes for Next SAC Training Run");

  const fixes = [
    {
      pri: "P1", lbl: "Remove terminal bonus",
      desc: "Set c_term = 0.  Per-step reward integrates to equivalent total biomass signal, distributed smoothly. Eliminates the 300× reward-concentration that causes Bellman error amplification.",
      col: C.red,
    },
    {
      pri: "P2", lbl: "Add gradient clipping",
      desc: "max_grad_norm = 1.0 in SAC constructor.  Breaks the positive feedback loop: large critic loss → large gradients → larger errors.",
      col: C.amber,
    },
    {
      pri: "P3", lbl: "Learning rate decay",
      desc: "Linear 3×10⁻⁴ → 5×10⁻⁵ over 500k steps.  Prevents noisy late-training updates from destabilising a near-converged Q-function.",
      col: C.teal,
    },
    {
      pri: "P4", lbl: "Relax target entropy",
      desc: "−13 → −20 to −25 (−0.15 to −0.2 × dim).  Improves action-space coverage in late training, reducing Q-overestimation at distribution edges.",
      col: C.teal2,
    },
  ];

  fixes.forEach((f, i) => {
    const y = 0.98 + i * 1.1;
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 0.65, h: 0.9, fill: { color: f.col }, line: { color: f.col } });
    slide.addText(f.pri, { x: 0.45, y, w: 0.65, h: 0.9, fontSize: 16, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(f.lbl, { x: 1.2, y: y + 0.04, w: 2.5, h: 0.42, fontSize: 14, bold: true, color: f.col, fontFace: "Calibri", valign: "middle" });
    slide.addText(f.desc, { x: 1.2, y: y + 0.46, w: 8.35, h: 0.42, fontSize: 12, color: C.bodyTxt, fontFace: "Calibri", valign: "top" });
  });

  slide.addText("Expected run plan: 5 seeds × 500k steps on Kaggle T4  (~2 h per seed)", {
    x: 0.45, y: 5.32, w: 9.1, h: 0.3, fontSize: 12, color: C.muted, fontFace: "Calibri", italic: true,
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 16 — CONCLUSIONS
// ─────────────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });

  slide.addText("Conclusions", { x: 0.4, y: 0.22, w: 9.1, h: 0.65, fontSize: 32, bold: true, color: C.white, fontFace: "Calibri" });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 0.92, w: 9.1, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  const concl = [
    { icon: "✓", col: C.teal, txt: "H1 CONFIRMED — MPC outperforms fixed-schedule by 492–969 kg/ha across all completed cells; waterlogging virtually eliminated." },
    { icon: "◎", col: C.amber, txt: "H2 PARTIALLY CONFIRMED — SAC underperforms MPC due to a training pathology (Q-value divergence), not an inherent RL limitation. Known fixes identified." },
    { icon: "⏳", col: C.muted, txt: "H3 PENDING — Forecast-noise evaluation runs not yet complete. Framework fully operational." },
    { icon: "⚡", col: C.teal2, txt: "KEY ENGINEERING RESULT: SAC delivers 25 000× faster inference (1 ms vs 25.9 min per day), enabling real-time edge deployment. The computational argument for RL holds even before agronomic parity is reached." },
  ];
  concl.forEach((c, i) => {
    slide.addShape(pres.shapes.OVAL, { x: 0.4, y: 1.12 + i * 1.02, w: 0.45, h: 0.45, fill: { color: c.col }, line: { color: c.col } });
    slide.addText(c.icon, { x: 0.4, y: 1.12 + i * 1.02, w: 0.45, h: 0.45, fontSize: 14, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(c.txt, { x: 1.0, y: 1.1 + i * 1.02, w: 8.55, h: 0.82, fontSize: 13, color: C.bodyTxt, fontFace: "Calibri", valign: "middle" });
  });

  slide.addText("All code, experiments, and results reproducible from the public GitHub repository.", {
    x: 0.4, y: 5.22, w: 9.1, h: 0.3, fontSize: 11, color: C.muted, fontFace: "Calibri", italic: true,
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 17 — FUTURE WORK
// ─────────────────────────────────────────────────────────────
{
  const slide = contentSlide("Future Work");

  const fw = [
    { lbl: "Immediate", col: C.teal, items: ["Complete moderate-year MPC Hp=8 (3 cells)", "5-seed SAC training with P1–P4 fixes", "Noisy forecast evaluation: MPC & SAC"] },
    { lbl: "Short-term", col: C.amber, items: ["Axis 3 parametric robustness (±10% soil params)", "SAC training under noisy-forecast domain randomisation", "Economic ROI analysis for smallholder deployment in Gilan"] },
    { lbl: "Long-term", col: C.red, items: ["Sim-to-real transfer on 360 Rain irrigation robot", "Multi-field / multi-crop generalisation", "Safe RL with hard constraint guarantees (Lagrangian methods)"] },
  ];

  fw.forEach((f, i) => {
    const x = 0.45 + i * 3.2;
    slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.98, w: 3.0, h: 4.3, fill: { color: C.slate }, line: { color: f.col }, shadow: makeShadow() });
    slide.addShape(pres.shapes.RECTANGLE, { x, y: 0.98, w: 3.0, h: 0.45, fill: { color: f.col }, line: { color: f.col } });
    slide.addText(f.lbl, { x, y: 0.98, w: 3.0, h: 0.45, fontSize: 15, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    slide.addText(
      f.items.map((item, j) => ({ text: item, options: { bullet: true, breakLine: j < f.items.length - 1 } })),
      { x: x + 0.12, y: 1.5, w: 2.76, h: 3.7, fontSize: 13, color: C.bodyTxt, fontFace: "Calibri", paraSpaceAfter: 12 }
    );
  });
}

// ─────────────────────────────────────────────────────────────
// SLIDE 18 — THANK YOU / Q&A
// ─────────────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(pres.shapes.OVAL, { x: -1.0, y: 3.5, w: 4.5, h: 4.5, fill: { color: C.teal, transparency: 85 }, line: { color: C.teal, transparency: 70 } });
  slide.addShape(pres.shapes.OVAL, { x: 7.8, y: -1.2, w: 4.0, h: 4.0, fill: { color: C.teal, transparency: 80 }, line: { color: C.teal, transparency: 65 } });

  slide.addText("Thank You", { x: 0.5, y: 0.9, w: 9.0, h: 1.1, fontSize: 52, bold: true, color: C.white, fontFace: "Calibri", align: "center" });
  slide.addText("Questions & Discussion", { x: 0.5, y: 2.1, w: 9.0, h: 0.65, fontSize: 24, color: C.teal2, fontFace: "Calibri", align: "center", italic: true });

  slide.addShape(pres.shapes.RECTANGLE, { x: 2.5, y: 2.9, w: 5.0, h: 0.025, fill: { color: C.teal }, line: { color: C.teal } });

  slide.addText("Tara Torbati   |   ITMO University R4237c   |   Supervisor: Peregudin A. A.", {
    x: 0.5, y: 3.1, w: 9.0, h: 0.42, fontSize: 14, color: C.bodyTxt, fontFace: "Calibri", align: "center",
  });
  slide.addText("github.com/taratorbati/thesis", {
    x: 0.5, y: 3.6, w: 9.0, h: 0.38, fontSize: 14, color: C.teal, fontFace: "Calibri", align: "center",
  });

  // Quick-reference stat strip
  const stats = [
    { v: "130", u: "ABM agents" }, { v: "26 yr", u: "climate data" }, { v: "9 cells", u: "test grid" },
    { v: "Hp=8", u: "MPC horizon" }, { v: "200k", u: "SAC steps" }, { v: "25 000×", u: "SAC speedup" },
  ];
  stats.forEach((s, i) => {
    slide.addText(s.v, { x: 0.45 + i * 1.62, y: 4.45, w: 1.52, h: 0.52, fontSize: 20, bold: true, color: C.teal, fontFace: "Calibri", align: "center" });
    slide.addText(s.u, { x: 0.45 + i * 1.62, y: 4.97, w: 1.52, h: 0.28, fontSize: 10, color: C.muted, fontFace: "Calibri", align: "center" });
  });
}

// pres.writeFile({ fileName: "/mnt/user-data/outputs/thesis_defense.pptx" })
pres.writeFile({ fileName: "./thesis_defense.pptx" })
  .then(() => console.log("OK"))
  .catch(e => { console.error(e); process.exit(1); });