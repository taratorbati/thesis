"use strict";
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// ─── Icon helpers ────────────────────────────────────────────────────────────
async function iconPng(IconComponent, color, size = 256) {
  const { FaSeedling, FaTint, FaBrain, FaChartBar, FaFlask,
          FaCheckCircle, FaExclamationTriangle, FaArrowRight,
          FaDatabase, FaMicroscope, FaCog, FaBolt, FaGlobe,
          FaLeaf, FaSun, FaCloudRain } = require("react-icons/fa");
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ─── Palette (Deep teal + charcoal + gold accent) ────────────────────────────
const C = {
  dark:      "0D2137",   // slide backgrounds (dark)
  teal:      "0D9488",   // primary accent
  tealLight: "14B8A6",   // secondary teal
  gold:      "F59E0B",   // highlight / alert
  white:     "FFFFFF",
  offWhite:  "E8F4F3",
  lightGray: "CBD5E1",
  midGray:   "64748B",
  cardBg:    "132035",   // slightly lighter dark for cards
  cardBg2:   "0F2D40",   // alt card
  green:     "22C55E",
  red:       "EF4444",
  amber:     "F97316",
};

const FONT_H  = "Calibri";
const FONT_B  = "Calibri";

// ─── Re-usable helpers ───────────────────────────────────────────────────────
function darkBg(slide) { slide.background = { color: C.dark }; }
function lightBg(slide) { slide.background = { color: "F0F9F8" }; }

function slideTitle(slide, text, y = 0.28) {
  slide.addText(text, {
    x: 0.45, y, w: 9.1, h: 0.55,
    fontSize: 28, bold: true, color: C.white, fontFace: FONT_H,
    align: "left", valign: "middle", margin: 0,
  });
  // Teal accent bar (left edge)
  slide.addShape("rect", { x: 0.1, y, w: 0.06, h: 0.55, fill: { color: C.teal }, line: { color: C.teal } });
}

function lightTitle(slide, text, y = 0.28) {
  slide.addText(text, {
    x: 0.45, y, w: 9.1, h: 0.55,
    fontSize: 28, bold: true, color: C.dark, fontFace: FONT_H,
    align: "left", valign: "middle", margin: 0,
  });
  slide.addShape("rect", { x: 0.1, y, w: 0.06, h: 0.55, fill: { color: C.teal }, line: { color: C.teal } });
}

function card(slide, x, y, w, h, color = C.cardBg) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.25 },
    line: { color: C.teal, width: 0.5 },
  });
}

function cardLight(slide, x, y, w, h) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color: "FFFFFF" },
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.10 },
    line: { color: C.teal, width: 0.8 },
  });
}

function statBlock(slide, x, y, value, label, valueColor = C.teal) {
  card(slide, x, y, 2.7, 1.1);
  slide.addText(value, {
    x: x + 0.1, y: y + 0.08, w: 2.5, h: 0.55,
    fontSize: 30, bold: true, color: valueColor, fontFace: FONT_H,
    align: "center", valign: "middle", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.1, y: y + 0.62, w: 2.5, h: 0.38,
    fontSize: 11, color: C.lightGray, fontFace: FONT_B,
    align: "center", valign: "top", margin: 0,
  });
}

// ─── Build presentation ──────────────────────────────────────────────────────
async function build() {
  const { FaSeedling, FaTint, FaBrain, FaChartBar, FaFlask,
          FaCheckCircle, FaExclamationTriangle, FaArrowRight,
          FaDatabase, FaMicroscope, FaCog, FaBolt, FaGlobe,
          FaLeaf, FaSun, FaCloudRain } = require("react-icons/fa");

  // Pre-render icons
  const icons = {
    seed:    await iconPng(FaSeedling, C.teal),
    water:   await iconPng(FaTint, C.tealLight),
    brain:   await iconPng(FaBrain, C.gold),
    chart:   await iconPng(FaChartBar, C.teal),
    flask:   await iconPng(FaFlask, C.tealLight),
    check:   await iconPng(FaCheckCircle, C.green),
    warn:    await iconPng(FaExclamationTriangle, C.gold),
    arrow:   await iconPng(FaArrowRight, C.white),
    db:      await iconPng(FaDatabase, C.teal),
    micro:   await iconPng(FaMicroscope, C.tealLight),
    cog:     await iconPng(FaCog, C.gold),
    bolt:    await iconPng(FaBolt, C.gold),
    globe:   await iconPng(FaGlobe, C.tealLight),
    leaf:    await iconPng(FaLeaf, C.green),
    sun:     await iconPng(FaSun, C.gold),
    rain:    await iconPng(FaCloudRain, C.tealLight),
    checkW:  await iconPng(FaCheckCircle, C.white),
    brainW:  await iconPng(FaBrain, C.white),
    chartW:  await iconPng(FaChartBar, C.white),
  };

  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Tara Torbati";
  pres.title = "Modern Control Methods for Agricultural Irrigation";

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 1 — Title
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);

    // Large teal stripe left
    s.addShape("rect", { x: 0, y: 0, w: 0.35, h: 5.625, fill: { color: C.teal }, line: { color: C.teal } });

    // Decorative teal rectangle top-right
    s.addShape("rect", { x: 6.5, y: 0, w: 3.5, h: 1.6, fill: { color: C.cardBg }, line: { color: C.teal, width: 1 } });

    // Main title
    s.addText("Modern Control Methods", {
      x: 0.6, y: 1.05, w: 9.0, h: 0.75,
      fontSize: 38, bold: true, color: C.white, fontFace: FONT_H,
      align: "left", margin: 0,
    });
    s.addText("for Agricultural Irrigation", {
      x: 0.6, y: 1.8, w: 9.0, h: 0.65,
      fontSize: 34, bold: false, color: C.tealLight, fontFace: FONT_H,
      align: "left", margin: 0,
    });

    // Subtitle bar
    s.addShape("rect", { x: 0.6, y: 2.62, w: 8.5, h: 0.04, fill: { color: C.midGray }, line: { color: C.midGray } });

    s.addText("MPC vs. Reinforcement Learning in Water-Constrained Rice Cultivation", {
      x: 0.6, y: 2.75, w: 8.6, h: 0.4,
      fontSize: 15, color: C.lightGray, fontFace: FONT_B,
      align: "left", italic: true, margin: 0,
    });

    // Metadata
    s.addText([
      { text: "Student: ", options: { bold: true, color: C.lightGray } },
      { text: "Tara Torbati, gr. R4237c", options: { color: C.white } },
    ], { x: 0.6, y: 3.55, w: 5.5, h: 0.32, fontSize: 13, fontFace: FONT_B, margin: 0 });

    s.addText([
      { text: "Supervisor: ", options: { bold: true, color: C.lightGray } },
      { text: "Alexey A. Peregudin", options: { color: C.white } },
    ], { x: 0.6, y: 3.88, w: 5.5, h: 0.32, fontSize: 13, fontFace: FONT_B, margin: 0 });

    s.addText([
      { text: "Institution: ", options: { bold: true, color: C.lightGray } },
      { text: "ITMO University  •  Saint Petersburg  •  MSc Defence 2026", options: { color: C.white } },
    ], { x: 0.6, y: 4.21, w: 8.5, h: 0.32, fontSize: 13, fontFace: FONT_B, margin: 0 });

    s.addText([
      { text: "Study site: ", options: { bold: true, color: C.lightGray } },
      { text: "Gilan Province, Iran  •  6 ha Hashemi rice field  •  38.298°N, 48.847°E", options: { color: C.tealLight } },
    ], { x: 0.6, y: 4.6, w: 8.5, h: 0.32, fontSize: 12, fontFace: FONT_B, margin: 0 });

    // Icon row
    for (const [ic, xpos] of [[icons.water, 7.1], [icons.seed, 7.85], [icons.globe, 8.6]]) {
      s.addImage({ data: ic, x: xpos, y: 0.35, w: 0.45, h: 0.45 });
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 2 — Motivation
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Motivation: The Water Crisis in Irrigated Agriculture");

    // Three stat cards
    const stats = [
      ["70%", "of global freshwater\nconsumed by agriculture", C.teal],
      ["40%", "of world population\nfacing water stress by 2030", C.gold],
      ["238K ha", "of paddy in\nGilan Province, Iran", C.tealLight],
    ];
    stats.forEach(([val, lbl, col], i) => {
      const x = 0.45 + i * 3.1;
      card(s, x, 1.1, 2.85, 1.5, C.cardBg);
      s.addShape("rect", { x, y: 1.1, w: 2.85, h: 0.06, fill: { color: col }, line: { color: col } });
      s.addText(val, { x: x+0.1, y: 1.25, w: 2.65, h: 0.6, fontSize: 32, bold: true, color: col, fontFace: FONT_H, align: "center", margin: 0 });
      s.addText(lbl, { x: x+0.1, y: 1.85, w: 2.65, h: 0.65, fontSize: 12, color: C.lightGray, fontFace: FONT_B, align: "center", margin: 0 });
    });

    // Control gap box
    card(s, 0.45, 2.9, 9.1, 2.45);
    s.addText("The Control Engineering Gap", { x: 0.65, y: 2.98, w: 8.7, h: 0.36, fontSize: 16, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });

    const gaps = [
      [icons.warn, "Traditional irrigation is open-loop — no forecast, no constraint awareness, reactive and wasteful"],
      [icons.cog,  "MPC enforces strict seasonal budgets and uses weather forecasts but requires expensive daily optimisation"],
      [icons.bolt, "RL offers millisecond-latency decisions after offline training — but constraint satisfaction is not guaranteed"],
      [icons.leaf, "This thesis rigorously benchmarks both on a high-fidelity ABM of a real Iranian rice field"],
    ];
    gaps.forEach(([ic, txt], i) => {
      const yy = 3.42 + i * 0.47;
      s.addImage({ data: ic, x: 0.6, y: yy + 0.03, w: 0.28, h: 0.28 });
      s.addText(txt, { x: 0.98, y: yy, w: 8.4, h: 0.4, fontSize: 12.5, color: C.offWhite, fontFace: FONT_B, margin: 0, valign: "middle" });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 3 — Research Goals
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Research Goals and Tasks");

    const tasks = [
      ["01", "Literature Review", "Modern control methods in agricultural irrigation; agent-based simulation frameworks", icons.micro],
      ["02", "Field & Data Selection", "6 ha Hashemi paddy in Gilan Province; NASA POWER reanalysis (2000–2025); SRTM elevation", icons.db],
      ["03", "Mathematical Modelling", "Agent-based model: 130 crop-soil agents, cascade water routing, FAO AquaCrop biomass dynamics", icons.flask],
      ["04", "Controller Design", "MPC (CasADi+IPOPT) and SAC-RL (CTDE-VDN) — both satisfying a seasonal water-budget constraint", icons.cog],
      ["05", "Performance Analysis", "9-cell evaluation grid (3 scenarios × 3 budgets); perfect and noisy forecast modes; yield and WUE", icons.chart],
    ];

    tasks.forEach(([num, title, body, ic], i) => {
      const yy = 1.05 + i * 0.9;
      card(s, 0.35, yy, 9.3, 0.78, C.cardBg);
      // number badge
      s.addShape("rect", { x: 0.35, y: yy, w: 0.55, h: 0.78, fill: { color: C.teal }, line: { color: C.teal } });
      s.addText(num, { x: 0.35, y: yy, w: 0.55, h: 0.78, fontSize: 16, bold: true, color: C.white, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });
      s.addImage({ data: ic, x: 1.05, y: yy + 0.23, w: 0.3, h: 0.3 });
      s.addText(title, { x: 1.45, y: yy + 0.05, w: 7.9, h: 0.32, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
      s.addText(body,  { x: 1.45, y: yy + 0.38, w: 7.9, h: 0.35, fontSize: 11.5, color: C.lightGray, fontFace: FONT_B, margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 4 — Literature
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Baseline Literature");

    card(s, 0.35, 1.1, 9.3, 1.55);
    s.addShape("rect", { x: 0.35, y: 1.1, w: 0.08, h: 1.55, fill: { color: C.gold }, line: { color: C.gold } });
    s.addText("Baseline Framework", { x: 0.6, y: 1.14, w: 8.9, h: 0.38, fontSize: 15, bold: true, color: C.gold, fontFace: FONT_H, margin: 0 });
    s.addText(
      "J. López-Jiménez, N. Quijano, L. Dewasme, A. Vande Wouwer — \"Agent-based model predictive control of\nsoil-crop irrigation with topographical information\" — Control Engineering Practice, vol. 150, 2024.",
      { x: 0.6, y: 1.52, w: 8.9, h: 0.5, fontSize: 12.5, color: C.lightGray, fontFace: FONT_B, margin: 0 }
    );
    s.addText("Key contributions of this thesis vs. baseline:", { x: 0.35, y: 2.82, w: 9.3, h: 0.35, fontSize: 14, bold: true, color: C.white, fontFace: FONT_H, margin: 0 });

    const contribs = [
      [icons.check, "Adds a realistic seasonal water-budget constraint reflecting Iranian water policy (484 mm/season)"],
      [icons.brain, "Introduces a Soft Actor-Critic RL controller operating on the same agent-based environment"],
      [icons.micro, "Characterises the training-stability limit of SAC on long-horizon agricultural MDPs"],
    ];
    contribs.forEach(([ic, txt], i) => {
      const yy = 3.22 + i * 0.62;
      card(s, 0.35, yy, 9.3, 0.52);
      s.addImage({ data: ic, x: 0.5, y: yy + 0.12, w: 0.28, h: 0.28 });
      s.addText(txt, { x: 0.88, y: yy + 0.08, w: 8.65, h: 0.38, fontSize: 12.5, color: C.offWhite, fontFace: FONT_B, margin: 0, valign: "middle" });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 5 — Field Description
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Field Description and Data");

    // Left column — field details
    card(s, 0.35, 1.05, 4.5, 4.2);
    s.addText("Study Site", { x: 0.55, y: 1.12, w: 4.1, h: 0.35, fontSize: 15, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const fieldLines = [
      ["Location", "38.298°N, 48.847°E"],
      ["Area", "6 hectares — 130 paddy plots"],
      ["Grid", "10 × 13 agents (~46 m² per agent)"],
      ["Elevation", "74–181 m (Talish Mountains)"],
      ["Routing", "Cascade — upslope surplus flows to downslope"],
      ["Crop", "Hashemi rice — season DOY 141–233 (93 days)"],
      ["Kc", "1.15 | depletion p = 0.20 | HI = 0.45"],
    ];
    fieldLines.forEach(([k, v], i) => {
      const yy = 1.55 + i * 0.5;
      s.addText(k + ":", { x: 0.55, y: yy, w: 1.5, h: 0.42, fontSize: 12, bold: true, color: C.lightGray, fontFace: FONT_B, margin: 0 });
      s.addText(v, { x: 2.0, y: yy, w: 2.65, h: 0.42, fontSize: 12, color: C.white, fontFace: FONT_B, margin: 0 });
    });

    // Right column — climate data
    card(s, 5.05, 1.05, 4.6, 2.0);
    s.addText("Climate Data (NASA POWER)", { x: 5.2, y: 1.12, w: 4.3, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const climVars = [
      "T2M mean/max/min (°C)",
      "PRECTOTCORR — corrected precipitation (mm/day)",
      "ALLSKY_SFC_SW_DWN — shortwave radiation",
      "RH2M (%), WS2M (m/s), surface pressure",
      "ET₀ via Penman-Monteith equation",
    ];
    climVars.forEach((v, i) => {
      s.addText([{ text: "•  " + v }], { x: 5.2, y: 1.55 + i * 0.28, w: 4.35, h: 0.28, fontSize: 11.5, color: C.lightGray, fontFace: FONT_B, margin: 0 });
    });

    card(s, 5.05, 3.2, 4.6, 2.05);
    s.addText("Data Split", { x: 5.2, y: 3.27, w: 4.3, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const split = [
      ["Training", "20 years  |  max rainfall 82 mm"],
      ["Dev", "3 years (2002, 2016, 2023)  |  max 88 mm"],
      ["Test Dry", "2022 — 40 mm  ←  in-distribution"],
      ["Test Mod", "2018 — 109 mm  ←  near edge"],
      ["Test Wet", "2024 — 177 mm  ←  OOD (+115%)"],
    ];
    split.forEach(([k, v], i) => {
      const yy = 3.65 + i * 0.32;
      const col = k.startsWith("Test Wet") ? C.gold : k.startsWith("Test") ? C.tealLight : C.lightGray;
      s.addText(k + ":", { x: 5.2, y: yy, w: 1.4, h: 0.3, fontSize: 11.5, bold: true, color: col, fontFace: FONT_B, margin: 0 });
      s.addText(v, { x: 6.55, y: yy, w: 2.95, h: 0.3, fontSize: 11.5, color: C.white, fontFace: FONT_B, margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 6 — ABM Dynamics
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Agent-Based Model: 130 Crop-Soil Agents");

    // Left: state vector table
    card(s, 0.35, 1.05, 4.6, 4.22);
    s.addText("Per-Agent State Vector", { x: 0.55, y: 1.13, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });

    const states = [
      ["x₁", "Root-zone soil moisture (mm)", "WP=80, FC=140, SAT≈220"],
      ["x₂", "Accumulated GDD (°C·day)", "Maturity at 1250 °C·day"],
      ["x₃", "Maturation/heat stress (–)", "Modulates growth rate"],
      ["x₄", "Biomass accumulation (g/m²)", "HI=0.45 → yield kg/ha"],
      ["x₅", "Surface ponding (mm)", "Cascade to downslope agents"],
    ];
    states.forEach(([sym, desc, note], i) => {
      const yy = 1.56 + i * 0.72;
      s.addShape("rect", { x: 0.45, y: yy, w: 0.55, h: 0.52, fill: { color: C.teal }, line: { color: C.teal } });
      s.addText(sym, { x: 0.45, y: yy, w: 0.55, h: 0.52, fontSize: 15, bold: true, color: C.white, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });
      s.addText(desc, { x: 1.08, y: yy, w: 3.8, h: 0.26, fontSize: 12, bold: true, color: C.white, fontFace: FONT_B, margin: 0 });
      s.addText(note, { x: 1.08, y: yy + 0.26, w: 3.8, h: 0.26, fontSize: 10.5, color: C.lightGray, fontFace: FONT_B, margin: 0, italic: true });
    });

    // Right: equations + cascade description
    card(s, 5.15, 1.05, 4.5, 2.1);
    s.addText("Soil-Water Balance", { x: 5.3, y: 1.13, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const eqLines = [
      "Δx₁ = rain + u  −  ET  −  percolation  −  runoff",
      "Percolation ∝ max(x₁ − FC, 0)  (drains overshoot)",
      "Runoff = surface surplus after infiltration capacity",
      "I_max = I_max_cap − x₁  (caps infiltration → SAT≈220 mm)",
    ];
    eqLines.forEach((l, i) => {
      s.addText(l, { x: 5.3, y: 1.58 + i * 0.36, w: 4.22, h: 0.34, fontSize: 11.5, color: C.offWhite, fontFace: "Consolas", margin: 0 });
    });

    card(s, 5.15, 3.28, 4.5, 2.0);
    s.addText("Biomass & Cascade", { x: 5.3, y: 3.35, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const bioLines = [
      "Δx₄ = g_base × h₂(x₁) × h₇(Tmax)",
      "h₂: water-stress multiplier (FAO AquaCrop)",
      "h₇: heat-stress multiplier",
      "Cascade: W_surf = x₅ + rain + u",
      "Surplus flows to downslope neighbours (DEM-derived)",
    ];
    bioLines.forEach((l, i) => {
      s.addText(l, { x: 5.3, y: 3.78 + i * 0.29, w: 4.22, h: 0.28, fontSize: 11, color: C.offWhite, fontFace: "Consolas", margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 7 — MPC Design
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "MPC Controller: Formulation");

    // Cost function card
    card(s, 0.35, 1.08, 9.3, 1.85);
    s.addText("Five-Term Normalised Cost Function   J(α*)", { x: 0.55, y: 1.14, w: 8.9, h: 0.35, fontSize: 15, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const terms = [
      ["α₁ = 1.0", "Terminal biomass reward", "anchor term"],
      ["α₂ = 0.016", "Water cost", "calibrated to ~7 000 toman/m³ (domestic-base tariff)"],
      ["α₃ = 0.1", "Drought stress regulariser", "penalises x₁ < ST"],
      ["α₅ = 0.005", "Control-rate regulariser  ‖Δu‖²", "tie-breaking smoothness"],
      ["α₆ = 8.0", "FC-overshoot penalty  [max(x₁−FC,0)/FC]²", "eliminates waterlogging"],
    ];
    terms.forEach(([w, name, note], i) => {
      const x = 0.55 + i * 1.84;
      s.addShape("rect", { x, y: 1.57, w: 0.92, h: 0.28, fill: { color: C.teal }, line: { color: C.teal } });
      s.addText(w, { x, y: 1.57, w: 0.92, h: 0.28, fontSize: 11, bold: true, color: C.white, fontFace: "Consolas", align: "center", valign: "middle", margin: 0 });
      s.addText(name, { x, y: 1.86, w: 1.8, h: 0.28, fontSize: 9.5, bold: true, color: C.white, fontFace: FONT_B, margin: 0 });
      s.addText(note, { x, y: 2.14, w: 1.8, h: 0.68, fontSize: 9, color: C.lightGray, fontFace: FONT_B, margin: 0, italic: true });
    });

    // Constraints + solver
    card(s, 0.35, 3.08, 4.5, 2.18);
    s.addText("Constraints & Solver", { x: 0.55, y: 3.15, w: 4.1, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const cons = [
      "0 ≤ uₙ(k) ≤ 12 mm/day  (per-agent daily bound)",
      "Σₖ Σₙ uₙ(k) ≤ budget_total  (seasonal constraint)",
      "ABM dynamics as equality constraints",
      "Multiple-shooting transcription",
      "Solver: CasADi + IPOPT (MA27 linear solver)",
      "Warm-start from previous solve → 2.5× speedup",
    ];
    cons.forEach((l, i) => {
      s.addText([{ text: "•  " + l }], { x: 0.55, y: 3.57 + i * 0.28, w: 4.22, h: 0.27, fontSize: 11.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    // Spec stats
    card(s, 5.1, 3.08, 4.55, 2.18);
    s.addText("Solver Specifications", { x: 5.28, y: 3.15, w: 4.2, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const specs = [
      ["Horizon Hp", "8 days (sensitivity-sweep optimum)"],
      ["Decision vars", "~3 120 per solve (130 × 8 × 3)"],
      ["Constraints", "~2 080 per solve"],
      ["Mean solve time", "~25 min/day"],
      ["Worst-case", "274 s (single day)"],
      ["α-sweep configs", "33 configurations, 7 groups"],
    ];
    specs.forEach(([k, v], i) => {
      const yy = 3.58 + i * 0.28;
      s.addText(k + ":", { x: 5.28, y: yy, w: 1.85, h: 0.27, fontSize: 11.5, bold: true, color: C.lightGray, fontFace: FONT_B, margin: 0 });
      s.addText(v, { x: 7.08, y: yy, w: 2.42, h: 0.27, fontSize: 11.5, color: C.white, fontFace: FONT_B, margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 8 — MPC Sensitivity
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "MPC: α-Weight Calibration & Sensitivity");

    // α₆ sweep chart (bar) — waterlog days vs α₆
    const a6vals = [0, 1, 2, 4, 6, 8, 10];
    const wlogDays = [83.7, 68, 48, 28, 18, 6.4, 5.8];
    const yieldVals = [2790, 2980, 3100, 3350, 3580, 3717, 3715];

    s.addChart(pres.charts.BAR, [
      { name: "Waterlog days/agent", labels: a6vals.map(String), values: wlogDays },
    ], {
      x: 0.35, y: 1.05, w: 4.6, h: 2.9, barDir: "col",
      showTitle: true, title: "Waterlog Days vs α₆ (wet/100%)",
      titleFontSize: 12, titleColor: C.white,
      chartColors: [C.red],
      chartArea: { fill: { color: C.cardBg } },
      catAxisLabelColor: C.lightGray, valAxisLabelColor: C.lightGray,
      valGridLine: { color: "334155", size: 0.5 }, catGridLine: { style: "none" },
      showValue: true, dataLabelColor: C.white, dataLabelFontSize: 9,
      catAxisTitle: "α₆ value", valAxisTitle: "Waterlog days/agent",
      catAxisTitleColor: C.lightGray, valAxisTitleColor: C.lightGray,
    });

    s.addChart(pres.charts.LINE, [
      { name: "Yield (kg/ha)", labels: a6vals.map(String), values: yieldVals },
    ], {
      x: 5.1, y: 1.05, w: 4.55, h: 2.9,
      showTitle: true, title: "Yield vs α₆ (wet/100%)",
      titleFontSize: 12, titleColor: C.white,
      chartColors: [C.teal],
      lineSize: 2,
      chartArea: { fill: { color: C.cardBg } },
      catAxisLabelColor: C.lightGray, valAxisLabelColor: C.lightGray,
      valGridLine: { color: "334155", size: 0.5 }, catGridLine: { style: "none" },
      showValue: true, dataLabelColor: C.white, dataLabelFontSize: 9,
      catAxisTitle: "α₆ value", valAxisTitle: "Yield (kg/ha)",
      catAxisTitleColor: C.lightGray, valAxisTitleColor: C.lightGray,
    });

    // α* selection rationale
    card(s, 0.35, 4.1, 9.3, 1.2);
    s.addImage({ data: icons.check, x: 0.5, y: 4.22, w: 0.3, h: 0.3 });
    s.addText("Recommended operating point: α₆* = 8.0", { x: 0.9, y: 4.15, w: 8.5, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    s.addText(
      "α₆ = 8 reduces waterlog days from 83.7 to 6.4 with no yield penalty vs. α₆ = 10 (saturation point). " +
      "Factorial analysis (Group F) confirmed α₄ is subsumed by α₆ and disabled (α₄ = 0).",
      { x: 0.9, y: 4.53, w: 8.5, h: 0.68, fontSize: 12, color: C.offWhite, fontFace: FONT_B, margin: 0 }
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 9 — SAC Architecture
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "SAC-RL Controller: CTDE-VDN Architecture");

    // Left: Actor
    card(s, 0.35, 1.08, 4.4, 2.45);
    s.addShape("rect", { x: 0.35, y: 1.08, w: 4.4, h: 0.35, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText("Shared Actor  (deployed at inference)", { x: 0.45, y: 1.08, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.white, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });
    s.addImage({ data: icons.brain, x: 0.55, y: 1.55, w: 0.45, h: 0.45 });
    const actorLines = [
      "Centralized Training / Decentralized Execution",
      "Same network applied once per agent (N=130)",
      "Input: 65-dim (8 local features + 57 global)",
      "Output: μ(s), σ(s) → squashed Gaussian action",
      "Architecture: 65 → 128 → 128 → 1",
      "Inference time: ~1 ms for all 130 agents",
    ];
    actorLines.forEach((l, i) => {
      s.addText([{ text: "•  " + l }], { x: 0.55, y: 1.55 + (i + 1) * 0.3, w: 4.1, h: 0.29, fontSize: 11.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    // Right: Critic
    card(s, 5.0, 1.08, 4.65, 2.45);
    s.addShape("rect", { x: 5.0, y: 1.08, w: 4.65, h: 0.35, fill: { color: C.gold }, line: { color: C.gold } });
    s.addText("Factorized Critic (VDN)  —  training only", { x: 5.1, y: 1.08, w: 4.45, h: 0.35, fontSize: 13, bold: true, color: C.dark, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });
    s.addImage({ data: icons.cog, x: 5.18, y: 1.55, w: 0.45, h: 0.45 });
    s.addText("Q_total = Σₙ Q_local(sₙ, g, aₙ)", { x: 5.75, y: 1.58, w: 3.82, h: 0.38, fontSize: 13, bold: true, color: C.gold, fontFace: "Consolas", margin: 0 });
    const critLines = [
      "Twin-Q (clipped double-Q) → anti-overestimation",
      "Local input: 66-dim (8 local + 57 global + 1 action)",
      "Architecture: 66 → 256 → 256 → 1  (per agent)",
      "VDN valid: reward is additive in agent contributions",
      "Sum-aggregation: bias_ratio = N = 130 (constant)",
    ];
    critLines.forEach((l, i) => {
      s.addText([{ text: "•  " + l }], { x: 5.18, y: 2.0 + i * 0.3, w: 4.35, h: 0.29, fontSize: 11.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    // Observation space
    card(s, 0.35, 3.68, 9.3, 1.58);
    s.addText("Observation Space: 1097-dimensional", { x: 0.55, y: 3.75, w: 8.9, h: 0.35, fontSize: 14, bold: true, color: C.teal, fontFace: FONT_H, margin: 0 });
    const obsBlocks = [
      ["Per-agent block", "8 × 130 = 1040 dim", "x₁_norm, x₅_norm, x₄_norm, x₃, elev, Nr, Nr_int, N_up", C.teal],
      ["Global scalars", "9 dim", "day fraction, budget frac/total, burn rate, rain, ETc, h₂, h₇, g_base", C.tealLight],
      ["Forecast block", "48 dim", "8-day forecast of: rain, ETc, radiation, h₂, h₇, g_base", C.gold],
    ];
    obsBlocks.forEach(([label, dim, desc, col], i) => {
      const x = 0.45 + i * 3.07;
      s.addShape("rect", { x, y: 4.17, w: 3.0, h: 0.28, fill: { color: col }, line: { color: col } });
      s.addText(label + "  " + dim, { x, y: 4.17, w: 3.0, h: 0.28, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });
      s.addText(desc, { x, y: 4.47, w: 3.0, h: 0.65, fontSize: 9.5, color: C.lightGray, fontFace: FONT_B, align: "center", margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 10 — SAC Training Hyperparameters
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "SAC Training Configuration");

    // Hyperparameter table
    const params = [
      ["Total timesteps", "250 000", "20 training years × 93 days × randomised budget"],
      ["Replay buffer size", "250 000", "Off-policy; uniform replay"],
      ["Batch size", "256", "Mini-batch for each gradient step"],
      ["Discount γ", "0.99", "Effective horizon ≈ 93 steps (matches season length)"],
      ["Soft update τ", "0.005", "Slow-moving target networks for stability"],
      ["Learning rate", "3×10⁻⁴ → 5×10⁻⁵", "Linear decay across 250k steps"],
      ["Entropy coef α", "0.05 (fixed)", "Auto-tune disabled — caused entropy-spike explosion in v2.4"],
      ["Gradient clipping", "max_norm = 1.0", "Prevents runaway critic updates"],
      ["Actor hidden", "[128, 128]", "Two ReLU layers"],
      ["Critic hidden", "[256, 256]", "Larger critic for complex VDN Q-function"],
      ["EvalCallback freq", "every 25 000 steps", "Saves best_model.zip when dev-set reward improves"],
    ];

    card(s, 0.35, 1.05, 9.3, 4.3);
    s.addShape("rect", { x: 0.35, y: 1.05, w: 9.3, h: 0.35, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText("Hyperparameter", { x: 0.45, y: 1.05, w: 2.1, h: 0.35, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_B, valign: "middle", margin: 0 });
    s.addText("Value", { x: 2.55, y: 1.05, w: 2.0, h: 0.35, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_B, valign: "middle", margin: 0 });
    s.addText("Rationale", { x: 4.55, y: 1.05, w: 4.95, h: 0.35, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_B, valign: "middle", margin: 0 });

    params.forEach(([k, v, r], i) => {
      const yy = 1.43 + i * 0.35;
      const bg = i % 2 === 0 ? C.cardBg : C.cardBg2;
      s.addShape("rect", { x: 0.35, y: yy, w: 9.3, h: 0.35, fill: { color: bg }, line: { color: bg } });
      s.addText(k, { x: 0.45, y: yy, w: 2.1, h: 0.35, fontSize: 10.5, bold: true, color: C.lightGray, fontFace: FONT_B, valign: "middle", margin: 0 });
      s.addText(v, { x: 2.55, y: yy, w: 2.0, h: 0.35, fontSize: 10.5, color: C.tealLight, fontFace: "Consolas", valign: "middle", margin: 0 });
      s.addText(r, { x: 4.55, y: yy, w: 4.95, h: 0.35, fontSize: 10, color: C.offWhite, fontFace: FONT_B, valign: "middle", margin: 0 });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 11 — Training Dynamics
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "SAC Training Trajectory — v2.7 (Best Configuration)");

    // Critic loss chart (log scale — approximate with large values)
    const steps = [1, 10, 25, 50, 75, 100, 125, 150, 156, 173, 200, 250, 300, 500];
    const closs = [2713, 40, 0.79, 1.06, 0.85, 1.07, 2.1, 11.8, 50, 252, 577000, 4624, 50000, 6.87e6];
    const aloss = [13.1, -180, -284, -385, -392, -395, -391, -353, -330, 20, 16900, 1380, 80000, 1.16e5];

    // Use log values for display
    const clog = closs.map(v => Math.log10(Math.max(v, 0.01)));
    const alabslog = aloss.map(v => Math.log10(Math.abs(v) + 0.01));

    s.addChart(pres.charts.LINE, [
      { name: "log₁₀(critic_loss)", labels: steps.map(String), values: clog },
    ], {
      x: 0.35, y: 1.05, w: 4.5, h: 2.6,
      showTitle: true, title: "Critic Loss (log scale)",
      titleFontSize: 12, titleColor: C.white,
      chartColors: [C.red],
      lineSize: 2,
      chartArea: { fill: { color: C.cardBg } },
      catAxisLabelColor: C.lightGray, valAxisLabelColor: C.lightGray,
      valGridLine: { color: "334155", size: 0.5 }, catGridLine: { style: "none" },
      catAxisTitle: "Step (×1000)", valAxisTitle: "log₁₀(loss)",
      catAxisTitleColor: C.lightGray, valAxisTitleColor: C.lightGray,
    });

    s.addChart(pres.charts.LINE, [
      { name: "log₁₀(|actor_loss|)", labels: steps.map(String), values: alabslog },
    ], {
      x: 5.1, y: 1.05, w: 4.55, h: 2.6,
      showTitle: true, title: "Actor Loss magnitude (log scale)",
      titleFontSize: 12, titleColor: C.white,
      chartColors: [C.gold],
      lineSize: 2,
      chartArea: { fill: { color: C.cardBg } },
      catAxisLabelColor: C.lightGray, valAxisLabelColor: C.lightGray,
      valGridLine: { color: "334155", size: 0.5 }, catGridLine: { style: "none" },
      catAxisTitle: "Step (×1000)", valAxisTitle: "log₁₀(|loss|)",
      catAxisTitleColor: C.lightGray, valAxisTitleColor: C.lightGray,
    });

    // Phase labels
    const phases = [
      ["Phase 1: 0–25k\nHealthy convergence", C.teal],
      ["Phase 2: 25–200k\nProductive plateau\nbest_model captured at 200k", C.tealLight],
      ["Phase 3: 200k+\nDeadly-triad cascade\ncritic_loss → 6.87×10¹²", C.red],
    ];
    phases.forEach(([txt, col], i) => {
      card(s, 0.35 + i * 3.1, 3.83, 2.85, 0.8, C.cardBg);
      s.addShape("rect", { x: 0.35 + i * 3.1, y: 3.83, w: 2.85, h: 0.06, fill: { color: col }, line: { color: col } });
      s.addText(txt, { x: 0.45 + i * 3.1, y: 3.92, w: 2.68, h: 0.65, fontSize: 10.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    // Diagnosis
    card(s, 0.35, 4.72, 9.3, 0.68);
    s.addImage({ data: icons.micro, x: 0.48, y: 4.83, w: 0.28, h: 0.28 });
    s.addText(
      "Deadly-Triad diagnosis (van Hasselt et al. 2018): function approximation × bootstrapping × off-policy replay. " +
      "With γ=0.99 and 93-day episodes, small Q-estimate errors accumulate ~100× through bootstrap propagation. " +
      "Once |Q| >> α·log π  (with α=0.05), the entropy brake fails and the cascade ignites.",
      { x: 0.85, y: 4.75, w: 8.7, h: 0.6, fontSize: 10.5, color: C.offWhite, fontFace: FONT_B, margin: 0 }
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 12 — Results table
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Comparative Results: Yield (kg/ha) — Perfect Forecast");

    const rows = [
      ["Dry (2022)",    "100%", "4122", "4163", "3607", "+1.0%",  false],
      ["Dry (2022)",    "85%",  "4068", "4101", "3619", "+0.8%",  false],
      ["Dry (2022)",    "70%",  "3771", "3766", "3439", "−0.1%",  false],
      ["Moderate (2018)","100%","3698", "3730", "3302", "+0.9%",  false],
      ["Moderate (2018)","85%", "3694", "3737", "3309", "+1.2%",  false],
      ["Moderate (2018)","70%", "3598", "3589", "3184", "−0.3%",  false],
      ["Wet (2024)",    "100%", "3717", "3434", "2790", "−7.6%",  true],
      ["Wet (2024)",    "85%",  "3722", "3432", "3144", "−7.8%",  true],
      ["Wet (2024)",    "70%",  "3727", "3492", "3428", "−6.3%",  true],
    ];

    const colW = [1.8, 0.7, 1.1, 1.1, 1.1, 1.1];
    const headers = ["Scenario", "Budget", "MPC Hp=3", "SAC v2.7", "Fixed Schedule", "SAC vs MPC"];
    const hRow = headers.map((h, i) => ({
      text: h,
      options: { fill: { color: C.teal }, color: C.white, bold: true, fontSize: 11, align: "center" }
    }));

    const tableData = [hRow];
    rows.forEach(([sc, bud, mpc, sac, fix, gap, isWet]) => {
      const r = [
        { text: sc,  options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: isWet ? C.gold : C.white, bold: isWet, fontSize: 10.5 } },
        { text: bud, options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: C.lightGray, fontSize: 10.5, align: "center" } },
        { text: mpc, options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: C.tealLight, bold: true, fontSize: 11, align: "center" } },
        { text: sac, options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: C.white, bold: true, fontSize: 11, align: "center" } },
        { text: fix, options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: C.lightGray, fontSize: 10.5, align: "center" } },
        { text: gap, options: { fill: { color: isWet ? "1A2F45" : C.cardBg }, color: isWet ? C.red : C.green, bold: true, fontSize: 11, align: "center" } },
      ];
      tableData.push(r);
    });

    s.addTable(tableData, {
      x: 0.35, y: 1.05, w: 9.3, h: 4.0,
      colW,
      border: { pt: 0.5, color: "334155" },
    });

    // Key observation
    card(s, 0.35, 5.15, 9.3, 0.32);
    s.addImage({ data: icons.check, x: 0.5, y: 5.18, w: 0.22, h: 0.22 });
    s.addText(
      "SAC matches MPC within ±1.2% across all 6 in-distribution cells.  " +
      "The 6.3–7.8% gap appears only in the held-out 2024 wet scenario (OOD).",
      { x: 0.8, y: 5.17, w: 8.7, h: 0.3, fontSize: 11, bold: true, color: C.tealLight, fontFace: FONT_H, margin: 0, valign: "middle" }
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 13 — Yield bar chart
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Results: Yield per Cell — SAC vs MPC");

    const cells = ["Dry/100", "Dry/85", "Dry/70", "Mod/100", "Mod/85", "Mod/70", "Wet/100", "Wet/85", "Wet/70"];
    const mpcY =  [4122, 4068, 3771, 3698, 3694, 3598, 3717, 3722, 3727];
    const sacY =  [4163, 4101, 3766, 3730, 3737, 3589, 3434, 3432, 3492];
    const fixY =  [3607, 3619, 3439, 3302, 3309, 3184, 2790, 3144, 3428];

    s.addChart(pres.charts.BAR, [
      { name: "MPC Hp=3",     labels: cells, values: mpcY },
      { name: "SAC v2.7",     labels: cells, values: sacY },
      { name: "Fixed schedule", labels: cells, values: fixY },
    ], {
      x: 0.35, y: 1.05, w: 9.3, h: 4.25, barDir: "col", barGrouping: "clustered",
      chartColors: [C.teal, C.gold, C.midGray],
      chartArea: { fill: { color: C.cardBg } },
      catAxisLabelColor: C.lightGray, valAxisLabelColor: C.lightGray,
      valGridLine: { color: "334155", size: 0.5 }, catGridLine: { style: "none" },
      showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: C.white,
      catAxisTitle: "Scenario / Budget", valAxisTitle: "Yield (kg/ha)",
      catAxisTitleColor: C.lightGray, valAxisTitleColor: C.lightGray,
      valAxisMinVal: 2400,
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 14 — Diagnostics
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Diagnosing the Wet-Year Gap");

    // Finding 1: OOD
    card(s, 0.35, 1.08, 2.95, 4.2);
    s.addShape("rect", { x: 0.35, y: 1.08, w: 2.95, h: 0.35, fill: { color: C.red }, line: { color: C.red } });
    s.addText("① OOD Test Year", { x: 0.42, y: 1.08, w: 2.82, h: 0.35, fontSize: 12, bold: true, color: C.white, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });
    const oodLines = [
      ["Training max", "82 mm (2013)"],
      ["Dev set max", "88 mm (2023)"],
      ["Test wet (2024)", "177 mm"],
      ["Excess", "+115% above training P100"],
    ];
    oodLines.forEach(([k, v], i) => {
      const col = i === 3 ? C.gold : C.lightGray;
      const vcol = i === 3 ? C.gold : C.white;
      s.addText(k + ":", { x: 0.48, y: 1.56 + i * 0.52, w: 1.3, h: 0.45, fontSize: 11, bold: true, color: col, fontFace: FONT_B, margin: 0 });
      s.addText(v, { x: 1.78, y: 1.56 + i * 0.52, w: 1.35, h: 0.45, fontSize: 11, bold: i === 3, color: vcol, fontFace: FONT_B, margin: 0 });
    });
    s.addShape("rect", { x: 0.48, y: 3.72, w: 2.7, h: 0.04, fill: { color: C.midGray }, line: { color: C.midGray } });
    s.addText("MPC has no training distribution — it optimises against actual weather each day. SAC must generalise from training data.", { x: 0.48, y: 3.78, w: 2.72, h: 1.3, fontSize: 10.5, color: C.lightGray, fontFace: FONT_B, margin: 0, italic: true });

    // Finding 2: checkpoint sweep
    card(s, 3.48, 1.08, 3.1, 4.2);
    s.addShape("rect", { x: 3.48, y: 1.08, w: 3.1, h: 0.35, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText("② Only step-200k is budget-aware", { x: 3.55, y: 1.08, w: 2.96, h: 0.35, fontSize: 11.5, bold: true, color: C.white, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });

    const ckptData = [
      ["50k", "2949", "484 (full)", false],
      ["100k", "2817", "484 (full)", false],
      ["150k", "2870", "484 (full)", false],
      ["200k", "3434", "414 ✓", true],
      ["250k", "2867", "484 (full)", false],
      ["300k", "2511", "484 (full)", false],
    ];
    s.addShape("rect", { x: 3.52, y: 1.52, w: 3.02, h: 0.3, fill: { color: C.cardBg }, line: { color: C.teal } });
    s.addText("Step", { x: 3.55, y: 1.52, w: 0.75, h: 0.3, fontSize: 10, bold: true, color: C.teal, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });
    s.addText("Yield", { x: 4.3, y: 1.52, w: 0.85, h: 0.3, fontSize: 10, bold: true, color: C.teal, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });
    s.addText("Water", { x: 5.15, y: 1.52, w: 1.25, h: 0.3, fontSize: 10, bold: true, color: C.teal, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });

    ckptData.forEach(([step, yld, water, hi], i) => {
      const yy = 1.85 + i * 0.49;
      const bg = hi ? "1A3A2A" : C.cardBg;
      s.addShape("rect", { x: 3.52, y: yy, w: 3.02, h: 0.44, fill: { color: bg }, line: { color: hi ? C.green : "334155" } });
      s.addText(step, { x: 3.55, y: yy + 0.06, w: 0.75, h: 0.32, fontSize: 11, bold: hi, color: hi ? C.green : C.lightGray, fontFace: FONT_B, align: "center", margin: 0 });
      s.addText(yld, { x: 4.3, y: yy + 0.06, w: 0.85, h: 0.32, fontSize: 11, bold: hi, color: hi ? C.green : C.white, fontFace: FONT_B, align: "center", margin: 0 });
      s.addText(water, { x: 5.15, y: yy + 0.06, w: 1.25, h: 0.32, fontSize: 10.5, bold: hi, color: hi ? C.green : C.lightGray, fontFace: FONT_B, align: "center", margin: 0 });
    });

    // Finding 3: VDN bias ratio
    card(s, 6.75, 1.08, 2.9, 4.2);
    s.addShape("rect", { x: 6.75, y: 1.08, w: 2.9, h: 0.35, fill: { color: C.gold }, line: { color: C.gold } });
    s.addText("③ VDN bias ratio = 130 (constant)", { x: 6.82, y: 1.08, w: 2.78, h: 0.35, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 });

    const vdnData = [
      ["50k",   "2.98",  "0.037"],
      ["100k",  "3.15",  "0.036"],
      ["150k",  "2.66",  "0.021"],
      ["200k",  "−38.2", "0.52"],
      ["250k",  "−2583", "32.0"],
      ["500k",  "−860k", "6202"],
    ];
    s.addShape("rect", { x: 6.79, y: 1.52, w: 2.82, h: 0.3, fill: { color: C.cardBg }, line: { color: C.gold } });
    s.addText("Step  local_mean  local_std", { x: 6.82, y: 1.52, w: 2.78, h: 0.3, fontSize: 9.5, bold: true, color: C.gold, fontFace: FONT_B, align: "center", valign: "middle", margin: 0 });
    vdnData.forEach(([step, lm, ls], i) => {
      const yy = 1.85 + i * 0.49;
      s.addShape("rect", { x: 6.79, y: yy, w: 2.82, h: 0.44, fill: { color: i >= 3 ? "2A1A0A" : C.cardBg }, line: { color: "334155" } });
      const tc = i >= 3 ? C.gold : C.white;
      s.addText(step, { x: 6.83, y: yy + 0.06, w: 0.72, h: 0.32, fontSize: 10.5, color: tc, fontFace: FONT_B, align: "center", margin: 0 });
      s.addText(lm, { x: 7.55, y: yy + 0.06, w: 0.85, h: 0.32, fontSize: 10.5, color: tc, fontFace: "Consolas", align: "right", margin: 0 });
      s.addText(ls, { x: 8.38, y: yy + 0.06, w: 1.1, h: 0.32, fontSize: 10.5, color: tc, fontFace: "Consolas", align: "right", margin: 0 });
    });
    s.addText("VDN sum ≠ cascade cause.\nCascade is in local_q magnitude\nafter step 200k.", { x: 6.82, y: 5.05, w: 2.78, h: 0.9, fontSize: 10, color: C.lightGray, fontFace: FONT_B, margin: 0, italic: true });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SLIDE 15 — Computational cost
  // ═══════════════════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    darkBg(s);
    slideTitle(s, "Computational Cost: The Deployment Argument");

    statBlock(s, 0.35, 1.15, "25.9 min",  "MPC mean solve\nper decision step",  C.red);
    statBlock(s, 3.55, 1.15, "~1 ms",     "SAC inference\nper decision step",   C.green);
    statBlock(s, 6.75, 1.15, "25 000×",   "Speedup factor\n(SAC vs MPC)",       C.gold);

    card(s, 0.35, 2.45, 4.4, 2.85);
    s.addText("MPC Compute Profile", { x: 0.55, y: 2.53, w: 4.1, h: 0.35, fontSize: 13, bold: true, color: C.red, fontFace: FONT_H, margin: 0 });
    const mpcComp = [
      "Mean per-decision: 25.9 min",
      "Worst-case single call: 274 s",
      "Total season: 22–51 wall-minutes",
      "Requires high-performance CPU",
      "~3 120 decision vars + ~2 080 constraints per solve",
      "IPOPT warm-start provides 2.5× speedup",
    ];
    mpcComp.forEach((l, i) => {
      s.addText([{ text: "•  " + l }], { x: 0.55, y: 2.97 + i * 0.37, w: 4.1, h: 0.36, fontSize: 11.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    card(s, 5.1, 2.45, 4.55, 2.85);
    s.addText("SAC Compute Profile", { x: 5.28, y: 2.53, w: 4.22, h: 0.35, fontSize: 13, bold: true, color: C.green, fontFace: FONT_H, margin: 0 });
    const sacComp = [
      "Inference: ~1 ms for all 130 agents",
      "Total season: 0.3 s (vs 22–51 min for MPC)",
      "Cold start: 41 ms (model load)",
      "Compatible with 3 ms MCU edge budget",
      "No online optimisation required",
      "Training: ~3.5 h on A100 GPU (one-time)",
    ];
    sacComp.forEach((l, i) => {
      s.addText([{ text: "•  " + l }], { x: 5.28, y: 2.97 + i * 0.37, w: 4.28, h: 0.36, fontSize: 11.5, color: C.offWhite, fontFace: FONT_B, margin: 0 });
    });

    card(s, 0.35, 5.45, 9.3, 0.0 });
    // Replace with simpler footer
    s.addShape("rect", { x: 0.35, y: 5.18, w: 9.3, h: 0.28, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(
      "Even with the 7.6% yield gap in the OOD wet scenario, SAC's 25 000× latency advantage makes it the only viable option for embedded, real-time edge deployment.",
      { x: 0.45, y: 5.18, w: 9.15, h: 0.28, fontSize: 11, bold: true, color: C.dark, fontFace: FONT_H, align: "center", valign: "middle", margin: 0 }
    );
  }

  await pres.writeFile({ fileName: "/home/claude/thesis_defense.pptx" });
  console.log("Done — written to /home/claude/thesis_defense.pptx");
}

build().catch(console.error);