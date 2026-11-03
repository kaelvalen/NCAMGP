// Ekosistem demosu arayüzü.
(function () {
  "use strict";
  const SIZE = 128;
  const models = window.EVO_MODELS || [];
  if (!models.length) {
    document.getElementById("app").style.display = "none";
    document.getElementById("nomodels").style.display = "block";
    return;
  }
  const $ = (id) => document.getElementById(id);
  const view = $("view"), overlay = $("overlay"), chart = $("chart");
  // dünya 128x128 kalır; tarayıcının büyütmesi yerine yüksek kaliteli yeniden örnekleme
  const src = document.createElement("canvas");
  src.width = src.height = SIZE;
  view.width = view.height = 900;
  const sctx = src.getContext("2d"), vctx = view.getContext("2d"), octx = overlay.getContext("2d"), cctx = chart.getContext("2d");
  vctx.imageSmoothingQuality = "high";
  const img = sctx.createImageData(SIZE, SIZE);
  const css = getComputedStyle(document.documentElement), color = (name) => css.getPropertyValue(name).trim();
  const BG = [0.945, 0.945, 0.937];
  let shown = null;

  const world = new EvoWorld(models[0], SIZE, SIZE);
  world.gpu = location.hash === "#cpu" ? null : EvoGPU.create(world);
  $("engine").textContent = world.gpu ? `Kural GPU'da çalışıyor: ${world.gpu.renderer}` : "Kural işlemcide çalışıyor. Hızı düşürmek ısınmayı azaltır.";
  // soy görünümü için sabit rastgele izdüşüm (genom -> renk)
  let seed = 12345;
  const rand01 = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const proj = Array.from({ length: 3 * world.G }, () => (rand01() - 0.5) * 1.6);

  let running = !matchMedia("(prefers-reduced-motion: reduce)").matches, speed = 30, budget = 0, lastFrame = performance.now(), histT = 0, tool = "cut", viewMode = "rgb", predRate = 0.3;
  let drag = null, hover = null, comps = null, compsT = -1, born = 0, lastIds = 0;
  const hist = [];
  let stepsSec = 0, fpsT = performance.now(), lastStats = -Infinity;

  const HINTS = {
    cut: "Bir canlının üzerinden çizgi çekin. Parçalar yavaşça ayrılır ve yenilenir.",
    erase: "Sürükleyerek bir bölgeyi silin. Kenarlarda potansiyel birikir.",
    seed: "Tıklayın. Rastgele genomlu yeni bir tohum eklenir.",
    inspect: "Bir canlıya tıklayın. Alanı ve genomu Görünüm altında yazılır.",
  };

  function newWorld() {
    world.reset();
    histT = 0;
    compsT = -1;
    born = 0;
    hist.length = 0;
    const n = +$("nseed").value;
    for (let k = 0; k < n; k++) {
      for (let t = 0; t < 40; t++) {
        const x = 12 + Math.floor(Math.random() * (SIZE - 24)), y = 12 + Math.floor(Math.random() * (SIZE - 24));
        let free = true;
        for (let dy = -10; dy <= 10 && free; dy++)
          for (let dx = -10; dx <= 10; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < SIZE && xx >= 0 && xx < SIZE && world.x[(yy * SIZE + xx) * world.C + 3] > 0.1) { free = false; break; }
          }
        if (free) { world.plant(x, y, world.randomGenome()); break; }
      }
    }
    born = n;
  }

  // --- kontroller ---
  const bind = (id, fmt, fn) => {
    const el = $(id), o = $(id + "-o");
    const upd = () => { o.textContent = fmt(+el.value); fn && fn(+el.value); };
    el.oninput = upd;
    upd();
  };
  bind("brush", (v) => v);
  bind("mu", (v) => v.toFixed(2), (v) => (world.p.mu = v));
  bind("pred", (v) => v.toFixed(2), (v) => (predRate = v));
  bind("regrow", (v) => v.toFixed(4), (v) => (world.p.regrow = v));
  bind("nseed", (v) => v);
  bind("speed", (v) => v, (v) => (speed = v));
  const syncPlay = () => ($("play").textContent = running ? "Duraklat" : "Oynat");
  $("play").onclick = () => { running = !running; syncPlay(); };
  syncPlay();
  $("step").onclick = () => world.step();
  $("reset").onclick = newWorld;
  document.querySelectorAll("#views button").forEach((b) => (b.onclick = () => {
    viewMode = b.dataset.view;
    document.querySelectorAll("#views button").forEach((o) => o.classList.toggle("on", o === b));
  }));
  $("smooth").onchange = (e) => (vctx.imageSmoothingEnabled = e.target.checked);
  function setTool(t) {
    tool = t;
    document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
    $("tool-hint").textContent = HINTS[t];
  }
  document.querySelectorAll("#tools button").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  setTool("cut");

  // --- fare ---
  const gridPos = (e) => {
    const r = overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  };
  overlay.addEventListener("pointerdown", (e) => {
    overlay.setPointerCapture(e.pointerId);
    const p = gridPos(e);
    drag = { a: p, b: p };
    if (tool === "erase") world.erase(p.x - 0.5, p.y - 0.5, +$("brush").value);
    else if (tool === "seed") { world.plant(Math.floor(p.x), Math.floor(p.y), world.randomGenome()); born++; }
    else if (tool === "inspect") inspect(p);
  });
  overlay.addEventListener("pointermove", (e) => {
    hover = gridPos(e);
    if (!drag) return;
    drag.b = hover;
    if (tool === "erase") world.erase(hover.x - 0.5, hover.y - 0.5, +$("brush").value);
  });
  const end = () => {
    if (drag && tool === "cut") {
      const { a, b } = drag;
      if (Math.hypot(b.x - a.x, b.y - a.y) > 2) {
        // yalnız çizginin geçtiği canlıları kes
        const { lab, comps: cs } = world.components(5);
        const hit = new Set();
        const n = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) * 2);
        for (let k = 0; k <= n; k++) {
          const x = Math.floor(a.x + ((b.x - a.x) * k) / n), y = Math.floor(a.y + ((b.y - a.y) * k) / n);
          if (x >= 0 && x < SIZE && y >= 0 && y < SIZE && lab[y * SIZE + x]) hit.add(lab[y * SIZE + x]);
        }
        for (const c of cs) if (hit.has(c.id)) world.cut(a.x, a.y, b.x, b.y, null, c.cells);
      }
    }
    drag = null;
  };
  overlay.addEventListener("pointerup", end);
  overlay.addEventListener("pointercancel", end);
  overlay.addEventListener("pointerleave", () => (hover = null));

  function inspect(p) {
    const { lab, comps: cs } = world.components(5);
    const id = lab[Math.floor(p.y) * SIZE + Math.floor(p.x)];
    const c = cs.find((q) => q.id === id);
    if (!c) { $("inspect-out").textContent = "Burada canlı yok."; return; }
    const gm = new Array(world.G).fill(0);
    for (const i of c.cells) for (let q = 0; q < world.G; q++) gm[q] += world.g[i * world.G + q] / c.area;
    $("inspect-out").textContent = `Alan ${c.area} hücre, genom ${gm.map((v) => v.toFixed(1)).join(" ")}`;
  }

  // --- yırtıcı: rastgele canlıyı rastgele açıyla kes ---
  function predation(dtSteps) {
    if (!comps || predRate <= 0) return;
    const pr = (predRate * dtSteps) / 60; // ~60 adım = 1 sn varsayımı
    for (const c of comps) {
      // kesilme olasılığı alanla orantılı (büyük bireyler daha sık bölünür)
      if (c.area < 80 || Math.random() > (pr * c.area) / 300 / Math.max(1, comps.length)) continue;
      // yarası kapanmamış canlı yeniden kesilmez: kural önce iyileştirsin
      let sy = 0, sx = 0, v = 0;
      for (const i of c.cells) { sy += (i / SIZE) | 0; sx += i % SIZE; v += world.x[i * world.C + 4]; }
      if (v / c.area > 1e-3) continue;
      sy /= c.area; sx /= c.area;
      const ang = Math.random() * Math.PI;
      world.cut(sx + 0.5, sy + 0.5, sx + 0.5 + Math.cos(ang), sy + 0.5 + Math.sin(ang), null, c.cells);
      break;
    }
  }

  function diversity(cs) {
    if (cs.length < 2) return 0;
    const ms = cs.map((c) => {
      const m = new Array(world.G).fill(0);
      for (const i of c.cells) for (let q = 0; q < world.G; q++) m[q] += world.g[i * world.G + q] / c.area;
      return m;
    });
    let s = 0, n = 0;
    for (let i = 0; i < ms.length; i++) for (let j = i + 1; j < ms.length; j++) { s += Math.hypot(...ms[i].map((v, q) => v - ms[j][q])); n++; }
    return s / n;
  }

  function drawChart() {
    const W = chart.width, H = chart.height;
    cctx.clearRect(0, 0, W, H);
    if (hist.length < 2) return;
    const series = [["n", color("--ink")], ["div", color("--wound")], ["res", color("--quiet")]];
    for (const [k, col] of series) {
      const max = Math.max(1e-6, ...hist.map((h) => h[k]));
      cctx.strokeStyle = col;
      cctx.lineWidth = 3;
      cctx.beginPath();
      hist.forEach((h, i) => {
        const x = (i / (hist.length - 1)) * (W - 4) + 2, y = H - 4 - (h[k] / max) * (H - 10);
        i ? cctx.lineTo(x, y) : cctx.moveTo(x, y);
      });
      cctx.stroke();
    }
  }

  function drawOverlay() {
    const k = overlay.width / SIZE;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (drag && tool === "cut") {
      const { a, b } = drag;
      octx.strokeStyle = color("--wound");
      octx.lineWidth = 2;
      octx.beginPath();
      octx.moveTo(a.x * k, a.y * k);
      octx.lineTo(b.x * k, b.y * k);
      octx.stroke();
    } else if (hover && tool === "erase") {
      octx.strokeStyle = color("--wound");
      octx.lineWidth = 2;
      octx.beginPath();
      octx.arc(hover.x * k, hover.y * k, +$("brush").value * k, 0, Math.PI * 2);
      octx.stroke();
    }
  }

  function frame(now) {
    // adım hızı kare hızından bağımsız ve sınırlı: çekirdeği sürekli doldurmasın
    budget = Math.min(budget + (speed * (now - lastFrame)) / 1000, 4);
    lastFrame = now;
    if (running && budget >= 1) {
      const n = Math.floor(budget);
      budget -= n;
      for (let i = 0; i < n; i++) world.step();
      stepsSec += n;
      if (world.steps - compsT >= 10) {
        world.cleanDebris();
        const { comps: cs } = world.components(20);
        if (comps && cs.length > comps.length) born += cs.length - comps.length;
        comps = cs;
        compsT = world.steps;
        predation(10);
        if (world.steps - histT >= 30) {
          histT = world.steps;
          let r = 0;
          for (let i = 0; i < world.N; i++) r += world.R[i];
          hist.push({ n: cs.length, div: diversity(cs), res: r / world.N });
          if (hist.length > 400) hist.shift();
        }
      }
    }
    world.render(img, viewMode, BG, proj);
    // ekranda zamansal yumuşatma: rastgele ateşleme titremesi ve ani ölümler sıçrama olarak görünmez
    if (!shown) shown = Float32Array.from(img.data);
    for (let q = 0; q < shown.length; q++) img.data[q] = shown[q] += (img.data[q] - shown[q]) * 0.3;
    sctx.putImageData(img, 0, 0);
    vctx.drawImage(src, 0, 0, view.width, view.height);
    drawOverlay();
    if (now - lastStats > 300) {
      $("s-step").textContent = world.steps;
      $("s-org").textContent = comps ? comps.length : 0;
      $("s-born").textContent = born;
      $("s-div").textContent = hist.length ? hist[hist.length - 1].div.toFixed(2) : "0";
      drawChart();
      lastStats = now;
    }
    if (now - fpsT > 1000) {
      $("s-fps").textContent = Math.round((stepsSec * 1000) / (now - fpsT));
      stepsSec = 0;
      fpsT = now;
    }
    requestAnimationFrame(frame);
  }
  newWorld();
  requestAnimationFrame(frame);
})();
