// Yaşayan ekosistem demosu arayüzü (demo/life.html). Fizik demo/life.js, kural GPU'da demo/life_gpu.js.
// Adres sonu: #cpu kuralı işlemcide çalıştırır, #selftest GPU ile CPU kural farkını konsola yazar.
(function () {
  "use strict";
  const model = window.LIFE_MODEL, states = window.LIFE_STATES || [];
  const $ = (id) => document.getElementById(id);
  if (!model || !states.length) {
    $("app").style.display = "none";
    $("nomodels").style.display = "block";
    return;
  }
  const S = states[0].H, M = 3, E = 5;
  const view = $("view"), overlay = $("overlay"), chart = $("chart");
  view.width = view.height = S;
  const vctx = view.getContext("2d"), octx = overlay.getContext("2d"), cctx = chart.getContext("2d");
  const img = vctx.createImageData(S, S);
  const css = getComputedStyle(document.documentElement), color = (n) => css.getPropertyValue(n).trim();
  const BG = [241, 241, 239];
  const world = new LifeWorld(model, S, S);
  // soy görünümü: sabit rastgele izdüşüm (genom -> renk)
  let seed = 12345;
  const rand01 = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const proj = Array.from({ length: 3 * world.G }, () => (rand01() - 0.5) * 1.2);

  let running = !matchMedia("(prefers-reduced-motion: reduce)").matches, speed = 20, budget = 0;
  let lastFrame = performance.now(), tool = "follow", viewMode = "rgb", drag = null, hover = null, followId = null;
  let stepsSec = 0, fpsT = performance.now(), lastStats = -Infinity, gpu = null;

  const HINTS = {
    follow: "Bir canlıya tıklayın. Hızı, avcılık payı, yaşı ve ebeveyni yukarıda izlenir.",
    cut: "Bir canlının üzerinden çizgi çekin. Çizgi kenardan taşarsa karşı kenardan devam eder.",
    erase: "Sürükleyerek bir bölgeyi silin. Besin toprağa döner, kenarda yara potansiyeli oluşur.",
    seed: "Tıklayın. İzlenen canlının (yoksa rastgele bir canlının) genomuyla yeni bir tohum eklenir.",
  };

  function start(k) {
    const st = states[k];
    world.loadState(st);
    followId = null;
    $("mu").value = st.name === "mu0" ? 0 : model.world.mu;
    $("mu").oninput();
    document.querySelectorAll("#starts button").forEach((b, i) => b.classList.toggle("on", i === k));
    $("start-hint").textContent = st.label;
    if (!gpu && location.hash !== "#cpu") {
      gpu = LifeGPU.create(world); // ilk durum yüklendikten sonra: denetim gerçek bedenlerde yapılır
      if (gpu) world.forwardHook = (out) => gpu.forward(out);
      if (location.hash === "#selftest") console.log(`life selftest: en büyük |GPU - CPU| = ${gpu ? gpu.error : "GPU yok"}`);
    }
    $("engine").textContent = world.forwardHook ? `Kural GPU'da çalışıyor: ${gpu.renderer}` : "Kural işlemcide çalışıyor (yavaş). Hızı düşürmek ısınmayı azaltır.";
  }

  // --- kontroller ---
  states.forEach((st, k) => {
    const b = document.createElement("button");
    b.textContent = { ata: "Ata", mu1: "Evrimleşmiş", mu0: "Mutasyonsuz" }[st.name] || st.name;
    b.onclick = () => start(k);
    $("starts").appendChild(b);
  });
  const bind = (id, fmt, fn) => {
    const el = $(id), o = $(id + "-o");
    el.oninput = () => { o.textContent = fmt(+el.value); fn && fn(+el.value); };
    el.oninput();
  };
  bind("brush", (v) => v);
  bind("mu", (v) => v.toFixed(2), (v) => (world.p.mu = v));
  bind("speed", (v) => v, (v) => (speed = v));
  const syncPlay = () => ($("play").textContent = running ? "Duraklat" : "Oynat");
  $("play").onclick = () => { running = !running; syncPlay(); };
  syncPlay();
  $("step").onclick = () => world.step();
  document.querySelectorAll("#views button").forEach((b) => (b.onclick = () => {
    viewMode = b.dataset.view;
    document.querySelectorAll("#views button").forEach((o) => o.classList.toggle("on", o === b));
  }));
  function setTool(t) {
    tool = t;
    document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
    $("tool-hint").textContent = HINTS[t];
  }
  document.querySelectorAll("#tools button").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  setTool("follow");

  // --- fare (koordinatlar kırpılmaz: tuval dışına taşan çizgi sarmal dünyada karşı kenardan sürer) ---
  const gridPos = (e) => {
    const r = overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * S, y: ((e.clientY - r.top) / r.height) * S };
  };
  const cellAt = (p) => (((Math.floor(p.y) % S) + S) % S) * S + (((Math.floor(p.x) % S) + S) % S);
  overlay.addEventListener("pointerdown", (e) => {
    overlay.setPointerCapture(e.pointerId);
    const p = gridPos(e);
    drag = { a: p, b: p };
    if (tool === "erase") world.erase(Math.floor(p.y), Math.floor(p.x), +$("brush").value);
    else if (tool === "seed") seedAt(p);
    else if (tool === "follow") {
      const l = world.labels[cellAt(p)];
      followId = l ? world.labelToId.get(l) : null;
      if (followId === undefined) followId = null;
    }
  });
  overlay.addEventListener("pointermove", (e) => {
    hover = gridPos(e);
    if (!drag) return;
    drag.b = hover;
    if (tool === "erase") world.erase(Math.floor(hover.y), Math.floor(hover.x), +$("brush").value);
  });
  const end = () => {
    if (drag && tool === "cut") {
      const { a, b } = drag;
      if (Math.hypot(b.x - a.x, b.y - a.y) > 2) world.cut(a.y, a.x, b.y, b.x);
    }
    drag = null;
  };
  overlay.addEventListener("pointerup", end);
  overlay.addEventListener("pointercancel", end);
  overlay.addEventListener("pointerleave", () => (hover = null));

  function seedAt(p) {
    const live = world.liveOrgs();
    const src = world.orgs.get(followId);
    const g = src && src.alive ? src.genome : live.length ? live[Math.floor(Math.random() * live.length)].genome : null;
    const i = cellAt(p);
    world.plant(Math.floor(i / S), i % S, g);
  }

  // --- çizim ---
  function render() {
    const d = img.data, { C, G, x, g, R } = world;
    for (let i = 0; i < world.N; i++) {
      const a = Math.min(1, Math.max(0, x[i * C + M]));
      let r, gg, b, al = a;
      if (viewMode === "rgb") {
        r = x[i * C] * 255; gg = x[i * C + 1] * 255; b = x[i * C + 2] * 255;
      } else if (viewMode === "lineage") {
        const c = [0, 0, 0];
        for (let k = 0; k < 3; k++) {
          let s = 0;
          for (let q = 0; q < G; q++) s += proj[k * G + q] * g[i * G + q];
          c[k] = 127.5 + 127.5 * Math.tanh(s);
        }
        [r, gg, b] = c;
      } else if (viewMode === "energy") {
        const e = Math.min(1, Math.max(0, x[i * C + E] / 2));
        r = 255 * e; gg = 120 * e; b = 20; al = a;
      } else {
        const v = Math.min(1, R[i]);
        r = 42; gg = 120; b = 214; al = v;
      }
      d[i * 4] = r * al + BG[0] * (1 - al);
      d[i * 4 + 1] = gg * al + BG[1] * (1 - al);
      d[i * 4 + 2] = b * al + BG[2] * (1 - al);
      d[i * 4 + 3] = 255;
    }
    vctx.putImageData(img, 0, 0);
  }

  function drawOverlay() {
    const k = overlay.width / S;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.strokeStyle = color("--wound");
    octx.lineWidth = 2;
    const o = world.orgs.get(followId);
    if (o && o.alive) {
      const [cy, cx] = o.centroid;
      octx.beginPath();
      octx.arc((cx + 0.5) * k, (cy + 0.5) * k, (Math.sqrt(o.area / Math.PI) + 3) * k, 0, Math.PI * 2);
      octx.stroke();
    }
    if (drag && tool === "cut") {
      octx.beginPath();
      octx.moveTo(drag.a.x * k, drag.a.y * k);
      octx.lineTo(drag.b.x * k, drag.b.y * k);
      octx.stroke();
    } else if (hover && tool === "erase") {
      octx.beginPath();
      octx.arc(hover.x * k, hover.y * k, +$("brush").value * k, 0, Math.PI * 2);
      octx.stroke();
    }
  }

  function drawChart() {
    const W = chart.width, H = chart.height, hist = world.history;
    cctx.clearRect(0, 0, W, H);
    if (hist.length < 2) return;
    for (const [key, col] of [["n_org", color("--ink")], ["R_mean", color("--res")], ["pred_share", color("--wound")]]) {
      const max = Math.max(1e-6, ...hist.map((h) => h[key]));
      cctx.strokeStyle = col;
      cctx.lineWidth = 3;
      cctx.beginPath();
      hist.forEach((h, i) => {
        const px = (i / (hist.length - 1)) * (W - 4) + 2, py = H - 4 - (h[key] / max) * (H - 10);
        i ? cctx.lineTo(px, py) : cctx.moveTo(px, py);
      });
      cctx.stroke();
    }
  }

  function updatePanels() {
    const st = world.history.length ? world.history[world.history.length - 1] : world.stats();
    $("s-step").textContent = world.t;
    $("s-org").textContent = st.n_org;
    $("s-res").textContent = st.R_mean.toFixed(3);
    $("s-pred").textContent = st.pred_share.toFixed(3);
    if (world.extinct) $("tool-hint").textContent = "Nüfus tükendi. Tohum aracıyla yeni canlı ekin.";
    const o = world.orgs.get(followId), set = (id, v) => ($(id).textContent = v);
    if (!o) { for (const id of ["f-id", "f-parent", "f-gen", "f-age", "f-area", "f-speed", "f-pred"]) set(id, "–"); return; }
    set("f-id", o.alive ? o.id : `${o.id} (öldü, adım ${o.died})`);
    set("f-parent", o.parent < 0 ? "yok" : o.parent);
    set("f-gen", o.generation);
    set("f-age", (o.alive ? world.t : o.died) - o.born);
    set("f-area", o.area);
    set("f-speed", o.speed.toFixed(3));
    set("f-pred", o.eaten + o.uptake > 0 ? (o.eaten / (o.eaten + o.uptake)).toFixed(3) : "0");
  }

  function frame(now) {
    // adım hızı kare hızından bağımsız ve sınırlı: yavaş makinede kareler donmasın
    budget = Math.min(budget + (speed * (now - lastFrame)) / 1000, 2);
    lastFrame = now;
    if (running && budget >= 1) {
      const n = Math.floor(budget);
      budget -= n;
      for (let i = 0; i < n; i++) world.step();
      stepsSec += n;
    }
    render();
    drawOverlay();
    if (now - lastStats > 300) {
      updatePanels();
      drawChart();
      lastStats = now;
    }
    if (now - fpsT > 1000) {
      $("s-fps").textContent = ((stepsSec * 1000) / (now - fpsT)).toFixed(1);
      stepsSec = 0;
      fpsT = now;
    }
    requestAnimationFrame(frame);
  }
  start(Math.max(0, states.findIndex((s) => s.name === "mu1")));
  requestAnimationFrame(frame);
})();
