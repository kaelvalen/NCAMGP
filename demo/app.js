// Demo arayüzü: sil, kes, müdahale, tohum; model seçimi; hız ve görünüm.
(function () {
  "use strict";
  const SIZE = 128;
  const models = window.NCA_MODELS || [];
  if (!models.length) {
    document.getElementById("app").style.display = "none";
    document.getElementById("nomodels").style.display = "block";
    return;
  }

  const $ = (id) => document.getElementById(id);
  const view = $("view"), overlay = $("overlay");
  const vctx = view.getContext("2d"), octx = overlay.getContext("2d");
  const img = vctx.createImageData(SIZE, SIZE);

  let model = models.find((m) => m.name.startsWith("r3")) || models[models.length - 1];
  const nca = new NCA(model, SIZE, SIZE);
  let running = true, speed = 4, tool = "erase", viewMode = "rgb";
  let saved = null, drag = null, hover = null, lastStats = -Infinity;
  let stepsThisSec = 0, fpsT = performance.now();

  const HINTS = {
    erase: "Sürükleyerek fırça boyutunda bölge silin (dairesel hasar).",
    cut: "Bir çizgi çizin: organizma bu doğru boyunca ikiye bölünür, parçalar ayırma mesafesi kadar uzaklaşır.",
    paint: "Seçili gizli kanalı canlı hücrelerde boyayın (kutupluluk müdahalesi). Boyama sırasında simülasyon akmaya devam eder.",
    seed: "Tıklayın: o noktaya tek hücrelik tohum ekler.",
  };

  // --- model ---
  models.forEach((m, i) => $("model").add(new Option(m.label || m.name, i)));
  $("model").value = models.indexOf(model);
  $("model").onchange = () => {
    model = models[+$("model").value];
    nca.setModel(model);
    fillChannels();
  };

  // --- müdahale kanalları ---
  function fillChannels() {
    const sel = $("pchan");
    sel.innerHTML = "";
    const pr = model.probe;
    if (pr) sel.add(new Option(`Baş sinyali (prob: kanal ${pr.greedy_order.slice(0, 3).map((c) => c + 1).join(", ")})`, "head"));
    for (let i = 0; i < model.n_hidden; i++) sel.add(new Option(`Gizli kanal ${i + 1}`, 4 + i));
    $("probe-hint").textContent = pr
      ? `Prob R² (tüm gizli kanallar) = ${pr.r2_hidden_all.toFixed(2)}. "Baş sinyali" seçiliyken boyanan hücrelerde konum taşıyan 3 kanal baş bölgesi ortalamasına ayarlanır — kuyruk yarasına uygulayıp iki başlı organizma deneyin.`
      : "Bu model için prob sonucu yok (experiments/d4_probe.py çalıştırılıp yeniden dışa aktarılabilir).";
    const vm = $("viewmode");
    const cur = vm.value;
    vm.innerHTML = "";
    vm.add(new Option("RGB", "rgb"));
    vm.add(new Option("Alfa", "alpha"));
    for (let i = 0; i < model.n_hidden; i++) vm.add(new Option(`Gizli kanal ${i + 1}`, 4 + i));
    vm.value = [...vm.options].some((o) => o.value === cur) ? cur : "rgb";
    viewMode = vm.value === "rgb" || vm.value === "alpha" ? vm.value : +vm.value;
  }
  fillChannels();
  $("viewmode").onchange = (e) => {
    const v = e.target.value;
    viewMode = v === "rgb" || v === "alpha" ? v : +v;
  };

  function paintSpec() {
    const v = $("pchan").value;
    if (v === "head") {
      const ch = model.probe.greedy_order.slice(0, 3);
      return { channels: ch.map((c) => 4 + c), values: ch.map((c) => model.probe.head_means[c]) };
    }
    return { channels: [+v], values: [+$("pval").value] };
  }

  // --- araçlar ---
  function setTool(t) {
    tool = t;
    document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
    $("tool-hint").textContent = HINTS[t];
  }
  document.querySelectorAll("#tools button").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  setTool("erase");

  const bind = (id, fmt, fn) => {
    const el = $(id), o = $(id + "-o");
    const upd = () => {
      o.textContent = fmt(+el.value);
      fn && fn(+el.value);
    };
    el.oninput = upd;
    upd();
  };
  bind("brush", (v) => v);
  bind("gap", (v) => v);
  bind("pval", (v) => v.toFixed(2));
  bind("speed", (v) => v, (v) => (speed = v));

  // --- simülasyon kontrolleri ---
  function reset() {
    nca.clear();
    nca.plantSeed(SIZE >> 1, SIZE >> 1);
  }
  function togglePlay() {
    running = !running;
    $("play").textContent = running ? "Duraklat" : "Oynat";
  }
  $("play").onclick = togglePlay;
  $("step").onclick = () => nca.step();
  $("reset").onclick = reset;
  $("clear").onclick = () => nca.clear();
  $("save").onclick = () => (saved = { state: nca.state.slice(), steps: nca.steps, C: nca.C });
  $("load").onclick = () => {
    if (saved && saved.C === nca.C) {
      nca.state.set(saved.state);
      nca.steps = saved.steps;
    }
  };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === " ") (togglePlay(), e.preventDefault());
    else if (e.key === ".") nca.step();
    else if (e.key === "r" || e.key === "R") reset();
    else if ("1234".includes(e.key)) setTool(["erase", "cut", "paint", "seed"][+e.key - 1]);
  });

  // --- fare / dokunma ---
  function gridPos(e) {
    const r = overlay.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  }
  function applyBrush(p) {
    const r = +$("brush").value;
    if (tool === "erase") nca.erase(p.x - 0.5, p.y - 0.5, r);
    else if (tool === "paint") {
      const s = paintSpec();
      nca.paint(p.x - 0.5, p.y - 0.5, r, s.channels, s.values);
    }
  }
  overlay.addEventListener("pointerdown", (e) => {
    overlay.setPointerCapture(e.pointerId);
    const p = gridPos(e);
    drag = { start: p, cur: p };
    if (tool === "seed") nca.plantSeed(Math.floor(p.x), Math.floor(p.y));
    else applyBrush(p);
  });
  overlay.addEventListener("pointermove", (e) => {
    hover = gridPos(e);
    if (!drag) return;
    drag.cur = hover;
    if (tool === "erase" || tool === "paint") applyBrush(hover);
  });
  const end = () => {
    if (drag && tool === "cut") {
      const { start: a, cur: b } = drag;
      if (Math.hypot(b.x - a.x, b.y - a.y) > 2) nca.cut(a.x, a.y, b.x, b.y, +$("gap").value);
    }
    drag = null;
  };
  overlay.addEventListener("pointerup", end);
  overlay.addEventListener("pointercancel", end);
  overlay.addEventListener("pointerleave", () => (hover = null));

  function drawOverlay() {
    const k = overlay.width / SIZE;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    octx.lineWidth = 2;
    if (drag && tool === "cut") {
      const { start: a, cur: b } = drag;
      // sonsuz kesme doğrusunu çiz
      const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
      const ux = dx / L, uy = dy / L;
      octx.strokeStyle = "rgba(194,71,122,.9)";
      octx.setLineDash([8, 6]);
      octx.beginPath();
      octx.moveTo((a.x - ux * 400) * k, (a.y - uy * 400) * k);
      octx.lineTo((a.x + ux * 400) * k, (a.y + uy * 400) * k);
      octx.stroke();
      octx.setLineDash([]);
    } else if (hover && (tool === "erase" || tool === "paint")) {
      octx.strokeStyle = tool === "erase" ? "rgba(60,60,60,.7)" : "rgba(194,71,122,.9)";
      octx.beginPath();
      octx.arc(hover.x * k, hover.y * k, +$("brush").value * k, 0, Math.PI * 2);
      octx.stroke();
    }
  }

  // --- döngü ---
  function frame(now) {
    if (running) {
      for (let i = 0; i < speed; i++) nca.step();
      stepsThisSec += speed;
    }
    nca.render(img, viewMode);
    vctx.putImageData(img, 0, 0);
    drawOverlay();
    if (now - lastStats > 250) {
      const s = nca.stats(model.palette);
      $("s-step").textContent = nca.steps;
      $("s-area").textContent = s.area;
      $("s-ind").textContent = s.individuals;
      $("s-head").textContent = s.heads;
      lastStats = now;
    }
    if (now - fpsT > 1000) {
      $("s-fps").textContent = Math.round((stepsThisSec * 1000) / (now - fpsT));
      stepsThisSec = 0;
      fpsT = now;
    }
    requestAnimationFrame(frame);
  }
  reset();
  requestAnimationFrame(frame);
})();
