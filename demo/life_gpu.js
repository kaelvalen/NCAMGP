// Yaşayan ekosistem kuralının (LifeCA.forward) WebGL2 hesabı. Girdi: x, R, F, g; çıktı: ham y (n_out),
// etkinleştirmeler (sigmoid, softmax) JS'te. Fizik, yeme ve izleme demo/life.js'te kalır.
// Kullanım: const gpu = LifeGPU.create(world); if (gpu) world.forwardHook = (out) => gpu.forward(out);
(function (root) {
  "use strict";
  const M = 3;

  class LifeGPU {
    constructor(world) {
      const { H, W, C, G, K, IN, HD, NOUT, kern, w1, b1, w2, ob, Ps, Pb } = world;
      Object.assign(this, { world, H, W, C, G, NOUT });
      const canvas = root.document.createElement("canvas");
      const gl = (this.gl = canvas.getContext("webgl2", { antialias: false, depth: false }));
      if (!gl || !gl.getExtension("EXT_color_buffer_float")) throw new Error("WebGL2 kayan nokta dokusu yok");
      const CT = Math.ceil(C / 4), GT = Math.ceil(G / 4), P4 = Math.ceil(IN / 4), H4 = Math.ceil(HD / 4);
      const OT = (this.OT = Math.ceil(NOUT / 4));
      // tek ağırlık dokusu, satır blokları: W1 [0, HD), W2 [HD, HD+NOUT), PS [.., +HD), PB [.., +HD)
      // (ayrı dokular + x, g, maske 19 birim tutuyordu; WebGL2 yalnız 16 garanti eder, çoğu Windows/Mac 16 verir)
      const R_W2 = HD, R_PS = HD + NOUT, R_PB = HD + NOUT + HD, WROWS = HD + NOUT + 2 * HD;
      if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < OT) throw new Error(`en az ${OT} çizim tamponu gerekli`);
      canvas.width = W;
      canvas.height = H;
      canvas.addEventListener("webglcontextlost", () => { world.forwardHook = null; }); // JS yoluna döner
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      const range = (n) => Array.from({ length: n }, (_, i) => i);
      const s = world.filmScale;
      const fs = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D rf, WT, ${range(CT).map((t) => "x" + t).concat(range(GT).map((t) => "g" + t)).join(", ")};
${range(OT).map((t) => `layout(location=${t}) out vec4 o${t};`).join("\n")}
const float KER[${K * 9}] = float[${K * 9}](${kern.flat().map((v) => v.toFixed(8)).join(",")});
ivec2 wrap(ivec2 p) { return ivec2((p.x + ${W}) % ${W}, (p.y + ${H}) % ${H}); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ${range(OT).map((t) => `o${t} = vec4(0.);`).join(" ")}
  if (texelFetch(rf, p, 0).b < .5) return; // rf = (R, F, canlı maskesi)
  vec4 n[${(CT + 1) * 9}];
  for (int j = 0; j < 9; j++) {
    ivec2 q = wrap(p + ivec2(j % 3 - 1, j / 3 - 1));
    ${range(CT).map((t) => `n[${t} * 9 + j] = texelFetch(x${t}, q, 0);`).join(" ")}
    n[${CT} * 9 + j] = texelFetch(rf, q, 0);
  }
  float pin[${P4 * 4}];
  for (int i = 0; i < ${P4 * 4}; i++) pin[i] = 0.;
  for (int c = 0; c < ${C + 2}; c++) {
    int t = c < ${C} ? c / 4 : ${CT}, q = c < ${C} ? c % 4 : c - ${C};
    for (int k = 0; k < ${K}; k++) {
      float v = 0.;
      for (int j = 0; j < 9; j++) v += KER[k * 9 + j] * n[t * 9 + j][q];
      pin[c * ${K} + k] = v;
    }
  }
  vec4 P[${P4}];
  for (int i = 0; i < ${P4}; i++) P[i] = vec4(pin[i * 4], pin[i * 4 + 1], pin[i * 4 + 2], pin[i * 4 + 3]);
  vec4 gv[${GT}];
  ${range(GT).map((t) => `gv[${t}] = texelFetch(g${t}, p, 0);`).join(" ")}
  vec4 h[${H4}];
  for (int i = 0; i < ${H4}; i++) h[i] = vec4(0.);
  for (int j = 0; j < ${HD}; j++) {
    float v = texelFetch(WT, ivec2(${P4}, j), 0).r;
    for (int i = 0; i < ${P4}; i++) v += dot(texelFetch(WT, ivec2(i, j), 0), P[i]);
    ${s ? `float gs = 0., gb = 0.;
    for (int i = 0; i < ${GT}; i++) { gs += dot(texelFetch(WT, ivec2(i, ${R_PS} + j), 0), gv[i]); gb += dot(texelFetch(WT, ivec2(i, ${R_PB} + j), 0), gv[i]); }
    v = v * (1. + ${s.toFixed(8)} * gs) + ${s.toFixed(8)} * gb;` : ""}
    h[j / 4][j % 4] = max(v, 0.);
  }
  float y[${OT * 4}];
  for (int o = 0; o < ${OT * 4}; o++) y[o] = 0.;
  for (int o = 0; o < ${NOUT}; o++) {
    float v = texelFetch(WT, ivec2(${H4}, ${R_W2} + o), 0).r;
    for (int i = 0; i < ${H4}; i++) v += dot(texelFetch(WT, ivec2(i, ${R_W2} + o), 0), h[i]);
    y[o] = v;
  }
  ${range(OT).map((t) => `o${t} = vec4(y[${t * 4}], y[${t * 4 + 1}], y[${t * 4 + 2}], y[${t * 4 + 3}]);`).join(" ")}
}`;
      const vs = `#version 300 es
void main() { gl_Position = vec4(vec2[3](vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.))[gl_VertexID], 0., 1.); }`;
      const prog = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
        gl.attachShader(prog, sh);
      }
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      gl.useProgram(prog);
      gl.bindVertexArray(gl.createVertexArray());
      let unit = 0;
      const tex = (name, internal, format, type, w, h, data) => {
        const t = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        gl.uniform1i(gl.getUniformLocation(prog, name), unit++);
        return { t, unit: unit - 1 };
      };
      const float = (name, w, h, data = null) => tex(name, gl.RGBA32F, gl.RGBA, gl.FLOAT, w, h, data);
      // ağırlıklar satır başına vec4 paketli; W1 ve W2'nin son sütunu sapma
      const pack = (rows, cols, get, bias) => {
        const c4 = Math.ceil(cols / 4), a = new Float32Array((c4 + 1) * 4 * rows);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) a[r * (c4 + 1) * 4 + c] = get(r, c);
          if (bias) a[r * (c4 + 1) * 4 + c4 * 4] = bias[r];
        }
        return a;
      };
      const WW = Math.max(P4, H4, GT) + 1, WTd = new Float32Array(WW * 4 * WROWS);
      const put = (row0, cols4, a) => { // pack() satırı (cols4 + 1) vec4 genişliğinde; WW'ye sığdır
        const rows = a.length / ((cols4 + 1) * 4);
        for (let r = 0; r < rows; r++) WTd.set(a.subarray(r * (cols4 + 1) * 4, (r + 1) * (cols4 + 1) * 4), (row0 + r) * WW * 4);
      };
      put(0, P4, pack(HD, IN, (r, c) => w1[r * IN + c], b1));
      put(R_W2, H4, pack(NOUT, HD, (r, c) => w2[r * HD + c], ob));
      put(R_PS, GT, pack(HD, G, (r, c) => Ps[r * G + c]));
      put(R_PB, GT, pack(HD, G, (r, c) => Pb[r * G + c]));
      float("WT", WW, WROWS, WTd);
      if (unit + 1 + CT + GT > gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)) throw new Error("doku birimi yetmiyor");
      this.rfTex = float("rf", W, H);
      this.xTex = range(CT).map((t) => float("x" + t, W, H));
      this.gTex = range(GT).map((t) => float("g" + t, W, H));
      gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer());
      gl.activeTexture(gl.TEXTURE0 + unit);
      const attach = range(OT).map((t) => gl.COLOR_ATTACHMENT0 + t);
      for (const t of range(OT)) {
        const o = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, o);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, attach[t], gl.TEXTURE_2D, o, 0);
      }
      gl.drawBuffers(attach);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("çerçeve tamponu eksik");
      gl.viewport(0, 0, W, H);
      const N = W * H;
      this.buf = new Float32Array(N * 4);
      this.mask = new Uint8Array(N);
      this.y = new Float32Array(N * OT * 4);
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      this.renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "bilinmiyor";
    }

    // src: HWC dizi (stride kanal), off: ilk kanal, n: bu dokudaki kanal sayısı (<= 4)
    upload(tx, src, stride, off, n) {
      const { gl, W, H, buf } = this, N = W * H;
      buf.fill(0);
      for (let i = 0; i < N; i++) for (let q = 0; q < n; q++) buf[i * 4 + q] = src[i * stride + off + q];
      gl.activeTexture(gl.TEXTURE0 + tx.unit);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, buf);
    }

    // world.forward ile aynı sözleşme: out.{drgb, dh, grow, flow, attack}; m <= 0 hücrede flow = kal
    forward(out) {
      const { gl, W, H, C, G, OT, buf, mask, y, world } = this, N = W * H, nh = world.nh, x = world.x, g = world.g;
      for (let i = 0; i < N; i++) mask[i] = x[i * C + M] > 0 ? 1 : 0;
      this.xTex.forEach((tx, t) => this.upload(tx, x, C, t * 4, Math.min(4, C - t * 4)));
      this.gTex.forEach((tx, t) => this.upload(tx, g, G, t * 4, Math.min(4, G - t * 4)));
      const rf = this.rf || (this.rf = new Float32Array(N * 3));
      for (let i = 0; i < N; i++) { rf[i * 3] = world.R[i]; rf[i * 3 + 1] = world.F[i]; rf[i * 3 + 2] = mask[i]; }
      this.upload(this.rfTex, rf, 3, 0, 3);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const Y = OT * 4;
      for (let t = 0; t < OT; t++) {
        gl.readBuffer(gl.COLOR_ATTACHMENT0 + t);
        gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, buf);
        for (let i = 0; i < N; i++) for (let q = 0; q < 4; q++) y[i * Y + t * 4 + q] = buf[i * 4 + q];
      }
      for (let i = 0; i < N; i++) {
        if (!mask[i]) {
          for (let o = 0; o < 9; o++) out.flow[i * 9 + o] = o === 0 ? 1 : 0;
          continue;
        }
        const b = i * Y;
        for (let c = 0; c < 3; c++) out.drgb[i * 3 + c] = y[b + c];
        for (let c = 0; c < nh; c++) out.dh[i * nh + c] = y[b + 3 + c];
        out.grow[i] = 1 / (1 + Math.exp(-y[b + 3 + nh]));
        let mx = -Infinity, sum = 0;
        for (let o = 0; o < 9; o++) mx = Math.max(mx, y[b + 4 + nh + o]);
        for (let o = 0; o < 9; o++) sum += out.flow[i * 9 + o] = Math.exp(y[b + 4 + nh + o] - mx);
        for (let o = 0; o < 9; o++) out.flow[i * 9 + o] /= sum;
        out.attack[i] = 1 / (1 + Math.exp(-y[b + 13 + nh]));
      }
    }

    // Dünyanın mevcut durumunda GPU ve CPU kural çıktıları karşılaştırılır (en büyük mutlak fark)
    compare() {
      const w = this.world, mk = () => ({ drgb: new Float32Array(w.N * 3), dh: new Float32Array(w.N * w.nh),
        grow: new Float32Array(w.N), flow: new Float32Array(w.N * 9), attack: new Float32Array(w.N) });
      const a = mk(), b = mk();
      w.foreignMass();
      w.forward(a);
      this.forward(b);
      let e = 0;
      for (const k in a) for (let i = 0; i < a[k].length; i++) e = Math.max(e, Math.abs(a[k][i] - b[k][i]));
      return (this.error = e);
    }

    // Yazılım GL'i (SwiftShader, llvmpipe) CPU'dan da yavaş; o durumda ya da sonuç tutmazsa JS yolu
    static create(world) {
      try {
        const g = new LifeGPU(world);
        if (/swiftshader|llvmpipe|software/i.test(g.renderer)) return null;
        if (!(g.compare() < 1e-3)) { console.warn("GPU sonucu CPU ile tutmuyor:", g.error); return null; }
        return g;
      } catch (e) {
        console.warn("GPU kullanılamıyor:", e.message);
        return null;
      }
    }
  }

  root.LifeGPU = LifeGPU;
})(typeof window !== "undefined" ? window : globalThis);
