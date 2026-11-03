// Kuralın (f_θ) WebGL2 hesabı: durum + genom + maske -> delta. Kalıtım, potansiyel, etiket ve ekoloji JS'te kalır.
(function (root) {
  "use strict";
  const VCH = 4;

  class EvoGPU {
    constructor(world) {
      const { H, W, C, G, K, IN, HD, kernels, w1, b1, w2, senseV } = world;
      Object.assign(this, { H, W, C, G });
      const canvas = root.document.createElement("canvas");
      const gl = (this.gl = canvas.getContext("webgl2", { antialias: false, depth: false, powerPreference: "low-power" }));
      if (!gl || !gl.getExtension("EXT_color_buffer_float")) throw new Error("WebGL2 kayan nokta dokusu yok");
      canvas.width = W;
      canvas.height = H;
      canvas.addEventListener("webglcontextlost", () => { world.gpu = null; }); // JS yoluna döner
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      const CT = (this.CT = Math.ceil(C / 4)), GT = (this.GT = Math.ceil(G / 4)), P4 = Math.ceil(IN / 4), H4 = Math.ceil(HD / 4);
      const range = (n) => Array.from({ length: n }, (_, i) => i);
      const fs = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D mask, W1, W2, ${range(CT).map((t) => "x" + t).concat(range(GT).map((t) => "g" + t)).join(", ")};
${range(CT).map((t) => `layout(location=${t}) out vec4 o${t};`).join("\n")}
const float KER[${K * 9}] = float[${K * 9}](${kernels.flat().map((v) => v.toFixed(8)).join(",")});
vec4 at(sampler2D s, ivec2 p) { return (p.x < 0 || p.y < 0 || p.x >= ${W} || p.y >= ${H}) ? vec4(0.) : texelFetch(s, p, 0); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ${range(CT).map((t) => `o${t} = vec4(0.);`).join(" ")}
  if (texelFetch(mask, p, 0).r < .5) return;
  vec4 n[${CT * 9}];
  for (int j = 0; j < 9; j++) {
    ivec2 q = p + ivec2(j % 3 - 1, j / 3 - 1);
    ${range(CT).map((t) => `n[${t} * 9 + j] = at(x${t}, q);`).join(" ")}
    ${senseV ? "" : `n[${(VCH >> 2)} * 9 + j][${VCH & 3}] = 0.;`}
  }
  float pin[${P4 * 4}];
  for (int i = 0; i < ${P4 * 4}; i++) pin[i] = 0.;
  for (int c = 0; c < ${C}; c++) for (int k = 0; k < ${K}; k++) {
    float v = 0.;
    for (int j = 0; j < 9; j++) v += KER[k * 9 + j] * n[(c / 4) * 9 + j][c % 4];
    pin[c * ${K} + k] = v;
  }
  ${range(G).map((q) => `pin[${C * K + q}] = texelFetch(g${q >> 2}, p, 0)[${q & 3}];`).join(" ")}
  vec4 P[${P4}];
  for (int i = 0; i < ${P4}; i++) P[i] = vec4(pin[i * 4], pin[i * 4 + 1], pin[i * 4 + 2], pin[i * 4 + 3]);
  vec4 h[${H4}];
  for (int i = 0; i < ${H4}; i++) h[i] = vec4(0.);
  for (int j = 0; j < ${HD}; j++) {
    float v = texelFetch(W1, ivec2(${P4}, j), 0).r;
    for (int i = 0; i < ${P4}; i++) v += dot(texelFetch(W1, ivec2(i, j), 0), P[i]);
    h[j / 4][j % 4] = max(v, 0.);
  }
  float d[${CT * 4}];
  for (int c = 0; c < ${CT * 4}; c++) d[c] = 0.;
  for (int c = 0; c < ${C}; c++) {
    if (c == ${VCH}) continue; // kural potansiyeli değiştiremez
    for (int i = 0; i < ${H4}; i++) d[c] += dot(texelFetch(W2, ivec2(i, c), 0), h[i]);
  }
  ${range(CT).map((t) => `o${t} = vec4(d[${t * 4}], d[${t * 4 + 1}], d[${t * 4 + 2}], d[${t * 4 + 3}]);`).join(" ")}
}`;
      const vs = `#version 300 es
void main() { gl_Position = vec4(vec2[3](vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.))[gl_VertexID], 0., 1.); }`;
      const prog = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        gl.attachShader(prog, s);
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
      // ağırlıklar satır başına vec4 paketli; W1'in son sütunu sapma
      const W1 = new Float32Array((P4 + 1) * 4 * HD), W2 = new Float32Array(H4 * 4 * C);
      for (let j = 0; j < HD; j++) {
        for (let i = 0; i < IN; i++) W1[j * (P4 + 1) * 4 + i] = w1[j * IN + i];
        W1[j * (P4 + 1) * 4 + P4 * 4] = b1[j];
      }
      for (let c = 0; c < C; c++) for (let j = 0; j < HD; j++) W2[c * H4 * 4 + j] = w2[c * HD + j];
      float("W1", P4 + 1, HD, W1);
      float("W2", H4, C, W2);
      this.maskTex = tex("mask", gl.R8, gl.RED, gl.UNSIGNED_BYTE, W, H, null);
      this.xTex = range(CT).map((t) => float("x" + t, W, H));
      this.gTex = range(GT).map((t) => float("g" + t, W, H));
      gl.bindFramebuffer(gl.FRAMEBUFFER, gl.createFramebuffer());
      gl.activeTexture(gl.TEXTURE0 + unit);
      const attach = range(CT).map((t) => gl.COLOR_ATTACHMENT0 + t);
      for (const t of range(CT)) {
        const o = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, o);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, attach[t], gl.TEXTURE_2D, o, 0);
      }
      gl.drawBuffers(attach);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("çerçeve tamponu eksik");
      gl.viewport(0, 0, W, H);
      this.buf = new Float32Array(W * H * 4);
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      this.renderer = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "bilinmiyor";
    }

    upload(list, src, n) {
      const { gl, W, H, buf } = this, N = W * H;
      list.forEach((tx, t) => {
        for (let i = 0; i < N; i++) for (let q = 0; q < 4; q++) buf[i * 4 + q] = t * 4 + q < n ? src[i * n + t * 4 + q] : 0;
        gl.activeTexture(gl.TEXTURE0 + tx.unit);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, buf);
      });
    }

    // mask: güncellenecek hücreler; delta nx'e eklenir (JS döngüsüyle aynı sözleşme)
    apply(x, g, mask, nx) {
      const { gl, W, H, C, G, CT, buf } = this, N = W * H;
      this.upload(this.xTex, x, C);
      this.upload(this.gTex, g, G);
      gl.activeTexture(gl.TEXTURE0 + this.maskTex.unit);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RED, gl.UNSIGNED_BYTE, mask);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      for (let t = 0; t < CT; t++) {
        gl.readBuffer(gl.COLOR_ATTACHMENT0 + t);
        gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, buf);
        for (let i = 0; i < N; i++) {
          if (!mask[i]) continue;
          for (let q = 0; q < 4 && t * 4 + q < C; q++) nx[i * C + t * 4 + q] += buf[i * 4 + q];
        }
      }
    }

    // Aynı rastgele durumda JS ve GPU adımı karşılaştırılır; sürücü farkı sessizce simülasyonu bozmasın
    matches(world) {
      const { H, W, C, G } = this;
      const t = new root.EvoWorld(world.model, H, W, { energy: false }), x0 = t.x.slice(), g0 = t.g.slice();
      for (let y = 40; y < 80; y++) for (let xx = 40; xx < 80; xx++) {
        const i = y * W + xx;
        for (let c = 0; c < C; c++) x0[i * C + c] = Math.random() - 0.3;
        for (let q = 0; q < G; q++) g0[i * G + q] = 2 * Math.random() - 1;
      }
      const run = (gpu) => { t.x = x0.slice(); t.g = g0.slice(); t.gpu = gpu; t.step(new Uint8Array(H * W).fill(1)); return t.x; };
      const a = run(null), b = run(this);
      let e = 0;
      for (let i = 0; i < a.length; i++) e = Math.max(e, Math.abs(a[i] - b[i]));
      this.error = e;
      return e < 1e-3;
    }

    // Yazılım GL'i (SwiftShader, llvmpipe) CPU'yu daha çok yorar; o durumda JS yolu kullanılır
    static create(world) {
      try {
        const g = new EvoGPU(world);
        if (/swiftshader|llvmpipe|software/i.test(g.renderer) || !g.matches(world)) return null;
        return g;
      } catch (e) {
        console.warn("GPU kullanılamıyor:", e.message);
        return null;
      }
    }
  }

  root.EvoGPU = EvoGPU;
})(typeof window !== "undefined" ? window : globalThis);
