// NCA çalışma zamanı: nca/model.py ile aynı kural, saf JavaScript.
// Durum düzeni HWC (satır, sütun, kanal). Algılama vektörü indeksi c * K + k (Python ile aynı).
// Yalnız canlı komşuluğu olan ve o adımda ateşlenen hücreler için MLP hesaplanır.

(function (root) {
  "use strict";

  const ALIVE = 0.1;
  const KERNELS = {
    identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    sobel_x: [-1, 0, 1, -2, 0, 2, -1, 0, 1].map((v) => v / 8),
    sobel_y: [-1, -2, -1, 0, 0, 0, 1, 2, 1].map((v) => v / 8),
    laplace: [1, 2, 1, 2, -12, 2, 1, 2, 1].map((v) => v / 16),
  };

  class NCA {
    constructor(model, H, W) {
      this.H = H;
      this.W = W;
      this.state = new Float32Array(H * W * model.channels);
      this.setModel(model);
      this.steps = 0;
    }

    setModel(model) {
      if (this.C !== undefined && model.channels !== this.C) {
        // kanal sayısı farklı model: durumu sıfırla
        this.C = model.channels;
        this.state = new Float32Array(this.H * this.W * this.C);
      }
      this.model = model;
      this.C = model.channels;
      this.HD = model.hidden_dim;
      this.kernels = model.filters.map((f) => KERNELS[f]);
      this.K = this.kernels.length;
      this.IN = this.C * this.K;
      this.w1 = Float32Array.from(model.w1);
      this.b1 = Float32Array.from(model.b1);
      this.w2 = Float32Array.from(model.w2);
      this.fireRate = model.fire_rate;
      this.next = new Float32Array(this.state.length);
      this.pre = new Uint8Array(this.H * this.W);
      this.nb = new Float32Array(9 * this.C);
      this.p = new Float32Array(this.IN);
      this.h = new Float32Array(this.HD);
    }

    lifeMask(s, out) {
      const { H, W, C } = this;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let alive = 0;
          for (let dy = -1; dy <= 1 && !alive; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx < 0 || xx >= W) continue;
              if (s[(yy * W + xx) * C + 3] > ALIVE) {
                alive = 1;
                break;
              }
            }
          }
          out[y * W + x] = alive;
        }
      }
    }

    // fire: null ise Bernoulli(fire_rate); aksi halde 0/1 Uint8Array (test için)
    step(fire = null, rand = Math.random) {
      const { H, W, C, K, IN, HD, kernels, w1, b1, w2, nb, p, h, state, next, pre } = this;
      this.lifeMask(state, pre);
      next.set(state);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const cell = y * W + x;
          if (!pre[cell]) continue;
          if (fire ? !fire[cell] : rand() > this.fireRate) continue;
          // 3x3 komşuluk (sıfır dolgulu)
          for (let j = 0, dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            for (let dx = -1; dx <= 1; dx++, j++) {
              const xx = x + dx;
              const o = j * C;
              if (yy < 0 || yy >= H || xx < 0 || xx >= W) {
                for (let c = 0; c < C; c++) nb[o + c] = 0;
              } else {
                const src = (yy * W + xx) * C;
                for (let c = 0; c < C; c++) nb[o + c] = state[src + c];
              }
            }
          }
          for (let c = 0; c < C; c++) {
            for (let k = 0; k < K; k++) {
              const ker = kernels[k];
              let v = 0;
              for (let j = 0; j < 9; j++) if (ker[j] !== 0) v += ker[j] * nb[j * C + c];
              p[c * K + k] = v;
            }
          }
          for (let j = 0; j < HD; j++) {
            let v = b1[j];
            const row = j * IN;
            for (let i = 0; i < IN; i++) v += w1[row + i] * p[i];
            h[j] = v > 0 ? v : 0;
          }
          const dst = cell * C;
          for (let c = 0; c < C; c++) {
            let v = 0;
            const row = c * HD;
            for (let j = 0; j < HD; j++) if (h[j] !== 0) v += w2[row + j] * h[j];
            next[dst + c] += v;
          }
        }
      }
      // canlılık: önceki ve sonraki maskenin kesişimi
      const post = this._post || (this._post = new Uint8Array(H * W));
      this.lifeMask(next, post);
      for (let cell = 0; cell < H * W; cell++) {
        if (!(pre[cell] && post[cell])) next.fill(0, cell * C, cell * C + C);
      }
      this.state = next;
      this.next = state;
      this.steps++;
    }

    clear() {
      this.state.fill(0);
      this.steps = 0;
    }

    plantSeed(cx, cy) {
      const { W, C } = this;
      const o = (cy * W + cx) * C;
      for (let c = 3; c < C; c++) this.state[o + c] = 1;
    }

    forDisk(cx, cy, r, fn) {
      const { H, W } = this;
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(H - 1, Math.ceil(cy + r)); y++) {
        for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(W - 1, Math.ceil(cx + r)); x++) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) fn(y * W + x);
        }
      }
    }

    erase(cx, cy, r) {
      const C = this.C;
      this.forDisk(cx, cy, r, (cell) => this.state.fill(0, cell * C, cell * C + C));
    }

    // Müdahale: canlı hücrelerde kanalları değerlere ayarlar
    paint(cx, cy, r, channels, values) {
      const C = this.C;
      this.forDisk(cx, cy, r, (cell) => {
        const o = cell * C;
        if (this.state[o + 3] <= ALIVE) return;
        channels.forEach((ch, i) => (this.state[o + ch] = values[i]));
      });
    }

    // (x0,y0)-(x1,y1) doğrusu boyunca kes ve iki tarafı normal yönünde gap/2 kadar uzaklaştır
    cut(x0, y0, x1, y1, gap) {
      const { H, W, C, state } = this;
      let nx = -(y1 - y0), ny = x1 - x0;
      const len = Math.hypot(nx, ny);
      if (len < 1e-6) return;
      nx /= len;
      ny /= len;
      const sx = Math.round((nx * gap) / 2), sy = Math.round((ny * gap) / 2);
      const out = new Float32Array(state.length);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const side = (x + 0.5 - x0) * nx + (y + 0.5 - y0) * ny >= 0 ? 1 : -1;
          const tx = x + side * sx, ty = y + side * sy;
          if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
          const src = (y * W + x) * C, dst = (ty * W + tx) * C;
          for (let c = 0; c < C; c++) out[dst + c] += state[src + c];
        }
      }
      this.state = out;
      this.next = new Float32Array(state.length);
    }

    // Bağlı bileşenler (8-komşuluk) — maske Uint8Array, alanı minArea altındakiler sayılmaz
    static countComponents(mask, H, W, minArea) {
      const lab = new Int32Array(H * W);
      const stack = [];
      let n = 0;
      for (let i = 0; i < H * W; i++) {
        if (!mask[i] || lab[i]) continue;
        let area = 0;
        lab[i] = -1;
        stack.push(i);
        while (stack.length) {
          const c = stack.pop();
          area++;
          const y = (c / W) | 0, x = c % W;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const yy = y + dy, xx = x + dx;
              if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
              const j = yy * W + xx;
              if (mask[j] && !lab[j]) {
                lab[j] = -1;
                stack.push(j);
              }
            }
          }
        }
        if (area >= minArea) n++;
      }
      return n;
    }

    stats(palette) {
      const { H, W, C, state } = this;
      const alive = new Uint8Array(H * W), head = new Uint8Array(H * W);
      let area = 0;
      for (let cell = 0; cell < H * W; cell++) {
        const o = cell * C, a = state[o + 3];
        if (a > ALIVE) {
          alive[cell] = 1;
          area++;
        }
        if (a > 0.5 && palette) {
          let best = 0, bd = Infinity;
          for (let k = 0; k < palette.length; k++) {
            let d = 0;
            for (let c = 0; c < 3; c++) d += (state[o + c] / a - palette[k][c]) ** 2;
            if (d < bd) (bd = d), (best = k);
          }
          if (best === 0 || best === 3) head[cell] = 1; // baş veya göz
        }
      }
      return {
        area,
        individuals: NCA.countComponents(alive, H, W, 20),
        heads: NCA.countComponents(head, H, W, 12),
      };
    }

    // RGBA görüntü: mode 'rgb' | 'alpha' | kanal indeksi (sayı)
    render(img, mode) {
      const { H, W, C, state } = this;
      const d = img.data;
      for (let cell = 0; cell < H * W; cell++) {
        const o = cell * C, q = cell * 4;
        if (mode === "rgb") {
          const a = Math.min(1, Math.max(0, state[o + 3]));
          for (let c = 0; c < 3; c++) d[q + c] = 255 * Math.min(1, Math.max(0, 1 - a + state[o + c]));
        } else {
          const v = mode === "alpha" ? state[o + 3] : state[o + mode];
          const alive = state[o + 3] > ALIVE;
          if (!alive) {
            d[q] = d[q + 1] = d[q + 2] = 245;
          } else {
            // ıraksak renk ölçeği: negatif mavi, pozitif turuncu
            const t = Math.max(-1, Math.min(1, v));
            if (t >= 0) {
              d[q] = 255;
              d[q + 1] = 255 - 130 * t;
              d[q + 2] = 255 - 230 * t;
            } else {
              d[q] = 255 + 200 * t;
              d[q + 1] = 255 + 110 * t;
              d[q + 2] = 255;
            }
          }
        }
        d[q + 3] = 255;
      }
    }
  }

  root.NCA = NCA;
  if (typeof module !== "undefined") module.exports = { NCA };
})(typeof window !== "undefined" ? window : globalThis);
