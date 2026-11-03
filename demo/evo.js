// Evrimsel NCA ekosistemi çalışma zamanı: nca/evo.py + nca/world.py ile aynı fizik, saf JavaScript.
// Durum HWC: x [R,G,B,A,V,gizli...] (C kanal), g genom (G), E enerji, R kaynak.
// Algılama indeksi c*K + k, ardından G genom girdisi (Python: torch.cat([p, g], 1)).

(function (root) {
  "use strict";
  const ALIVE = 0.1, VCH = 4;
  const KERNELS = {
    identity: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    sobel_x: [-1, 0, 1, -2, 0, 2, -1, 0, 1].map((v) => v / 8),
    sobel_y: [-1, -2, -1, 0, 0, 0, 1, 2, 1].map((v) => v / 8),
    laplace: [1, 2, 1, 2, -12, 2, 1, 2, 1].map((v) => v / 16),
  };

  function gauss(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  class EvoWorld {
    constructor(model, H, W, world = {}) {
      this.H = H;
      this.W = W;
      this.N = H * W;
      this.setModel(model);
      this.p = Object.assign(
        { uptake: 0.15, cost: 0.006, birth_cost: 0.3, regrow: 0.004, r_diffuse: 0.25, e_share: 0.5, e_share_iters: 4, e_max: 1.0, mu: 0.15, g_clip: 3.5, min_viable: 10, seed_grace: 150, self_tol: 0.5, energy: true, cut_fade: 3, drift_every: 4, cut_release: 1 },
        world
      );
      this.reset();
    }

    setModel(model) {
      this.model = model;
      this.C = model.channels;
      this.G = model.n_genome;
      this.HD = model.hidden_dim;
      this.kernels = model.filters.map((f) => KERNELS[f]);
      this.K = this.kernels.length;
      this.IN = this.C * this.K + this.G;
      this.w1 = Float32Array.from(model.w1);
      this.b1 = Float32Array.from(model.b1);
      this.w2 = Float32Array.from(model.w2);
      this.fireRate = model.fire_rate;
      this.ph = model.physics;
      this.senseV = model.sense_v !== false;
      this.nb = new Float32Array(9 * this.C);
      this.pin = new Float32Array(this.IN);
      this.h = new Float32Array(this.HD);
    }

    reset() {
      const { N, C, G } = this;
      this.x = new Float32Array(N * C);
      this.g = new Float32Array(N * G);
      this.E = new Float32Array(N);
      this.R = new Float32Array(N).fill(1);
      this.tag = new Float32Array(N); // kalıtsal kimlik etiketi (kural görmez)
      this.steps = 0;
      this.planted = [];
      this.drifts = [];
    }

    // 3x3 toplam (sıfır dolgulu), tek kanallı Float32Array
    box3(src, out) {
      const { H, W } = this;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          let s = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx >= 0 && xx < W) s += src[yy * W + xx];
            }
          }
          out[y * W + x] = s;
        }
      return out;
    }

    alpha(x, i) {
      return x[i * this.C + 3];
    }

    life(x, out) {
      const { H, W, C } = this;
      for (let y = 0; y < H; y++)
        for (let xx = 0; xx < W; xx++) {
          let a = 0;
          for (let dy = -1; dy <= 1 && !a; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const x2 = xx + dx;
              if (x2 >= 0 && x2 < W && x[(yy * W + x2) * C + 3] > ALIVE) { a = 1; break; }
            }
          }
          out[y * W + xx] = a;
        }
      return out;
    }

    // fire: null ise Bernoulli; rand: rastgele kaynağı
    step(fire = null, rand = Math.random) {
      const { H, W, N, C, G, K, IN, HD, kernels, w1, b1, w2, nb, pin, h, ph } = this;
      const x = this.x, g = this.g;
      const was = new Float32Array(N);
      for (let i = 0; i < N; i++) was[i] = x[i * C + 3] > ALIVE ? 1 : 0;
      const pre = this.life(x, new Uint8Array(N));
      const nx = x.slice();
      if (this.gpu) {
        const mask = new Uint8Array(N);
        for (let i = 0; i < N; i++) mask[i] = pre[i] && (fire ? fire[i] : rand() <= this.fireRate) ? 255 : 0;
        this.gpu.apply(x, g, mask, nx);
      } else for (let y = 0; y < H; y++)
        for (let xx = 0; xx < W; xx++) {
          const cell = y * W + xx;
          if (!pre[cell]) continue;
          if (fire ? !fire[cell] : rand() > this.fireRate) continue;
          for (let j = 0, dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            for (let dx = -1; dx <= 1; dx++, j++) {
              const x2 = xx + dx, o = j * C;
              if (yy < 0 || yy >= H || x2 < 0 || x2 >= W) for (let c = 0; c < C; c++) nb[o + c] = 0;
              else {
                const s = (yy * W + x2) * C;
                for (let c = 0; c < C; c++) nb[o + c] = x[s + c];
              }
              if (!this.senseV) nb[o + VCH] = 0;
            }
          }
          for (let c = 0; c < C; c++)
            for (let k = 0; k < K; k++) {
              const ker = kernels[k];
              let v = 0;
              for (let j = 0; j < 9; j++) if (ker[j] !== 0) v += ker[j] * nb[j * C + c];
              pin[c * K + k] = v;
            }
          for (let q = 0; q < G; q++) pin[C * K + q] = g[cell * G + q];
          for (let j = 0; j < HD; j++) {
            let v = b1[j];
            const row = j * IN;
            for (let i = 0; i < IN; i++) v += w1[row + i] * pin[i];
            h[j] = v > 0 ? v : 0;
          }
          for (let c = 0; c < C; c++) {
            if (c === VCH) continue; // kural potansiyeli değiştiremez
            let v = 0;
            const row = c * HD;
            for (let j = 0; j < HD; j++) if (h[j] !== 0) v += w2[row + j] * h[j];
            nx[cell * C + c] += v;
          }
        }
      const post = this.life(nx, new Uint8Array(N));
      const af = new Float32Array(N), died = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        if (!(pre[i] && post[i])) nx.fill(0, i * C, i * C + C);
        af[i] = nx[i * C + 3] > ALIVE ? 1 : 0;
        died[i] = was[i] && !af[i] ? 1 : 0;
      }
      // potansiyel
      const V = new Float32Array(N);
      const bd = this.box3(died, new Float32Array(N));
      for (let i = 0; i < N; i++) V[i] = nx[i * C + VCH] * ph.v_decay + (ph.v_gain * bd[i]) / 8;
      const VA = new Float32Array(N);
      for (let i = 0; i < N; i++) VA[i] = V[i] * af[i];
      const nbv = this.box3(VA, new Float32Array(N)), nbn = this.box3(af, new Float32Array(N));
      for (let i = 0; i < N; i++) {
        let v = V[i] + ph.v_diffuse * (nbv[i] / Math.max(nbn[i], 1) - V[i]);
        v = Math.min(Math.max(v * af[i], 0), ph.v_max);
        nx[i * C + VCH] = v;
      }
      // kalıtım (+ öz/öz-olmayan: komşu genom yayılımı)
      const cnt = this.box3(was, new Float32Array(N));
      const ng = new Float32Array(N * G), tmp = new Float32Array(N), sum = new Float32Array(N);
      // dünya kipinde homojenleşme ayrıca, yalnız aynı bireyin hücreleriyle yapılır
      const kappa = this.p.energy ? 0 : ph.kappa;
      for (let q = 0; q < G; q++) {
        for (let i = 0; i < N; i++) tmp[i] = g[i * G + q] * was[i];
        this.box3(tmp, sum);
        for (let i = 0; i < N; i++) {
          const mean = sum[i] / Math.max(cnt[i], 1);
          const born = af[i] && !was[i];
          const v = born ? mean : g[i * G + q] + kappa * (mean - g[i * G + q]) * was[i];
          ng[i * G + q] = v * af[i];
        }
      }
      if (this.p.energy) {
        // öz/öz-olmayan: iki ayrı bireyin (farklı kimlik etiketi) arasına doğum yok
        const T = this.tag, tw = new Float32Array(N), tw2 = new Float32Array(N);
        for (let i = 0; i < N; i++) { tw[i] = T[i] * was[i]; tw2[i] = tw[i] * T[i]; }
        const st = this.box3(tw, new Float32Array(N)), st2 = this.box3(tw2, new Float32Array(N));
        const tol2 = this.p.self_tol * this.p.self_tol, nt = new Float32Array(N);
        for (let i = 0; i < N; i++) {
          const c = Math.max(cnt[i], 1), m = st[i] / c;
          if (af[i] && !was[i] && st2[i] / c - m * m > tol2) { nx.fill(0, i * C, i * C + C); ng.fill(0, i * G, i * G + G); af[i] = 0; }
          nt[i] = af[i] ? (was[i] ? T[i] : m) : 0;
        }
        this.tag = nt;
        if (ph.kappa > 0) this.homogenize(ng, nt, af);
      }
      this.x = nx;
      this.g = ng;
      if (this.p.energy) this.ecology(was, af, rand);
      this.drift();
      this.steps++;
    }

    // genomu yalnız aynı kimlik etiketli canlı komşuların ortalamasına doğru çeker (yerinde)
    homogenize(g, tag, af) {
      const { H, W, N, G, ph } = this;
      const src = g.slice();
      const m = new Float32Array(G);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const i = y * W + x;
          if (!af[i]) continue;
          m.fill(0);
          let n = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            const j = yy * W + xx;
            if (!af[j] || Math.abs(tag[j] - tag[i]) >= 1e-4) continue;
            for (let q = 0; q < G; q++) m[q] += src[j * G + q];
            n++;
          }
          for (let q = 0; q < G; q++) g[i * G + q] = src[i * G + q] + ph.kappa * (m[q] / Math.max(n, 1) - src[i * G + q]);
        }
    }

    ecology(was, af, rand) {
      const { N, C, G, p, ph } = this;
      const x = this.x, g = this.g, R = this.R;
      const E = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const take = p.uptake * R[i] * af[i];
        R[i] -= take;
        E[i] = this.E[i] + take - p.cost * af[i];
      }
      const EW = new Float32Array(N);
      for (let i = 0; i < N; i++) EW[i] = E[i] * was[i];
      const se = this.box3(EW, new Float32Array(N)), sw = this.box3(was, new Float32Array(N));
      for (let i = 0; i < N; i++) if (af[i] && !was[i]) E[i] = se[i] / Math.max(sw[i], 1) - p.birth_cost;
      // gövde içinde hızlı paylaşım: canlı bütün olarak açlık çeker
      const na = this.box3(af, new Float32Array(N)), EA = new Float32Array(N), sa = new Float32Array(N);
      for (let k = 0; k < p.e_share_iters; k++) {
        for (let i = 0; i < N; i++) EA[i] = E[i] * af[i];
        this.box3(EA, sa);
        for (let i = 0; i < N; i++) E[i] += p.e_share * (sa[i] / Math.max(na[i], 1) - E[i]) * af[i];
      }
      const starving = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let e = Math.min(E[i] * af[i], p.e_max);
        if (af[i] && e <= 0) {
          starving[i] = 1;
          x.fill(0, i * C, i * C + C);
          g.fill(0, i * G, i * G + G);
          this.tag[i] = 0;
          e = 0;
        }
        E[i] = e;
      }
      this.E = E;
      this.wound(starving);
      // kaynak yenilenmesi ve yayılması (kenarda yalnız içerideki komşuların ortalaması)
      const Rn = new Float32Array(N);
      for (let i = 0; i < N; i++) Rn[i] = R[i] + p.regrow * (1 - R[i]);
      const sr = this.box3(Rn, new Float32Array(N));
      const ones = this.box3(new Float32Array(N).fill(1), new Float32Array(N));
      for (let i = 0; i < N; i++) R[i] = Math.min(1, Math.max(0, Rn[i] + p.r_diffuse * (sr[i] / ones[i] - Rn[i])));
      // potansiyele bağlı mutasyon (5x5 düzgünleştirilmiş gürültü)
      if (p.mu > 0) this.mutate(rand);
    }

    wound(removed) {
      const { N, C, ph } = this;
      let any = false;
      for (let i = 0; i < N; i++) if (removed[i]) { any = true; break; }
      if (!any) return;
      const b = this.box3(removed, new Float32Array(N));
      for (let i = 0; i < N; i++) {
        const o = i * C;
        const alive = this.x[o + 3] > ALIVE ? 1 : 0;
        this.x[o + VCH] = Math.min(Math.max((this.x[o + VCH] + (ph.v_gain * b[i]) / 8) * alive, 0), ph.v_max);
      }
    }

    mutate(rand) {
      const { H, W, N, C, G, p } = this;
      let vmax = 0;
      for (let i = 0; i < N; i++) vmax = Math.max(vmax, this.x[i * C + VCH]);
      if (vmax <= 0) return;
      // yalnız potansiyelli hücrelerin 2 piksel çevresinde gürültü üret (verim için)
      const xi = new Float32Array(N);
      for (let q = 0; q < G; q++) {
        xi.fill(0);
        for (let i = 0; i < N; i++) if (this.x[i * C + VCH] > 0) {
          const y = (i / W) | 0, x0 = i % W;
          for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
            const yy = y + dy, xx = x0 + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W && xi[yy * W + xx] === 0) xi[yy * W + xx] = gauss(rand) || 1e-9;
          }
        }
        for (let i = 0; i < N; i++) {
          const v = this.x[i * C + VCH];
          if (v <= 0) continue;
          const y = (i / W) | 0, x0 = i % W;
          let s = 0;
          for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
            const yy = y + dy, xx = x0 + dx;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += xi[yy * W + xx];
          }
          // Python ile aynı ölçek: 5x5 ortalama x 5 = toplam / 5 (std 1)
          this.g[i * G + q] = Math.max(-p.g_clip, Math.min(p.g_clip, this.g[i * G + q] + (p.mu * v * s) / 5));
        }
      }
    }

    // yaşayamayacak kadar küçük döküntüyü temizle (koruma süresindeki tohumlar hariç)
    cleanDebris() {
      const { C, G, p } = this;
      this.planted = (this.planted || []).filter((q) => this.steps - q.t < p.seed_grace);
      const { lab, comps } = this.components(1);
      const keep = new Set(this.planted.map((q) => lab[q.i]));
      for (const c of comps) {
        if (c.area >= p.min_viable || keep.has(c.id)) continue;
        for (const i of c.cells) { this.x.fill(0, i * C, i * C + C); this.g.fill(0, i * G, i * G + G); this.E[i] = 0; this.tag[i] = 0; }
      }
    }

    plant(cx, cy, genome) {
      const { W, C, G } = this;
      const i = cy * W + cx;
      (this.planted = this.planted || []).push({ i, t: this.steps });
      this.x[i * C + 3] = 1;
      for (let c = VCH + 1; c < C; c++) this.x[i * C + c] = 1;
      for (let q = 0; q < G; q++) this.g[i * G + q] = genome[q];
      this.E[i] = this.p.e_max;
      this.tag[i] = 1 + 9 * Math.random();
    }

    randomGenome(rand = Math.random) {
      return Array.from({ length: this.G }, () => gauss(rand));
    }

    erase(cx, cy, r) {
      const { H, W, N, C, G } = this;
      const removed = new Float32Array(N), f = this.p.cut_fade, R = r + f;
      for (let y = Math.max(0, Math.floor(cy - R)); y <= Math.min(H - 1, Math.ceil(cy + R)); y++)
        for (let x = Math.max(0, Math.floor(cx - R)); x <= Math.min(W - 1, Math.ceil(cx + R)); x++) {
          const dist = Math.hypot(x - cx, y - cy);
          if (dist > r && dist < R) this.fade(y * W + x, (dist - r) / f);
          else if (dist <= r) {
            const i = y * W + x;
            if (this.x[i * C + 3] > ALIVE) removed[i] = 1;
            this.x.fill(0, i * C, i * C + C);
            this.g.fill(0, i * G, i * G + G);
            this.E[i] = 0;
            this.tag[i] = 0;
          }
        }
      this.wound(removed);
    }

    // kenarı birden kesmek yerine alfayı azaltır; sönen hücreler bir sonraki adımda yara sayılır
    fade(i, d) {
      const k = d * d * (3 - 2 * d), o = i * this.C;
      for (let c = 0; c < 4; c++) this.x[o + c] *= k;
    }

    // Bağlı bileşenler (8-komşuluk): etiket dizisi ve bileşen listesi
    components(minArea = 20) {
      const { H, W, N, C } = this;
      const lab = new Int32Array(N), comps = [], stack = [];
      for (let i = 0; i < N; i++) {
        if (lab[i] || this.x[i * C + 3] <= ALIVE) continue;
        const tv = Math.round(this.tag[i] * 1000);
        const id = comps.length + 1, cells = [];
        lab[i] = id;
        stack.push(i);
        while (stack.length) {
          const c = stack.pop();
          cells.push(c);
          const y = (c / W) | 0, x = c % W;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            const j = yy * W + xx;
            if (!lab[j] && this.x[j * C + 3] > ALIVE && Math.round(this.tag[j] * 1000) === tv) { lab[j] = id; stack.push(j); }
          }
        }
        comps.push({ id, cells, area: cells.length });
      }
      return { lab, comps: comps.filter((c) => c.area >= minArea) };
    }

    // Doğru boyunca kes (bandı sil, kenarı söndür); parçalar normal yönünde adım adım uzaklaşır.
    // cells: kesilecek hücreler (null: tüm canlılar), (x0,y0)-(x1,y1) doğru, gap: null ise boy/2+3
    cut(x0, y0, x1, y1, gap = null, cells = null) {
      const { W, H, N, C, G } = this;
      let nx = -(y1 - y0), ny = x1 - x0;
      const len = Math.hypot(nx, ny);
      if (len < 1e-6) return;
      nx /= len; ny /= len;
      if (!cells) {
        cells = [];
        for (let i = 0; i < N; i++) if (this.x[i * C + 3] > ALIVE) cells.push(i);
      }
      const proj = (i) => ((i % W) + 0.5 - x0) * nx + (((i / W) | 0) + 0.5 - y0) * ny;
      let pmin = Infinity, pmax = -Infinity;
      for (const i of cells) { const v = proj(i); pmin = Math.min(pmin, v); pmax = Math.max(pmax, v); }
      const d = gap === null ? Math.ceil((pmax - pmin) / 2 + 3) : Math.round(gap / 2);
      const removed = new Float32Array(N), A = [], B = [];
      for (const i of cells) {
        const v = proj(i);
        const dist = Math.abs(v);
        if (dist < 1) { removed[i] = 1; continue; }
        if (dist < 1 + this.p.cut_fade) this.fade(i, (dist - 1) / this.p.cut_fade);
        (v > 0 ? A : B).push(i);
      }
      for (let i = 0; i < N; i++) if (removed[i]) { this.x.fill(0, i * C, i * C + C); this.g.fill(0, i * G, i * G + G); this.E[i] = 0; this.tag[i] = 0; }
      this.wound(removed);
      const nt = 1 + 9 * Math.random(); // B parçası yeni birey
      for (const i of B) if (this.tag[i] !== 0) this.tag[i] = nt;
      for (const tag of new Set(A.map((i) => this.tag[i]))) if (tag) this.drifts.push({ tag, nx, ny, left: d, t: 0 });
      this.drifts.push({ tag: nt, nx: -nx, ny: -ny, left: d, t: 0 });
      // parçalar artık tükenmiş zeminde yenilenir: kesilen doku altındaki kaynağı serbest bırakır
      for (const i of cells) this.R[i] = Math.max(this.R[i], this.p.cut_release);
    }

    // Kesilen parçayı bir hücre iter; kural aralarda yarayı iyileştirir. Yol kapalıysa parça durur, ezilmez.
    drift() {
      if (!this.drifts.length || this.steps % this.p.drift_every) return;
      const { W, H, N, C, G } = this;
      this.drifts = this.drifts.filter((q) => {
        const sx = Math.round(q.nx * (q.t + 1)) - Math.round(q.nx * q.t), sy = Math.round(q.ny * (q.t + 1)) - Math.round(q.ny * q.t);
        q.t++;
        const cells = [];
        for (let i = 0; i < N; i++) if (Math.abs(this.tag[i] - q.tag) < 1e-3) cells.push(i);
        for (const i of cells) {
          const ty = ((i / W) | 0) + sy, tx = (i % W) + sx;
          if (ty < 0 || ty >= H || tx < 0 || tx >= W) return false;
          const j = ty * W + tx;
          if (this.x[j * C + 3] > ALIVE && Math.abs(this.tag[j] - q.tag) >= 1e-3) return false;
        }
        // büyüme cephesi (alfa < eşik, gizli durum dolu) da taşınır: kural onu gövdeyle hizalı görmeli
        const seen = new Set(cells);
        for (const i of cells) {
          const y = (i / W) | 0, x0 = i % W;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const yy = y + dy, xx = x0 + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            const j = yy * W + xx;
            if (this.x[j * C + 3] <= ALIVE) seen.add(j);
          }
        }
        const saved = [...seen].map((i) => [i, this.x.slice(i * C, i * C + C), this.g.slice(i * G, i * G + G), this.E[i], this.tag[i]]);
        for (const [i] of saved) { this.x.fill(0, i * C, i * C + C); this.g.fill(0, i * G, i * G + G); this.E[i] = 0; this.tag[i] = 0; }
        for (const [i, xs, gs, e, tg] of saved) {
          const ty = ((i / W) | 0) + sy, tx = (i % W) + sx;
          if (ty < 0 || ty >= H || tx < 0 || tx >= W) continue;
          const j = ty * W + tx;
          if (this.x[j * C + 3] > ALIVE) continue; // başka bireyin gövdesi üstüne cephe yazılmaz
          this.x.set(xs, j * C);
          this.g.set(gs, j * G);
          this.E[j] = e;
          this.tag[j] = tg;
        }
        return --q.left > 0 && cells.length > 0;
      });
    }

    // Genomdan renk (soy görünümü): sabit rastgele izdüşüm
    genomeColor(i, proj) {
      const { G } = this;
      const out = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let q = 0; q < G; q++) v += proj[c * G + q] * this.g[i * G + q];
        out[c] = 0.5 + 0.5 * Math.tanh(v);
      }
      return out;
    }

    render(img, mode, bg = [0.96, 0.95, 0.91], proj = null) {
      const { N, C, x } = this;
      const d = img.data;
      for (let i = 0; i < N; i++) {
        const o = i * C, q = i * 4;
        const a = Math.min(1, Math.max(0, x[o + 3]));
        let r, gg, b;
        if (mode === "rgb") {
          r = bg[0] * (1 - a) + x[o]; gg = bg[1] * (1 - a) + x[o + 1]; b = bg[2] * (1 - a) + x[o + 2];
        } else if (mode === "potential") {
          const v = Math.min(1, x[o + VCH] / 0.5);
          const base = a > ALIVE ? 0.82 : 0.97;
          r = base + (1 - base) * v; gg = base * (1 - 0.75 * v); b = base * (1 - 0.9 * v);
          if (a > ALIVE && v < 0.02) { r = 0.75; gg = 0.74; b = 0.72; }
        } else if (mode === "lineage") {
          if (a > ALIVE) [r, gg, b] = this.genomeColor(i, proj);
          else { const R = this.R[i]; r = gg = b = 0.97 - 0.12 * (1 - R); }
        } else if (mode === "resource") {
          const R = this.R[i];
          r = 0.98 - 0.55 * (1 - R); gg = 0.98 - 0.25 * (1 - R); b = 0.98 - 0.55 * (1 - R);
          if (a > ALIVE) { r *= 0.6; gg *= 0.6; b *= 0.6; }
        }
        d[q] = 255 * Math.min(1, Math.max(0, r));
        d[q + 1] = 255 * Math.min(1, Math.max(0, gg));
        d[q + 2] = 255 * Math.min(1, Math.max(0, b));
        d[q + 3] = 255;
      }
    }
  }

  root.EvoWorld = EvoWorld;
  if (typeof module !== "undefined") module.exports = { EvoWorld };
})(typeof window !== "undefined" ? window : globalThis);
