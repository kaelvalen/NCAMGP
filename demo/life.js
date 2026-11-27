// Yaşayan ekosistem çalışma zamanı: nca/life.py + nca/life_world.py ile aynı fizik, saf JavaScript.
// Dünya sarmal (torus). Alanlar HWC Float32Array: x [R,G,B,m,V,E,gizli...] (C), g genom (G), R kaynak, tag kimlik.
// Kural ileri geçişi forward() (CPU) ya da forwardHook (WebGL, demo/life_gpu.js); fizik ve izleme burada.
// Parite: tests/test_life_js_parity.py. Python'dan bilinçli farklar: otomatik dış kesme yok (kullanıcı keser),
// kesme/silmede besin aynı hücrede R'ye döner, bölünme nedeni sayılmaz.

(function (root) {
  "use strict";
  const ALIVE = 0.1, M = 3, V = 4, E = 5, H0 = 6, H_MAX = 5.0, TAG_SPLIT = 5.5;
  const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

  function gauss(rand) {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // base64 -> ArrayBuffer (tarayıcı atob, node Buffer)
  function b64buf(s) {
    if (typeof atob === "function") {
      const bin = atob(s), u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      return u.buffer;
    }
    const b = Buffer.from(s, "base64");
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
  }

  function f16(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  }

  function decode(s, half) {
    if (!half) return new Float32Array(b64buf(s));
    const u = new Uint16Array(b64buf(s)), out = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) out[i] = f16(u[i]);
    return out;
  }

  class LifeWorld {
    constructor(model, H, W, opts = {}) {
      this.H = H;
      this.W = W;
      this.N = H * W;
      this.model = model;
      this.C = model.channels;
      this.G = model.n_genome;
      this.nh = model.n_hidden;
      this.HD = model.hidden_dim;
      this.NIN = model.n_in;
      this.kern = model.filters;
      this.K = this.kern.length;
      this.IN = this.NIN * this.K;
      this.NOUT = model.n_out;
      this.w1 = Float32Array.from(model.w1);
      this.b1 = Float32Array.from(model.b1);
      this.w2 = Float32Array.from(model.w2);
      this.ob = Float32Array.from(model.out_bias);
      this.Ps = Float32Array.from(model.film_gamma);
      this.Pb = Float32Array.from(model.film_beta);
      this.ph = model.physics;
      this.p = Object.assign(
        { mu: 0.15, mu_base: 0, g_clip: 3.5, min_viable: 10, seed_grace: 150, track_every: 10, min_org_area: 20 },
        model.world || {}, opts);
      this.fireRate = opts.fire_rate !== undefined ? opts.fire_rate : model.fire_rate;
      this.filmScale = opts.film_scale !== undefined ? opts.film_scale : model.film_scale;
      this.rand = opts.rand || Math.random;
      this.forwardHook = null;
      const { N, C, G, nh } = this;
      this.x = new Float32Array(N * C);
      this.g = new Float32Array(N * G);
      this.R = new Float32Array(N).fill(1);
      this.tag = new Float32Array(N);
      this.F = new Float32Array(N);
      this.out = { drgb: new Float32Array(N * 3), dh: new Float32Array(N * nh), grow: new Float32Array(N),
        flow: new Float32Array(N * 9), attack: new Float32Array(N) };
      this.uptake = new Float32Array(N);
      this.eatenAcc = new Float64Array(N);
      this.uptakeAcc = new Float64Array(N);
      // sarmal komşu tablosu: nb[i*9 + (dy+1)*3 + (dx+1)] = (y+dy, x+dx) hücresi
      this.nb = new Int32Array(N * 9);
      for (let y = 0; y < H; y++)
        for (let xx = 0; xx < W; xx++)
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++)
              this.nb[(y * W + xx) * 9 + (dy + 1) * 3 + dx + 1] = ((y + dy + H) % H) * W + ((xx + dx + W) % W);
      this.dirNb = DIRS.map(([dy, dx]) => (dy + 1) * 3 + dx + 1);
      this.dirSrc = DIRS.map(([dy, dx]) => (-dy + 1) * 3 + -dx + 1); // k yönüne giden kaynağın konumu
      this.resetTracking();
    }

    resetTracking() {
      this.t = 0;
      this.orgs = new Map();
      this.nextId = 0;
      this.labels = new Int32Array(this.N);
      this.labelToId = new Map();
      this.orgTag = new Map();
      this.tagOrigin = new Map();
      this.seedTags = [];
      this.history = [];
      this.extinct = false;
      this.lastPred = 0;
    }

    loadState(s) {
      this.x.set(decode(s.x, true));
      this.g.set(decode(s.g, true));
      this.R.set(decode(s.R, true));
      this.tag.set(decode(s.tag, false));
      this.resetTracking();
      this.t = s.t;
      this.eatenAcc.fill(0);
      this.uptakeAcc.fill(0);
      this.track();
    }

    // --- kural (CPU) ---
    forward(out) {
      const { N, C, G, nh, HD, IN, K, NOUT, x, g, R, F, nb, kern, w1, b1, w2, ob, Ps, Pb } = this;
      const s = this.filmScale, NIN = this.NIN;
      const p = new Float32Array(IN), z = new Float32Array(HD), y = new Float32Array(NOUT), val = new Float32Array(9);
      for (let i = 0; i < N; i++) {
        if (!(x[i * C + M] > 0)) { // ölü hücrede kural çıktısı kullanılmaz (ateşleme maskesi m > 0)
          for (let o = 0; o < 9; o++) out.flow[i * 9 + o] = o === 0 ? 1 : 0;
          continue;
        }
        for (let c = 0; c < NIN; c++) {
          for (let j = 0; j < 9; j++) {
            const q = nb[i * 9 + j];
            val[j] = c < C ? x[q * C + c] : c === C ? R[q] : F[q];
          }
          for (let k = 0; k < K; k++) {
            const f = kern[k];
            let a = 0;
            for (let j = 0; j < 9; j++) a += f[j] * val[j];
            p[c * K + k] = a;
          }
        }
        for (let h = 0; h < HD; h++) {
          let a = b1[h];
          const r = h * IN;
          for (let q = 0; q < IN; q++) a += w1[r + q] * p[q];
          if (s) {
            let gs = 0, gb = 0;
            const rg = h * G;
            for (let q = 0; q < G; q++) {
              const gv = g[i * G + q];
              gs += Ps[rg + q] * gv;
              gb += Pb[rg + q] * gv;
            }
            a = a * (1 + s * gs) + s * gb;
          }
          z[h] = a > 0 ? a : 0;
        }
        for (let o = 0; o < NOUT; o++) {
          let a = ob[o];
          const r = o * HD;
          for (let h = 0; h < HD; h++) a += w2[r + h] * z[h];
          y[o] = a;
        }
        for (let c = 0; c < 3; c++) out.drgb[i * 3 + c] = y[c];
        for (let c = 0; c < nh; c++) out.dh[i * nh + c] = y[3 + c];
        out.grow[i] = 1 / (1 + Math.exp(-y[3 + nh]));
        let mx = -Infinity;
        for (let o = 0; o < 9; o++) mx = Math.max(mx, y[4 + nh + o]);
        let sum = 0;
        for (let o = 0; o < 9; o++) sum += out.flow[i * 9 + o] = Math.exp(y[4 + nh + o] - mx);
        for (let o = 0; o < 9; o++) out.flow[i * 9 + o] /= sum;
        out.attack[i] = 1 / (1 + Math.exp(-y[13 + nh]));
      }
    }

    // 3x3 komşulukta başka bireylere ait canlı hücreler: link(i, k)
    link(i, k) {
      const j = this.nb[i * 9 + this.dirNb[k]], C = this.C;
      return this.x[i * C + M] > ALIVE && this.x[j * C + M] > ALIVE && Math.abs(this.tag[j] - this.tag[i]) > 1e-4 ? j : -1;
    }

    foreignMass() {
      const { N, C, x, F } = this;
      for (let i = 0; i < N; i++) {
        let a = 0;
        for (let k = 0; k < 8; k++) {
          const j = this.link(i, k);
          if (j >= 0) a += x[j * C + M];
        }
        F[i] = a;
      }
    }

    // 3x3 sarmal toplam
    box3(src, out) {
      const nb = this.nb;
      for (let i = 0; i < this.N; i++) {
        let s = 0;
        for (let j = 0; j < 9; j++) s += src[nb[i * 9 + j]];
        out[i] = s;
      }
    }

    // --- tek adım (LifeWorld.step) ---
    step() {
      this.foreignMass();
      this.lifeStep();
      this.eat();
      const alive = new Uint8Array(this.N);
      for (let i = 0; i < this.N; i++) alive[i] = this.x[i * this.C + M] > ALIVE;
      if (this.ph.kappa > 0) this.homogenize(alive);
      this.mutate(alive);
      const gc = this.p.g_clip, g = this.g;
      for (let i = 0; i < g.length; i++) g[i] = Math.min(gc, Math.max(-gc, g[i]));
      this.t += 1;
      if (this.t % this.p.track_every === 0) {
        this.retagSplits();
        this.track();
      }
    }

    lifeStep() {
      const { N, C, G, nh, x, g, R, tag, nb, ph, out } = this;
      const NI = 3 + nh + G;
      if (this.forwardHook) this.forwardHook(out);
      else this.forward(out);
      const S = this.scratch || (this.scratch = {
        m: new Float64Array(N), En: new Float64Array(N), Vold: new Float32Array(N), was: new Uint8Array(N),
        flow: new Float64Array(N * 9), inten: new Float64Array(N * NI), tmax: new Float64Array(N), tmin: new Float64Array(N),
        move: new Float64Array(N * 8), stay: new Float64Array(N), m2: new Float64Array(N), E2: new Float64Array(N),
        num: new Float64Array(N * NI), tag2: new Float32Array(N), alive: new Uint8Array(N) });
      const { m, En, Vold, was, flow, inten, tmax, tmin, move, stay, m2, E2, num, tag2, alive } = S;
      move.fill(0);
      for (let i = 0; i < N; i++) {
        const mi = x[i * C + M], Ei = x[i * C + E];
        was[i] = mi > ALIVE;
        Vold[i] = x[i * C + V];
        const fm = this.rand() <= this.fireRate && mi > 0 ? 1 : 0;
        for (let o = 0; o < 9; o++) flow[i * 9 + o] = fm ? out.flow[i * 9 + o] : o === 0 ? 1 : 0;
        const grow = out.grow[i] * fm;
        out.attack[i] *= fm;
        const b = i * NI;
        for (let c = 0; c < 3; c++) inten[b + c] = Math.min(1, Math.max(0, x[i * C + c] + out.drgb[i * 3 + c] * fm));
        for (let c = 0; c < nh; c++)
          inten[b + 3 + c] = Math.min(H_MAX, Math.max(-H_MAX, x[i * C + H0 + c] + out.dh[i * nh + c] * fm));
        for (let c = 0; c < G; c++) inten[b + 3 + nh + c] = g[i * G + c];
        const Ep = Math.max(Ei, 0), dm = ph.r_g * grow * Ep;
        const cmove = Math.min(ph.c_move * mi * (1 - flow[i * 9]), Ep - dm);
        m[i] = mi + dm;
        En[i] = Ei - dm - cmove;
        R[i] += cmove;
      }
      // taşınım (transport): kütle korunumlu, bireyler kaynaşmaz
      for (let i = 0; i < N; i++) {
        let a = -1e9, b = 1e9;
        for (let j = 0; j < 9; j++) {
          const q = nb[i * 9 + j];
          if (m[q] > 0) { a = Math.max(a, tag[q]); b = Math.min(b, tag[q]); }
        }
        tmax[i] = a;
        tmin[i] = b;
      }
      for (let i = 0; i < N; i++) {
        let st = flow[i * 9];
        for (let k = 0; k < 8; k++) {
          const j = nb[i * 9 + this.dirNb[k]];
          const ok = m[j] > 0 ? Math.abs(tag[j] - tag[i]) < 1e-4
            : tmax[j] <= -5e8 || (Math.abs(tmax[j] - tag[i]) < 1e-4 && Math.abs(tmin[j] - tag[i]) < 1e-4);
          const f = flow[i * 9 + 1 + k];
          if (ok) move[i * 8 + k] = f;
          else st += f;
        }
        stay[i] = st;
      }
      for (let i = 0; i < N; i++) {
        let qm = m[i] * stay[i], qe = En[i] * stay[i];
        const b = i * NI;
        for (let c = 0; c < NI; c++) num[b + c] = inten[b + c] * m[i] * stay[i];
        let tg = m[i] * stay[i] > 0 ? tag[i] : 0;
        for (let k = 0; k < 8; k++) {
          const s = nb[i * 9 + this.dirSrc[k]], f = move[s * 8 + k];
          if (f === 0) continue;
          qm += m[s] * f;
          qe += En[s] * f;
          const bs = s * NI;
          for (let c = 0; c < NI; c++) num[b + c] += inten[bs + c] * m[s] * f;
          if (m[s] * f > 0) tg = Math.max(tg, tag[s]);
        }
        m2[i] = qm;
        E2[i] = qe;
        tag2[i] = tg;
      }
      // kaynak alımı, bakım, ölüm
      for (let i = 0; i < N; i++) {
        const u = Math.min(R[i], ph.k_up * R[i] * Math.min(1, Math.max(0, m2[i]))), c = ph.c_m * m2[i];
        R[i] = R[i] - u + c;
        E2[i] = E2[i] + u - c;
        this.uptake[i] = u;
      }
      for (let i = 0; i < N; i++) {
        let lm = false;
        for (let j = 0; j < 9; j++) if (m2[nb[i * 9 + j]] > ALIVE) { lm = true; break; }
        const dead = m2[i] > 0 && (E2[i] <= 0 || !lm);
        const b = i * NI;
        if (dead) {
          R[i] += m2[i] + E2[i];
          x[i * C + M] = 0;
          x[i * C + E] = 0;
          for (let c = 0; c < 3; c++) x[i * C + c] = 0;
          for (let c = 0; c < nh; c++) x[i * C + H0 + c] = 0;
          for (let c = 0; c < G; c++) g[i * G + c] = 0;
          tag[i] = 0;
        } else {
          const w = m2[i] > 1e-8 ? 1 / Math.max(m2[i], 1e-8) : 0;
          x[i * C + M] = m2[i];
          x[i * C + E] = E2[i];
          for (let c = 0; c < 3; c++) x[i * C + c] = num[b + c] * w;
          for (let c = 0; c < nh; c++) x[i * C + H0 + c] = num[b + 3 + c] * w;
          for (let c = 0; c < G; c++) g[i * G + c] = num[b + 3 + nh + c] * w;
          tag[i] = tag2[i];
        }
        alive[i] = x[i * C + M] > ALIVE;
      }
      // yara potansiyeli (update_potential, sarmal)
      const died = new Float32Array(N), af = new Float32Array(N), Vn = new Float32Array(N);
      const b1 = new Float32Array(N), b2 = new Float32Array(N), b3 = new Float32Array(N);
      for (let i = 0; i < N; i++) { died[i] = was[i] && !alive[i] ? 1 : 0; af[i] = alive[i]; }
      this.box3(died, b1);
      for (let i = 0; i < N; i++) Vn[i] = Vold[i] * ph.v_decay + (ph.v_gain * b1[i]) / 8;
      const va = new Float32Array(N);
      for (let i = 0; i < N; i++) va[i] = Vn[i] * af[i];
      this.box3(va, b2);
      this.box3(af, b3);
      for (let i = 0; i < N; i++) {
        const v = Vn[i] + ph.v_diffuse * (b2[i] / Math.max(b3[i], 1) - Vn[i]);
        x[i * C + V] = Math.min(ph.v_max, Math.max(0, v * af[i]));
      }
      // kaynak difüzyonu (sarmal, 4 komşu)
      const R0 = Float64Array.from(R), d = ph.r_diffuse / 4;
      for (let i = 0; i < N; i++) {
        const o = i * 9;
        R[i] = R0[i] + d * (R0[nb[o + 1]] + R0[nb[o + 7]] + R0[nb[o + 3]] + R0[nb[o + 5]] - 4 * R0[i]);
      }
    }

    // --- yeme (eat) ---
    eat() {
      const { N, C, x, R, ph, out } = this;
      const A = new Float64Array(N), frac = new Float64Array(N), perA = new Float64Array(N), lost = new Float64Array(N);
      const links = new Int32Array(N * 8);
      for (let i = 0; i < N; i++) {
        let a = 0;
        for (let k = 0; k < 8; k++) {
          const j = (links[i * 8 + k] = this.link(i, k));
          if (j >= 0) a += out.attack[j];
        }
        A[i] = a;
        frac[i] = Math.min(ph.k_eat * a, 1);
        lost[i] = x[i * C + M] * frac[i];
        const eloss = Math.max(x[i * C + E], 0) * frac[i];
        perA[i] = a > 0 ? (lost[i] + eloss) / Math.max(a, 1e-8) : 0;
      }
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let k = 0; k < 8; k++) {
          const j = links[i * 8 + k];
          if (j >= 0) s += perA[j];
        }
        const gain = out.attack[i] * s, mi = x[i * C + M];
        let Ei = x[i * C + E] - Math.max(x[i * C + E], 0) * frac[i] + ph.eta * gain;
        R[i] += (1 - ph.eta) * gain;
        const cost = mi > ALIVE ? Math.min(ph.c_attack * out.attack[i], Math.max(Ei, 0)) : 0;
        Ei -= cost;
        R[i] += cost;
        x[i * C + E] = Ei;
        x[i * C + M] = mi - lost[i];
        x[i * C + V] = Math.min(x[i * C + V] + ph.v_gain * frac[i], ph.v_max);
        this.eatenAcc[i] += ph.eta * gain;
        this.uptakeAcc[i] += this.uptake[i];
      }
    }

    // aynı kimlikli canlı komşuların genom ortalamasına çekim (Python: sıfır dolgulu, kenardan sarmalanmaz)
    homogenize(alive) {
      const { H, W, G, g, tag } = this, kappa = this.ph.kappa;
      const g0 = Float32Array.from(g);
      for (let y = 0; y < H; y++)
        for (let xx = 0; xx < W; xx++) {
          const i = y * W + xx;
          if (!alive[i]) continue;
          const idx = [];
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              const yy = y + dy, x2 = xx + dx;
              if (yy < 0 || yy >= H || x2 < 0 || x2 >= W) continue;
              const j = yy * W + x2;
              if (alive[j] && Math.abs(tag[j] - tag[i]) < 1e-4) idx.push(j);
            }
          const n = Math.max(idx.length, 1);
          for (let c = 0; c < G; c++) {
            let s = 0;
            for (const j of idx) s += g0[j * G + c];
            g[i * G + c] = g0[i * G + c] + kappa * (s / n - g0[i * G + c]);
          }
        }
    }

    // yara potansiyeline bağlı mutasyon: g += mu (V + base) xi, xi 5x5 kutu toplamı / 5 (sıfır dolgulu)
    mutate(alive) {
      const { H, W, N, C, G, g, x } = this, mu = this.p.mu, base = this.p.mu_base;
      if (mu <= 0) return;
      let vmax = -Infinity;
      for (let i = 0; i < N; i++) vmax = Math.max(vmax, x[i * C + V] + base);
      if (vmax <= 0) return;
      // shortcut: gürültü önceden üretilmiş 2^20'lik gaussian tablosundan rastgele kaymayla okunur (Box-Muller
      // adım başına 1,2 M çağrı, demoyu 115 ms yavaşlatıyordu); ardışık adımların alanları kısmen kaydırılmış
      // kopya olabilir, görsel demo için yeterli; istatistik deney gerekirse doğrudan gauss'a dön
      if (!this.gtab) { this.gtab = new Float32Array(1 << 20); for (let i = 0; i < this.gtab.length; i++) this.gtab[i] = gauss(this.rand); }
      const tab = this.gtab, TM = tab.length - 1, xi = this._xi || (this._xi = new Float32Array(N)), hs = this._hs || (this._hs = new Float32Array(N));
      for (let c = 0; c < G; c++) {
        const off = Math.floor(this.rand() * tab.length);
        for (let i = 0; i < N; i++) xi[i] = tab[(off + i) & TM];
        for (let y = 0; y < H; y++)
          for (let xx = 0; xx < W; xx++) {
            let s = 0;
            for (let d = -2; d <= 2; d++) if (xx + d >= 0 && xx + d < W) s += xi[y * W + xx + d];
            hs[y * W + xx] = s;
          }
        for (let y = 0; y < H; y++)
          for (let xx = 0; xx < W; xx++) {
            const i = y * W + xx;
            if (!alive[i]) continue;
            let s = 0;
            for (let d = -2; d <= 2; d++) if (y + d >= 0 && y + d < H) s += hs[(y + d) * W + xx];
            g[i * G + c] += mu * (x[i * C + V] + base) * (s / 5);
          }
      }
    }

    // --- kimlik ve izleme ---
    key(i) {
      return Math.round(this.tag[i] * 1000);
    }

    newTag(like) {
      const lo = like === undefined ? 1 : like >= TAG_SPLIT ? TAG_SPLIT : 1;
      const hi = like === undefined ? 10 : like >= TAG_SPLIT ? 10 : TAG_SPLIT;
      return lo + this.rand() * (hi - lo);
    }

    // canlı hücrelerin etiket başına sarmal 8-bağlantılı bileşenleri; sıra: etiket anahtarı, sonra ilk hücre (raster)
    segment() {
      const { N, C, x, nb } = this;
      const lab = new Int32Array(N), comps = [], stack = [];
      for (let i0 = 0; i0 < N; i0++) {
        if (lab[i0] || !(x[i0 * C + M] > ALIVE)) continue;
        const k = this.key(i0), cells = [i0];
        lab[i0] = -1;
        stack.push(i0);
        while (stack.length) {
          const i = stack.pop();
          for (let j = 0; j < 9; j++) {
            const q = nb[i * 9 + j];
            if (!lab[q] && x[q * C + M] > ALIVE && this.key(q) === k) {
              lab[q] = -1;
              cells.push(q);
              stack.push(q);
            }
          }
        }
        cells.sort((a, b) => a - b);
        comps.push({ key: k, cells });
      }
      comps.sort((a, b) => a.key - b.key || a.cells[0] - b.cells[0]);
      comps.forEach((c, n) => { for (const i of c.cells) lab[i] = n + 1; });
      return { lab, comps };
    }

    retagSplits() {
      const byTag = new Map();
      for (const c of this.segment().comps)
        if (c.cells.length >= this.p.min_viable) {
          if (!byTag.has(c.key)) byTag.set(c.key, []);
          byTag.get(c.key).push(c);
        }
      for (const [tv, cs] of byTag) {
        cs.sort((a, b) => b.cells.length - a.cells.length);
        for (const c of cs.slice(1)) {
          const nt = this.newTag(tv / 1000);
          for (const i of c.cells) this.tag[i] = nt;
          this.tagOrigin.set(this.key(c.cells[0]), tv);
        }
      }
    }

    centroid(cells) {
      const { H, W } = this;
      let sy = 0, cy = 0, sx = 0, cx = 0;
      for (const i of cells) {
        const a = (2 * Math.PI * Math.floor(i / W)) / H, b = (2 * Math.PI * (i % W)) / W;
        sy += Math.sin(a); cy += Math.cos(a); sx += Math.sin(b); cx += Math.cos(b);
      }
      const wrap = (v, n) => ((v % n) + n) % n;
      return [wrap((Math.atan2(sy, cy) * H) / (2 * Math.PI), H), wrap((Math.atan2(sx, cx) * W) / (2 * Math.PI), W)];
    }

    tdist(a, b) {
      const d = (u, n) => ((u + n / 2) % n + n) % n - n / 2;
      return Math.hypot(d(a[0] - b[0], this.H), d(a[1] - b[1], this.W));
    }

    track() {
      const { C, G, x, g, p } = this;
      const prevCent = new Map();
      for (const o of this.orgs.values()) if (o.alive) prevCent.set(o.id, o.centroid);
      const tagOwner = new Map();
      for (const [oid, k] of this.orgTag) tagOwner.set(k, oid);
      const { lab, comps } = this.segment();
      // döküntü temizliği (koruma süresindeki tohumlar hariç): besin aynı hücrede R'ye döner
      const grace = new Set(this.seedTags.filter(([, t0]) => this.t - t0 < p.seed_grace).map(([k]) => k));
      this.seedTags = this.seedTags.filter(([, t0]) => this.t - t0 < p.seed_grace);
      for (const c of comps)
        if (c.cells.length < p.min_viable && !grace.has(c.key)) {
          for (const i of c.cells) this.clear(i);
          c.cleaned = true;
        }
      const order = comps.map((c, n) => n).sort((a, b) => comps[b].cells.length - comps[a].cells.length);
      const newMap = new Map(), claimed = new Set();
      for (const n of order) {
        const c = comps[n], l = n + 1;
        if (c.cleaned) continue;
        const cnt = new Map();
        for (const i of c.cells) {
          const v = this.labels[i];
          if (v > 0) cnt.set(v, (cnt.get(v) || 0) + 1);
        }
        let best = -1, bc = 0;
        for (const [v, k] of cnt) if (k > bc || (k === bc && v < best)) { best = v; bc = k; }
        let parent = best > 0 ? this.labelToId.get(best) : undefined;
        if (parent === undefined) {
          const own = tagOwner.get(c.key);
          parent = own !== undefined ? own : tagOwner.get(this.tagOrigin.get(c.key));
        }
        let oid;
        if (parent !== undefined && !claimed.has(parent)) {
          oid = parent;
          claimed.add(oid);
        } else {
          oid = this.nextId++;
          const par = parent === undefined ? -1 : parent;
          const pr = this.orgs.get(par);
          this.orgs.set(oid, { id: oid, parent: par, born: this.t, area: 0, centroid: [0, 0], genome: [], speed: 0,
            eaten: 0, uptake: 0, alive: true, died: -1, generation: pr ? pr.generation + 1 : 0 });
          if (pr) pr.n_children = (pr.n_children || 0) + 1;
        }
        const o = this.orgs.get(oid), gm = new Float64Array(G);
        let ea = 0, up = 0;
        for (const i of c.cells) {
          for (let q = 0; q < G; q++) gm[q] += g[i * G + q];
          ea += this.eatenAcc[i];
          up += this.uptakeAcc[i];
        }
        o.area = c.cells.length;
        o.genome = Array.from(gm, (v) => v / c.cells.length);
        o.centroid = this.centroid(c.cells);
        o.speed = prevCent.has(oid) ? this.tdist(o.centroid, prevCent.get(oid)) / p.track_every : 0;
        o.eaten = ea;
        o.uptake = up;
        newMap.set(l, oid);
      }
      const live = new Set(newMap.values());
      for (const o of this.orgs.values()) if (o.alive && !live.has(o.id)) { o.alive = false; o.died = this.t; }
      for (const c of comps) if (c.cleaned) for (const i of c.cells) lab[i] = 0;
      this.labels = lab;
      this.labelToId = newMap;
      this.orgTag = new Map();
      for (const [l, oid] of newMap) this.orgTag.set(oid, comps[l - 1].key);
      let ea = 0, up = 0;
      for (let i = 0; i < this.N; i++) { ea += this.eatenAcc[i]; up += this.uptakeAcc[i]; }
      this.lastPred = ea + up > 0 ? ea / (ea + up) : 0;
      this.eatenAcc.fill(0);
      this.uptakeAcc.fill(0);
      let any = false;
      for (let i = 0; i < this.N && !any; i++) any = x[i * C + M] > ALIVE;
      this.extinct = this.t > 0 && !any;
      this.history.push({ t: this.t, ...this.stats() });
      if (this.history.length > 2000) this.history.shift();
    }

    // hücreyi boşalt; besini (m + E) aynı hücrede R'ye
    clear(i) {
      const { C, G, x, g } = this;
      this.R[i] += x[i * C + M] + x[i * C + E];
      for (let c = 0; c < C; c++) x[i * C + c] = 0;
      for (let c = 0; c < G; c++) g[i * G + c] = 0;
      this.tag[i] = 0;
    }

    liveOrgs() {
      return [...this.orgs.values()].filter((o) => o.alive && o.area >= this.p.min_org_area);
    }

    stats() {
      const { N, C, x, R } = this;
      let r = 0, nut = 0;
      for (let i = 0; i < N; i++) { r += R[i]; nut += R[i] + x[i * C + M] + x[i * C + E]; }
      return { n_org: this.liveOrgs().length, R_mean: r / N, pred_share: this.lastPred, nutrient: nut };
    }

    // --- kullanıcı araçları ---
    plant(y, xx, genome) {
      const { C, G, W } = this, i = y * W + xx;
      this.x[i * C + M] = 1;
      for (let c = V + 1; c < C; c++) this.x[i * C + c] = 1;
      for (let c = 0; c < G; c++) this.g[i * G + c] = genome ? genome[c] : 0;
      this.tag[i] = this.newTag();
      this.seedTags.push([this.key(i), this.t]);
    }

    // hücreleri sil (besin R'ye), çevresinde yara potansiyeli (wound)
    remove(cells) {
      const { N, C, x, ph } = this;
      const rm = new Float32Array(N), b = new Float32Array(N);
      for (const i of cells) if (x[i * C + M] > 0) { this.clear(i); rm[i] = 1; }
      this.box3(rm, b);
      for (let i = 0; i < N; i++)
        if (b[i] > 0) x[i * C + V] = x[i * C + M] > ALIVE ? Math.min(ph.v_max, x[i * C + V] + (ph.v_gain * b[i]) / 8) : 0;
    }

    disk(y, xx, r) {
      const { H, W } = this, out = [];
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++)
          if (dy * dy + dx * dx <= r * r) out.push((((y + dy) % H + H) % H) * W + (((xx + dx) % W + W) % W));
      return out;
    }

    erase(y, xx, r) {
      this.remove(this.disk(y, xx, r));
    }

    // (y0, x0) - (y1, x1) doğrusu boyunca 1 hücre genişliğinde kesme; koordinatlar sarmalanır
    cut(y0, x0, y1, x1) {
      const n = Math.ceil(Math.hypot(y1 - y0, x1 - x0) * 2) + 1, cells = new Set();
      for (let s = 0; s <= n; s++) {
        const y = Math.round(y0 + ((y1 - y0) * s) / n), xx = Math.round(x0 + ((x1 - x0) * s) / n);
        for (const i of this.disk(y, xx, 1)) cells.add(i);
      }
      this.remove(cells);
    }
  }

  root.LifeWorld = LifeWorld;
  if (typeof module !== "undefined") module.exports = { LifeWorld };
})(typeof window !== "undefined" ? window : globalThis);
