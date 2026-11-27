# Yaşayan Ekosistem 2: Deneyler, Demo, Makale — Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Karar noktası sonrasındaki Y2–Y5 deneylerini bitirmek, yaşayan ekosistemi tarayıcı demosuna taşımak ve IEEE makalesini yazmak.

**Architecture:** Deneyler mevcut `LifeWorld` ve `experiments/y_run.py` üzerine kurulur; yeni fizik yazılmaz.
Demo, `nca/life.py` ile `nca/life_world.py`'nin JS kopyasıdır (`demo/life.js`). Kural ileri geçişi WebGL2'de
(`demo/life_gpu.js`), fizik ve izleme JS'tedir; Python ile parite testi node üzerinden yapılır. Evrimleşmiş
dünya durumları Python'da kaydedilip `<script>` ile yüklenen JS dosyalarına gömülür (demo sunucusuz).

**Tech Stack:** Python 3.13, PyTorch (CUDA, `torch.compile` yok), numpy/pandas/matplotlib, uv; düz JS + WebGL2;
node 24 (parite testleri); LaTeX (IEEEtran).

**Spec:** `docs/superpowers/specs/2026-11-25-yasayan-ekosistem-design.md` (özellikle §7 Deneyler, §8 Demo,
"Ata v2 evrim sonuçları (27 Kasım)" bölümü).

## Global Constraints

- Kod yorumları, belgeler, makale ve arayüz metinleri Türkçe.
- Commit, PR ve dosyalarda Claude/AI atfı yok (Co-Authored-By, "Generated with" vb.).
- Teslim ~22 Aralık 2026: canlı tarayıcı demosu + IEEE makalesi.
- Donanım: RTX 5060 Laptop 8 GB; `torch.compile` yok; paralel süreçler GPU'yu böler.
- Demo sunucusuz çalışır (`file://`): model ve durumlar `<script>` ile yüklenir, `fetch` yok. Kural WebGL2'de.
- Python ile JS arasında adım paritesi test edilir (`tests/test_js_parity.py` deseni, node gerekir).
- 10 dakikadan uzun her koşudan önce `~/.claude/CLAUDE.md` "AI/ML work" listesi: eğitim/kullanım farkı,
  ufuk, kısa sınama koşusu, beklenen sonuç yazılı.
- Dünya ayarları sabit: `configs/life_world.yaml` (ata `runs/life2ft_s0` checkpoint 3000, film_scale 0,5,
  μ 0,05, mu_base 0,03, 192², 12 tohum, 100k adım, kesmeler 50k'da söner). Koşular `torch.manual_seed(seed)` ile tekrarlanabilir.

## Review Focus

1. Demo `file://` ile açılınca model ve durum dosyaları yüklenmeli (fetch yok) — Task 7 duman testi.
2. WebGL2 kayan nokta dokusu yoksa ya da bağlam kaybolursa demo JS yoluna dönmeli — Task 6 `#cpu` testi.
3. Demoda nüfus tükenirse arayüz bozulmamalı; "tohum ekle" dünyayı yeniden canlandırmalı — Task 7.
4. Sarmal kenardan geçen kesme çizgisi ve bedenler doğru işlemeli — Task 5 parite testinde kenardan geçen beden, Task 7 duman testinde kesme.
5. Uzun demo çalışmasında besin kayması (float32 birikimi) — Task 5 korunum testi (2000 adım, göreli kayma < 1e-4).

---

### Task 1: Soy düzeltmeli koşuların analizi (Y2 seçilim/sürüklenme, Y4)

Koşular başladı: `runs/life2l_{mu1,mu0}_s{0..3}` (soy izleme düzeltmesi + `lineage.npy`, 27 Kasım).

**Files:**
- Use: `analysis/price_life.py` (yazıldı, sınama verisinde Price özdeşliği doğrulandı)
- Modify: `analysis/plots_life.py` (`fig_price`), `analysis/summary_life_life2l.md` (üretilir + yorum),
  spec "Ata v2 evrim sonuçları" bölümü

**Interfaces:**
- Produces: `analysis/price_life2l.csv` (seed, t, sel_u, tr_u, sel_norm, tr_norm, lost);
  `analysis/figures/life_price_life2l.{png,pdf}`; düzeltilmiş bölünme/yaş/kuşak tablosu.

- [ ] **Step 1:** Koşular bitince tükenme ve `ebeveynsiz doğum (t > 0) / doğum < %1` denetimi (events.csv).
- [ ] **Step 2:** `uv run python analysis/price_life.py --name life2l --ref life2`. Geçme: Price özdeşliği assert'i
  geçer, ortalama bağlanamayan < 0,01. Yorum kuralı: Σseçilim·u > 0 ve tohumların ≥ 3'ünde Σaktarım·u'dan büyükse
  "birey düzeyi seçilim baskın"; aktarım baskınsa "beden içi (somatik) seçilim baskın".
- [ ] **Step 3:** `plots_life.py`'ye `fig_price(name)`: tohum başına birikimli seçilim·u ve aktarım·u (iki renk
  ailesi, 50k'da kesikli çizgi). `--name life2l` ile çalıştır.
- [ ] **Step 4:** Düzeltilmiş ölçülerle mu1/mu0 tablosu (bölünme/1000 adım, ortalama yaş, kuşak, kapsama, R) ve
  önceki raporla karşılaştırma `summary_life_life2l.md`'ye; spec bölümüne iki cümle.
- [ ] **Step 5:** `gzip -9 runs/life2l_*/orgs.json`; commit (koşular, şekiller, özet, spec).

### Task 2: Ortak bahçe deneyi (Y2 kontrolü + Y5 kalıtsallık)

Soru: evrimleşmiş genom, mutasyonsuz taze dünyada da büyük/yavaş/az bölünen bedeni üretiyor mu (kalıtsal), ve
bu etki aynı büyüklükteki rastgele yönlü genomlarda yok mu (seçilim, sürüklenme değil)?

**Files:**
- Create: `experiments/y5_garden.py`, `tests/test_y5_garden.py`
- Output: `experiments/results/y5_garden.csv`, `analysis/figures/life_garden.{png,pdf}`

**Interfaces:**
- Consumes: `LifeWorld(model, phys, size, n_seeds, params, seed, genomes)`, `load_life`, `World.live_orgs()`,
  `runs/life2l_mu1_s*/history.csv` (son satır g0..g31).
- Produces: `garden_genomes(evolved: np.ndarray (k, G), n_random: int, rng) -> list[tuple[str, int, np.ndarray]]`
  (grup "ata" | "evrim" | "rastgele", kaynak tohum, genom); `measure(world, hist_window) -> dict` (n_org,
  mean_area, speed, pred_share, fission_self / 1000 adım, coverage, R).

- [ ] **Step 1: Test yaz** — `test_garden_genomes_norms`: `garden_genomes(E, 2, rng)` 1 ata (sıfır vektör), k evrim,
  2k rastgele döndürür; her rastgele genomun normu kaynağının normuna eşit (`np.isclose`), kaynağıyla kosinüsü < 0,9.
- [ ] **Step 2:** `uv run pytest tests/test_y5_garden.py -v` → FAIL (modül yok).
- [ ] **Step 3:** Uygula. Dünya: 128², 6 kurucu aynı genomla, `mu=0, mu_base=0, cut_rate=0`, 15000 adım,
  ölçüm son 5000 adımın ortalaması; genom başına 3 tekrar (dünya tohumu 0..2). Gruplar: ata, 4 evrim (life2l
  mu1 tohum başına son ortalama genom), 8 rastgele (evrim genomu başına 2). CLI `--steps --size --reps --jobs`.
- [ ] **Step 4:** Test PASS. Sınama: `--steps 1000 --reps 1` ile ms/adım ölç; toplam süreyi hesapla; 1,5 saati
  aşarsa önce `--size 96` (yazılı not).
- [ ] **Step 5:** Beklenen sonucu spec'e yaz, tam koşuyu arka planda başlat.
- [ ] **Step 6:** Analiz: evrim vs rastgele ve evrim vs ata için alan, hız, bölünme; birim = genom (tekrarlar
  ortalanır), permütasyon testi. Şekil: grup başına noktalar + ortalama. Özete yorum; commit.

### Task 3: Ekolojik dinamik özeti (Y3)

**Files:**
- Modify: `analysis/plots_life.py` (`y3_dynamics(name)`), `analysis/summary_life_life2l.md`

- [ ] **Step 1:** t > 50000 için n_org ve pred_share'in doğrusal eğilimden arındırılmış otokorelasyonu; ilk
  negatif minimumun gecikmesi ve değeri; |ACF| < 2/√n ise "salınım yok". Sonuç tabloya.
- [ ] **Step 2:** Özete 2–3 cümle (salınım varsa periyot, yoksa "kararlı denge"). Commit.

### Task 4: Demo durumları ve dışa aktarım

**Files:**
- Modify: `experiments/y_run.py` (`--save_state`: koşu sonunda `state_<t>.npz` {x, g, R, tag}, float16)
- Create: `export/export_life.py`, `tests/test_export_life.py`
- Output: `demo/life_models.js` (`window.LIFE_MODEL`), `demo/life_states.js` (`window.LIFE_STATES`)

**Interfaces:**
- Produces: `window.LIFE_MODEL = {channels, n_hidden, n_genome, hidden_dim, filters, fire_rate, film_scale,
  w1, b1, w2, out_bias, film_gamma, film_beta, physics, world}` (diziler düz, satır-ana; `world` =
  `configs/life_world.yaml` `world` bölümü); `window.LIFE_STATES = [{name, label, t, H, W, x, g, R, tag}]`
  (diziler base64 float16, HWC).
- `export(run_dir, cfg) -> dict`, `pack_state(npz_path) -> dict`.

- [ ] **Step 1: Test yaz** — `test_export_roundtrip`: geçici bir `LifeCA` kaydı dışa aktarılır; `w1` uzunluğu
  `hidden_dim * n_in * K`, `film_gamma` uzunluğu `n_genome * hidden_dim`; `pack_state` float16 çözüldüğünde
  orijinale `atol=1e-2` yakın.
- [ ] **Step 2:** FAIL. **Step 3:** Uygula (`export/export_evo.py` deseni; FiLM izdüşümlerinin adı ve biçimi
  `nca/life.py` `LifeCA.__init__`'ten okunur). **Step 4:** PASS.
- [ ] **Step 5:** Durum koşuları: `y_run.py --cond mu1 --seed 0 --name life2d --save_state` ve aynısı mu0.
  Tekrarlanabilirlik denetimi: `history.csv` `life2l` ile birebir aynı olmalı; değilse dur ve kök nedeni bul.
  Ata durumu: `--set steps=3000` ile ayrı koşu.
- [ ] **Step 6:** `uv run python export/export_life.py` → iki JS dosyası; toplam < 15 MB. Commit.

### Task 5: JS fizik ve dünya (`demo/life.js`) + parite

**Files:**
- Create: `demo/life.js` (`LifeWorld` sınıfı; `module.exports` ve `window` ikisi de)
- Create: `tests/test_life_js_parity.py`

**Interfaces:**
- Consumes: `window.LIFE_MODEL` biçimi (Task 4).
- Produces: `new LifeWorld(model, H, W, opts)` ile `step()`, `forward(out)` (CPU kural; GPU yokken ve parite için),
  `loadState(s)`, `plant(y, x, genome)`, `cut(y0, x0, y1, x1)`, `erase(y, x, r)`, `track()` → `orgs`
  (Map id → {id, parent, born, area, centroid, genome, speed, eaten, uptake}), `stats()` → {n_org, R_mean,
  pred_share, nutrient}. Alanlar HWC Float32Array: `x` (C), `g` (G), `R`, `tag`.
  `this.forwardHook` atanmışsa kural çıktısı oradan alınır (Task 6).

Python karşılıkları birebir: `life_step`, `transport`, `gather_flow`, `diffuse`, `life_mask`, `foreign_mass`,
`eat`, `update_potential`, `mutate` (base dahil), `LifeWorld.homogenize`, `retag_splits`, `segment`/`label_wrap`,
`clean_small`, `track` (propagül izleme ve etiketle soy dahil). Dünya sarmal (torus).

- [ ] **Step 1: Test yaz** — `test_life_js_matches_python`: `seed_state` ile 48² sarmal dünyada 3 beden (biri
  kenardan geçen), `mu=0`, `fire_rate=1`, 5 adım; JS ve Python `x, R, tag` max fark < 1e-4.
- [ ] **Step 2:** `test_life_js_conserves_nutrient`: aynı dünyada 2000 adım JS; `ΣR + Σm + ΣE` göreli kayması < 1e-4.
- [ ] **Step 3:** `test_life_js_track`: Python `test_bud_keeps_lineage_and_is_not_counted` senaryosu JS'te aynı sonucu verir.
- [ ] **Step 4:** FAIL → uygula → PASS (`uv run pytest tests/test_life_js_parity.py -v`). Commit.

### Task 6: WebGL2 kural (`demo/life_gpu.js`)

**Files:**
- Create: `demo/life_gpu.js` (`LifeGPU.create(world)` → nesne ya da null)

**Interfaces:**
- Consumes: `LifeWorld` alanları ve `forwardHook` (Task 5).
- Produces: `gpu.forward(out)` dünyanın kural çıktısı tamponlarını doldurur (drgb, dh, grow, flow, attack);
  `gpu.renderer` metni. `demo/evo_gpu.js` deseni: algılama (R ve F dahil n_in kanal × K çekirdek, sarmal
  `texelFetch`), FiLM ve çıkış biçimi `LifeCA.forward`'dan birebir.

- [ ] **Step 1:** `life.html#selftest`: bir adımda GPU ve CPU `forward` çıktılarının max farkını konsola yazar;
  geçme < 1e-3. **Step 2:** Uygula. **Step 3:** Tarayıcıda selftest çıktısı ve `#cpu` ile açıldığında demonun JS
  yoluna düştüğü doğrulanır. Commit.

### Task 7: Demo arayüzü (`demo/life.html`, `demo/life_app.js`)

**Files:**
- Create: `demo/life.html`, `demo/life_app.js`
- Modify: `demo/index.html` (bağlantı), `README.md` (demo bölümü)

**Interfaces:**
- Consumes: `LIFE_MODEL`, `LIFE_STATES`, `LifeWorld`, `LifeGPU`.

Arayüz (`evo_app.js` deseninden): başlangıç seçimi (ata / evrimleşmiş mu1 / mutasyonsuz mu0), görünümler
(renk, soy = genom izdüşümü, enerji, besin), araçlar (kes, sil, tohum), hız kaydırıcısı, canlı grafik (birey sayısı,
ortalama besin, avcılık payı), "bir canlıyı izle" paneli (tıklanan bireyin hızı, avcılık payı, yaşı, ebeveyni, alanı).

- [ ] **Step 1:** Duman testi (claude-in-chrome): `file://…/demo/life.html` açılır, üç başlangıç durumu yüklenir,
  konsolda hata yok, 30 sn sonra adım sayacı artmış. Nüfusu sil aracıyla sıfırla → grafik ve panel hata vermez,
  "tohum" ile yeni birey doğar. Kenardan geçen kesme çizgisi iki kenarda da keser.
- [ ] **Step 2:** Uygula. **Step 3:** Duman testini çalıştır, ekran görüntüsü `analysis/figures/demo_life.png`. Commit.

### Task 8: Makale

**Files:**
- Modify: `paper/main.tex`, `paper/refs.bib`, `analysis/plots_life.py` (`fig_snapshots`)

- [ ] **Step 1:** `fig_snapshots`: mu0 ve mu1 (tohum 0) t = 5k, 50k, 100k, 2×3 ızgara, pdf.
- [ ] **Step 2:** Bölümler: Giriş (canlılık hedefi), İlgili çalışmalar (+ Price 1970, Tilman 1982 R*, FiLM
  Perez 2018, paralel evrim; Flow-Lenia mevcut), Yöntem (LifeCA, kütle korunumlu taşınım ve kimlik bariyeri,
  kapalı besin döngüsü, yeme, yara potansiyeli ve mutasyon, ata ön eğitimi v2: havuz yaşlandırma ve kıtlık),
  Deneyler (Y1–Y5), Sonuçlar (zaman serisi, anlık görüntüler, paralel evrim, Price, ortak bahçe, Y3), Tartışma
  (K-stratejisi, durağan mozaik, sınırlar: somatik mutasyon, F girdisi ön eğitimde sıfır), planarya yaklaşımı
  "ilk yaklaşım ve neden terk edildi" olarak bir paragraf, Sonuç.
- [ ] **Step 3:** `cd paper && latexmk -pdf main.tex` hatasız; şekiller bulunur; sayfa sınırı denetimi. Commit.

### Task 9: Dalı kapatma

- [ ] **Step 1:** Tüm dalın gözden geçirilmesi (en yetenekli modelle tek gözden geçirici); bulgular düzeltilir.
- [ ] **Step 2:** Kullanıcı onayıyla deneme koşuları (`runs/_*`) silinir, `.measure/` `.gitignore`'a.
- [ ] **Step 3:** `superpowers:finishing-a-development-branch`.

## Takvim

| Bitiş | İş |
|---|---|
| 29 Kasım | Task 1–3 (deneyler ve analiz) |
| 2 Aralık | Task 4 (durumlar, dışa aktarım) |
| 10 Aralık | Task 5–7 (demo) |
| 18 Aralık | Task 8 (makale) |
| 21 Aralık | Task 9 |
