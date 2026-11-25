# Yaşayan Ekosistem — Plan 1: Fizik, Ön Eğitim, İlk Evrim (karar noktasına kadar)

> Adımlar `- [ ]` onay kutularıyla izlenir.

**Amaç:** Hareket, yeme ve bölünmeyi mümkün kılan kütle korunumlu fizik, şekilsiz "en basit
canlı" ön eğitimi, FiLM ile kalıtsal kural ve kesmeleri kademeli sönen ilk evrim koşuları; 9 Aralık
karar noktasında "atada olmayan davranış çıktı mı?" sorusunu cevaplayacak veriyi üretmek.

**Mimari:** `nca/life.py` türevlenebilir tek-canlı fiziğini (kural, taşınım, metabolizma, ölüm,
potansiyel) içerir. Hem ön eğitim hem dünya bunu kullanır. `nca/life_world.py` mevcut `World`'ü
miras alır ve çok-canlı kısmını ekler: kimlikler, yeme, bölünme nedeni, sönen kesmeler,
davranış betimleyicileri. Durum düzeni mevcut kodla uyumludur: kanal 3 kütle (eski alfa, canlılık
eşiği 0,1 aynı), kanal 4 V. Bu sayede `alive_cells`, `wound`, `mutate`, `make_seed`,
`apply_damage`, `World.cut/move/segment/track` değişmeden çalışır.

**Teknoloji:** Python 3.13, PyTorch (CUDA), NumPy, SciPy, pytest. Yeni bağımlılık yok.

**Tasarım:** `docs/yasayan-ekosistem/tasarim.md`

**Kapsam dışı (Plan 2, karar noktasından sonra):** Y2–Y5 tam deneyleri ve şekilleri, demo
(`demo/evo.*`, WebGL2, JS paritesi), makale.

## Genel Kısıtlar

- Kod yorumları ve belgeler Türkçe yazılır.
- Commit mesajları mevcut biçimdedir (`feat(life): ...`, `test: ...`, `docs: ...`).
- Testler: `uv run pytest -q tests`. Tüm eski testler geçmeye devam eder.
- GPU: RTX 5060 8 GB. `torch.compile` kullanılmaz. Eğitimler tek süreçte çalışır.
- Elle yazılmış davranış yasaktır: fizik hareket/saldırı/bölünmeyi yalnızca *mümkün* kılar.
  Kayıplarda şekil, renk veya davranış hedefi bulunmaz.
- Besin kapalı döngüdür: `ΣR + Σm + ΣE` korunur (kesme ve ezilme hariç, bkz. Görev 7).
- Ön eğitimde `g = 0`. FiLM izdüşümleri (`P_s`, `P_b`) asla eğitilmez.
- Çalışma dalı: `yasayan-ekosistem`.

## Odak Noktaları (incelemede)

1. **Izgara kenarındaki canlı**: kenara akan kütle dışarı çıkmaz, hücrede kalır (Task 2 testi).
2. **Sıfır kütleli bölgede bölme**: yoğun nicelik taşınımı `0/0` ile NaN üretmemeli (Task 2 testi).
3. **Karşılıklı saldırı**: birbirine saldıran iki komşu canlıda kütle negatif olmamalı ve besin
   korunmalı (Task 4 testi).
4. **Tükenme**: tüm canlılar ölünce `track/record` çökmemeli, koşu tükenmeyi kaydedip erken bitmeli
   (Task 7 testi).
5. **Kaynak tükenmesi**: kalabalıkta `R` hiçbir zaman negatif olmamalı (Task 3 testi).

---

### Task 1: `LifeCA` modeli (FiLM'li kural)

**Dosyalar:**
- Oluştur: `nca/life.py`
- Test: `tests/test_life.py`

**Arayüzler:**
- Üretir:
  - Durum kanal sabitleri: `RGB = slice(0, 3)`, `M_CH = 3`, `V_CH = 4` (`nca.evo`'dan),
    `E_CH = 5`, gizli kanallar `6:`.
  - `class LifeCA(nn.Module)` — `__init__(n_hidden=12, n_genome=32, hidden_dim=128,
    perception="sobel_lap", fire_rate=0.5, film_scale=1.0, film_seed=0)`, `channels = 6 + n_hidden`.
    Algı girdisi `cat([x, R, F], 1)` (`channels + 2` kanal) üzerinde `KERNELS/PERCEPTIONS`
    filtreleridir (`nca/model.py`).
  - `LifeCA.forward(x, R, F, g) -> dict` anahtarlar: `drgb (B,3)`, `dh (B,n_hidden)`,
    `grow (B,1) ∈ (0,1)`, `flow (B,9) softmax` (indeks 0 = kal, 1..8 = `DIRS` sırası),
    `attack (B,1) ∈ (0,1)`. Tümü `(B, k, H, W)`.
  - `DIRS: list[tuple[int,int]]` = `[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]`.
  - `LifeCA.config() -> dict` (yükleme için; `film_scale` ve `film_seed` dahil).
- FiLM: `z' = z * (1 + film_scale * (P_s g)) + film_scale * (P_b g)`. `P_s`, `P_b` kayıtlı
  tamponlardır: `torch.randn(hidden_dim, n_genome, generator=film_seed) / sqrt(n_genome)`. 1x1
  konvolüsyon olarak uygulanır. `film_scale` yüklemeden sonra değiştirilebilir bir özniteliktir
  (kalibrasyon için).
- Başlatma: son katman sıfırdır. `flow` logitlerinde "kal" sapması +5, `attack = sigmoid(y − 4)`.

- [ ] **Adım 1: Başarısız testleri yaz** (`tests/test_life.py`)

```python
def test_film_zero_genome_matches_plain():
    m = LifeCA(); torch.nn.init.normal_(m.w2.weight, std=0.1)
    x, R, F = rand_state(m), torch.rand(1,1,16,16), torch.zeros(1,1,16,16)
    a = m(x, R, F, torch.zeros(1, 32, 16, 16)); m.film_scale = 5.0
    b = m(x, R, F, torch.zeros(1, 32, 16, 16))
    assert all(torch.allclose(a[k], b[k]) for k in a)

def test_film_genome_changes_output():
    # aynı m, x; g = randn -> en az bir çıktı anahtarında max fark > 1e-3

def test_init_is_sedentary_and_peaceful():
    out = LifeCA()(x, R, F, g0)
    assert (out["flow"][:, 0] > 0.95).all() and (out["attack"] < 0.02).all()
    assert torch.allclose(out["flow"].sum(1), torch.ones_like(out["flow"][:, 0]))
```

`rand_state(m)` test dosyasında küçük bir yardımcıdır: `(1, m.channels, 16, 16)` rastgele
durum, kanal 3 ∈ [0, 1].

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör** — `uv run pytest -q tests/test_life.py` → ImportError
- [ ] **Adım 3: `LifeCA`'yı `nca/life.py` içinde uygula** (yukarıdaki arayüz)
- [ ] **Adım 4: Çalıştır, geçtiğini gör** — `uv run pytest -q tests/test_life.py` → 3 passed
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): FiLM genomlu LifeCA kuralı"`

---

### Task 2: Kütle korunumlu taşınım

**Dosyalar:**
- Değiştir: `nca/life.py`
- Test: `tests/test_life.py`

**Arayüzler:**
- Tüketir: `DIRS` (Task 1).
- Üretir: `transport(m, flow, ext, inten, tag=None, self_tol=0.05) -> (m2, ext2, inten2, tag2)`
  - `m (B,1,H,W)`, `flow (B,9,H,W)`, `ext (B,k,H,W)` yaygın nicelikler (kütle gibi
    taşınır; burada `E`), `inten (B,j,H,W)` yoğun nicelikler (kütle ağırlıklı ortalanır: RGB,
    gizli, `g`), `tag (B,1,H,W)` veya `None`.
  - Kenar: ızgara dışına giden pay "kal"a eklenir.
  - Kimlik engeli (`tag` verilince): hedef hücre canlıysa ve etiketi kaynaktan farklıysa pay
    geri döner (kal'a eklenir). Hedef boşsa ama hedefin canlı komşularının etiket varyansı
    `self_tol²`'yi aşıyorsa (iki bireyin arası, mevcut `World.step` hilesi) pay geri döner. Böylece
    bir hücre hiçbir zaman iki farklı etiketten kütle almaz.
  - Yeni hücrenin etiketi, katkı veren kaynağın etiketidir.
  - Yoğun nicelikler `Σ(m_k q_k) / max(m2, 1e-8)` ile ortalanır. `m2 == 0` olan hücrede 0'dır.
  - Türevlenebilir (`m`, `flow`, `ext`, `inten` üzerinden). `tag` gradyansızdır.
  - Gerçekleme: 9 kaydırılmış toplama (gather), `F.pad` + dilimleme. `nca/world.py`'deki
    `F.unfold` deseni de kullanılabilir.

- [ ] **Adım 1: Başarısız testleri yaz**

```python
def test_transport_conserves_mass_and_energy():   # rastgele m, flow, ext; tag=None
    assert torch.allclose(m2.sum(), m.sum(), rtol=1e-5) and torch.allclose(ext2.sum(), ext.sum(), rtol=1e-5)

def test_transport_edge_mass_stays():               # tek hücre (0,0), tüm akış (-1,-1) yönüne
    assert m2[0,0,0,0] == m[0,0,0,0]

def test_transport_intensive_is_mass_weighted():    # iki kaynak (m=1, q=0) ve (m=3, q=4) aynı boş hücreye tam akar
    assert torch.isclose(inten2[0,0,y,x], torch.tensor(3.0))

def test_transport_no_nan_on_empty_grid():          # m = 0 her yerde
    assert torch.isfinite(inten2).all() and (inten2 == 0).all()

def test_transport_tags_never_mix():                # iki farklı etiketli blok, tüm akış birbirine doğru
    # her hücre için: m2 > 0 ise etiketi tek bir kaynaktandır; diğer bloğun hücrelerine giren kütle 0
    assert torch.allclose(m2.sum(), m.sum())
```

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör**
- [ ] **Adım 3: `transport`'u uygula**
- [ ] **Adım 4: Çalıştır, geçtiğini gör** — `uv run pytest -q tests/test_life.py` → 8 passed
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): kütle korunumlu, kimlik engelli taşınım"`

---

### Task 3: Adım fiziği `life_step` (metabolizma, ölüm, potansiyel)

**Dosyalar:**
- Değiştir: `nca/life.py`, `nca/evo.py` (potansiyel güncellemesini yardımcıya çıkar)
- Test: `tests/test_life.py`, `tests/test_evo.py` (değişmeden geçmeli)

**Arayüzler:**
- Tüketir: `LifeCA`, `transport`.
- Üretir:
  - `nca/evo.py`: `update_potential(x, was_alive, alive, phys) -> Tensor (B,1,H,W)`. Bu,
    `evo.step` içindeki V bloğunun aynısıdır ve `evo.step` artık bunu çağırır.
  - `@dataclass class LifePhysics`: mevcut `Physics` alanları (`v_decay=0.97, v_gain=1.0,
    v_diffuse=0.2, v_max=2.0, kappa=0.02`) + `k_up=0.1, c_m=0.002, c_move=0.01, r_g=0.05,
    r_diffuse=0.25, self_tol=0.05`. (Sayılar başlangıç değeridir, Task 6'da ayarlanabilir.)
    `World.__init__` `type(phys)(**vars(phys))` ile kopyaladığı için dataclass uygundur.
  - `life_step(model, x, g, R, phys, tag=None, F=None, fire_rate=None) -> (x, g, R, tag, out)`.
    `out` model çıktıları dict'idir (`attack` Task 4'te kullanılır) ve `moved (B,1,H,W)`
    içerir. `F=None` ise yabancı kütle sıfırdır (tek canlı).
- Sıra (tasarım 4.1, 4.2, 4.4, 4.5):
  1. Kural. Ateşlemeyen hücrede `flow = one_hot(kal)`, `grow = 0`, `attack = 0`, Δ = 0.
  2. `RGB`, `h` += Δ. Ardından `transport(m, flow, ext=E, inten=[RGB, h, g], tag)`.
  3. Alım: `u = min(R, k_up · R · clamp(m, 0, 1))`, `R -= u`, `E += u`.
  4. Maliyet: `c = c_m · m + c_move · moved` (`moved = m · (1 − flow_kal)`), `E -= c`, `R += c`.
  5. Büyüme: `dm = r_g · grow · clamp(E, min=0)`, `m += dm`, `E -= dm`.
  6. Ölüm: `E ≤ 0` veya `m ≤ 0,1` olan hücrenin `m + E` değeri `R`'ye eklenir, tüm kanalları
     ve `g`'si sıfırlanır. Bu hücreler `died` maskesini oluşturur. `E` negatifse bu, 4. adımda
     `R`'ye tamamen eklenen maliyetin karşılanmayan kısmını geri alır: korunum tam kalır ve `R`
     negatif olmaz, çünkü açık o hücrenin `R`'sine aynı adımda eklenen maliyetten büyük olamaz.
  7. `V = update_potential(...)`, `R` difüzyonu (replicate dolgulu Laplace; korunumlu).
- Türevlenebilir: `x`'in `RGB, m, E, h` kanalları ve `R`. `V`, `g`, `tag` gradyansızdır
  (`evo.step`'teki gibi `no_grad` blokları).

- [ ] **Adım 1: Başarısız testleri yaz**

```python
def test_life_step_conserves_nutrient():          # rastgele ağırlıklı LifeCA, 3 tohum, 50 adım
    tot = lambda x, R: R.sum() + x[:, M_CH].sum() + x[:, E_CH].sum()
    assert torch.isclose(tot(x, R), t0, rtol=1e-4)

def test_starving_cell_returns_mass_to_R():       # E = 0'lık tek canlı blok, büyüme kapalı
    assert alive_cells(x).sum() == 0 and torch.isclose(tot(x, R), t0, rtol=1e-4)

def test_resource_never_negative():               # kaynak az (R = 0.01), çok canlı, 100 adım
    assert R.min() >= 0

def test_life_step_backprop():                    # loss = x[:, M_CH].sum(); loss.backward()
    assert model.w1.weight.grad is not None and torch.isfinite(model.w1.weight.grad).all()
```

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör**
- [ ] **Adım 3: `update_potential`'ı çıkar, `LifePhysics` ve `life_step`'i uygula**
- [ ] **Adım 4: Çalıştır** — `uv run pytest -q tests` → hepsi geçer (eski `test_evo.py` dahil)
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): kapalı besin döngülü adım fiziği"`

---

### Task 4: Yeme

**Dosyalar:**
- Değiştir: `nca/life.py`
- Test: `tests/test_life.py`

**Arayüzler:**
- Üretir:
  - `LifePhysics`'e alanlar: `k_eat=0.2, eta=0.6, c_attack=0.005`.
  - `foreign_mass(x, tag) -> Tensor (B,1,H,W)`: 3x3 komşulukta, farklı etiketli canlı
    hücrelerin kütle toplamı (modelin `F` girdisi).
  - `eat(x, R, tag, attack, phys) -> (x, R, lost)`. `lost (B,1,H,W)` her hücrenin yenilen
    kütlesidir.
    - Kurban `v`: saldırganlar = farklı etiketli canlı komşular. `A = Σ a_j`.
      `T = min(m_v, k_eat · A · m_v)`.
    - Kurban `T` kütle ve `E_v · T / m_v` enerji kaybeder.
    - Saldırgan `j` alır: `pay_j = (T + E payı) · a_j / A`. `η · pay_j` saldırganın `E`'sine,
      `(1 − η) · pay_j` `R`'ye gider.
    - Saldırı maliyeti: `min(c_attack · a, max(E, 0))` saldırgan `E`'sinden `R`'ye.
    - Kurbanın `V`'si `v_gain · T / m_v` kadar artar (`v_max` ile kırpılır).
    - Gerçekleme: `F.unfold` 3x3, iki geçiş (kurban toplamları, saldırgan payları).
- `eat` yalnızca dünyada çağrılır, ön eğitimde çağrılmaz.

- [ ] **Adım 1: Başarısız testleri yaz**

```python
def test_eat_transfers_to_attacker():             # A (a=1) ve B (a=0) bitişik bloklar
    assert x2[0, M_CH][B].sum() < x[0, M_CH][B].sum() and x2[0, E_CH][A].sum() > x[0, E_CH][A].sum()

def test_same_tag_cannot_eat():                   # tek etiket, a = 1 her yerde
    assert torch.allclose(x2, x) and torch.allclose(R2, R)

def test_mutual_attack_conserves_and_nonnegative():   # iki blok, ikisi de a = 1
    assert x2[:, M_CH].min() >= 0 and torch.isclose(tot(x2, R2), tot(x, R), rtol=1e-5)

def test_victim_potential_rises():                # B kenarında V > 0, A'da V == 0
```

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör**
- [ ] **Adım 3: `foreign_mass` ve `eat`'i uygula**
- [ ] **Adım 4: Çalıştır, geçtiğini gör**
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): komşu bireyi yeme fiziği"`

---

### Task 5: Ön eğitim "en basit canlı"

**Dosyalar:**
- Oluştur: `nca/train_life.py`, `configs/life.yaml`
- Test: `tests/test_life.py`

**Arayüzler:**
- Tüketir: `LifeCA`, `LifePhysics`, `life_step`. Yeniden kullanılanlar: `nca.train.normalize_grads`,
  `nca.train_evo.apply_damage` (E kanal 5'te olduğu için hasar onu da siler),
  `nca.utils` (`load_config`, `set_seed`, `device`, `save_grid`, `to_uint8`). Gradyan
  checkpoint'i için `nca.train_evo.run` deseni, `life_step` ile.
- Üretir:
  - `debris_loss(x, min_viable) -> Tensor (B,)`: `min_viable`'dan küçük bağlı bileşenlerdeki
    kütle / toplam kütle. Bileşenler, `detach` edilmiş canlılık maskesinde `scipy.ndimage.label`
    (8-bağlantı) ile bulunur. **"Tek beden" terimi kullanılmaz.**
  - `body_descriptors(x) -> Tensor (B, 6)`: toplam kütle, kütle ağırlıklı ortalama RGB (3),
    uzama (ikinci moment özdeğer oranı), ortalama E. Ağırlık merkezine göredir, yani öteleme
    altında değişmez.
  - `life_losses(desc_prev, x, cfg) -> dict`: `mass` (`[A_min, A_max]` dışı kare),
    `debris`, `energy` (`relu(e_min − ΣE/Σm)²`), `homeo` (`|desc(x) − desc_prev|²`,
    `desc_prev` son `homeo_k` adım önceki betimleyicidir), `bounds` (`train_ancestor.bounds`
    biçimi: RGB [0,1], gizli |·| ≤ 3).
  - `train(cfg) -> Path` (`train_ancestor.train` iskeleti: pool, hasar, BPTT + checkpoint,
    log.csv, snapshots, model.pt). Pool'da `x, g(=0), R` birlikte tutulur. `R` başlangıçta 1'dir.
    Kayıt: `{"state_dict", "model_config", "physics", "config"}`.
  - CLI: `uv run python -m nca.train_life configs/life.yaml --seed 0 [--set k=v ...]`.
- `configs/life.yaml`: `grid: 64`, `batch_size: 8`, `pool_size: 1024`, `steps: [64, 96]`,
  `homeo_k: 16`, `lr: 2e-3`, `iterations: 10000`, `A_min: 40`, `A_max: 400`, `e_min: 0.2`,
  `min_viable: 10`, `damage` (`ancestor.yaml` ile aynı), model `n_genome: 32`. Kayıp ağırlıkları
  1'dir (Task 6'da ayarlanır).

- [ ] **Adım 1: Başarısız testleri yaz**

```python
def test_debris_loss_allows_two_viable_bodies():   # iki ayrı 5x5 blok, min_viable=10
    assert debris_loss(x, 10).item() == 0

def test_debris_loss_penalizes_small_fragment():   # bir 5x5 + bir 2x2 blok
    assert debris_loss(x, 10).item() > 0

def test_descriptors_translation_invariant():      # aynı beden 7 piksel kaydırılmış
    assert torch.allclose(body_descriptors(x), body_descriptors(x_shift), atol=1e-5)

def test_train_life_smoke(tmp_path):               # --set iterations=3 grid=32 batch_size=2 pool_size=8 steps=[8,8] out_dir=<tmp>
    assert (out / "model.pt").exists()
```

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör**
- [ ] **Adım 3: Kayıpları ve `train`'i uygula**, `configs/life.yaml`'ı yaz
- [ ] **Adım 4: Çalıştır, geçtiğini gör** — `uv run pytest -q tests`
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): şekilsiz en basit canlı ön eğitimi"`

---

### Task 6: Ön eğitimi koş ve Y1 (deney görevi)

**Dosyalar:**
- Oluştur: `experiments/y1_viability.py`
- Çıktı: `runs/life_s0/`, `experiments/results/y1.csv`

**Arayüzler:**
- Tüketir: `train_life`, `life_step`, `LifeCA(**model_config)` ile yükleme.
- Üretir: `experiments/y1_viability.py --run life_s0 --n 64 --steps 2000` →
  `results/y1.csv` sütunları: `sample, alive_2000, n_bodies, mass, regrew_after_cut`.
  Tek tohumdan büyütür, 1000. adımda yarım düzlem kesmesi (`nca.damage.halfplane_cut`) yapar,
  2000. adımda ölçer.

- [ ] **Adım 1:** `uv run python -m nca.train_life configs/life.yaml --seed 0 --set iterations=500`
  ile kısa koşu. Süreyi ölç ve `log.csv`'de kaybın düştüğünü gör.
- [ ] **Adım 2:** Tam koşu (`iterations: 10000`). `snapshots/` galerisine bak: canlı tek parça
  mı, kalıcı mı?
- [ ] **Adım 3:** `y1_viability.py`'yi yaz ve çalıştır.
  **Geçiş ölçütü:** `alive_2000 ≥ 0,9` ve `regrew_after_cut ≥ 0,8`.
- [ ] **Adım 4:** Ölçüt tutmazsa yalnızca kayıp ağırlıklarını ve `LifePhysics` sayılarını
  ayarla. Hedef terim eklemek yasaktır. Her denemeyi `runs/_life_v*` olarak sakla ve tasarım
  belgesinin sonuna kısa bir not düş.
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): Y1 canlılık deneyi ve ön eğitim sonuçları"`
  (`runs/life_s0/model.pt`, `config.yaml`, `log.csv`, `summary.yaml` ve
  `experiments/results/y1.csv`; mevcut `.gitignore` kurallarına uy)

---

### Task 7: `LifeWorld`

**Dosyalar:**
- Oluştur: `nca/life_world.py`, `configs/life_world.yaml`
- Test: `tests/test_life_world.py`

**Arayüzler:**
- Tüketir: `World`, `WorldParams`, `Organism` (`nca/world.py`), `life_step`, `eat`,
  `foreign_mass`, `mutate`.
- Üretir:
  - `@dataclass LifeParams(WorldParams)`: ek alanlar `anneal_steps: int = 50000`,
    `cause_window: int = 50`. `cut_rate` (miras, 0,002) başlangıç oranıdır.
  - `class LifeWorld(World)`:
    - `step()`: `life_step(..., tag=self.tag, F=foreign_mass(x, tag))` → `eat` → aynı etiket
      içinde `homogenize` → `mutate` + `g_clip` → `t += 1`. `track_every`'de sırasıyla
      `retag_splits()`, `clean_debris()`, `track()`, `predation()` çağrılır.
      `self.eaten_acc`, `self.uptake_acc` (H,W) ve etiket başına son saldırı kaybı adımı
      `self.last_attacked: dict[int, int]` (anahtar `round(tag·1000)`) biriktirilir.
    - `retag_splits()`: aynı etiketin ≥ 2 bileşeni olup her biri `min_viable` boyutundaysa,
      en büyüğü dışındakilere `new_tag(like=...)` verilir.
    - `clean_debris()`: `min_viable` altı bileşenlerin `m + E`'si `R`'ye eklenip silinir (korunum).
      Koruma süresindeki tohumlar hariçtir (`self.planted`). Böylece `World.track`'in kendi
      temizliği bir şey bulmaz.
    - `current_cut_rate() -> float = cut_rate · max(0, 1 − t / anneal_steps)`.
      `predation()` bu oranı kullanır.
    - `cut(...)`: kesilen bandın `m + E`'si önce `R`'ye eklenir, sonra `super().cut`.
      `shortcut:` `World.move` ezilen hücrelerin besinini kaybeder. Bu, kesmeler sönünce biter.
      Korunum sapması > %0,1 ölçülürse `move` da düzeltilir.
    - `track()`: `super().track()` çağrılır. Bu turda `"dogum"` olaylarına neden eklenir:
      `"disturbance"` (son `cause_window` adımda ebeveyn kesildi), `"attack"` (ebeveynin etiketi
      bu pencerede yenildi) ya da `"self"`. `self.events`'e `(t, "bolunme", oid, parent, neden)`
      yazılır.
    - `record()`: `super().record()` satırına `speed` (canlı bireylerin iki track arası ağırlık
      merkezi yer değiştirmesi / `track_every`, ortalama), `pred_share` (`eaten / (eaten +
      uptake)`), `fission_self`, `fission_attack`, `fission_disturbance` (bu penceredeki sayılar),
      `cut_rate` ve `nutrient_total` eklenir.
    - Tükenme: canlı birey kalmadığında `self.extinct = True` olur. `record` boş listeyle çökmez.
- `configs/life_world.yaml`: `size: 192`, `n_seeds: 12`, `mu: 1.0`, `anneal_steps: 50000`,
  `steps: 100000`, `run: life_s0`.

- [ ] **Adım 1: Başarısız testleri yaz** (sabit, elle kurulmuş durumlar; eğitilmiş model gerekmez)

```python
def test_self_split_gets_new_tag_and_cause_self():  # tek etiketli iki ayrı 6x6 blok, kesme/saldırı yok
    w.retag_splits(); w.track()
    assert len(set(tags_of_bodies)) == 2 and any(e[1] == "bolunme" and e[4] == "self" for e in w.events)

def test_cut_rate_anneals_to_zero():
    w.t = 0; assert w.current_cut_rate() == p.cut_rate
    w.t = p.anneal_steps; assert w.current_cut_rate() == 0

def test_world_step_conserves_without_cuts():       # cut_rate = 0, rastgele ağırlıklı model, 100 adım
    assert abs(nutrient(w) - n0) / n0 < 1e-4

def test_extinction_does_not_crash():               # tüm hücreleri sil, 20 adım
    assert w.extinct and w.history[-1]["n_org"] == 0
```

- [ ] **Adım 2: Çalıştır, başarısız olduğunu gör**
- [ ] **Adım 3: `LifeWorld`'ü uygula**
- [ ] **Adım 4: Çalıştır** — `uv run pytest -q tests` → hepsi geçer
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): çok canlılı dünya, sönen kesmeler, bölünme nedenleri"`

---

### Task 8: `σ_P` (film_scale) kalibrasyonu ve süre ölçümü (deney görevi)

**Dosyalar:**
- Oluştur: `experiments/calibrate_film.py`
- Çıktı: `experiments/results/film_scale.csv`

**Arayüzler:**
- Üretir: `--run life_s0 --scales 0.25 0.5 1 2 4 --n 32 --steps 2000` → sütunlar
  `scale, survive_frac, mean_mass, desc_dist`. `desc_dist`, `g = 0` atasına göre ortalama
  `body_descriptors` mesafesidir. Ayrıca `LifeWorld` (192², 12 tohum) için 500 adımın
  saniye/adım değeri yazdırılır.

- [ ] **Adım 1:** Betiği yaz ve çalıştır.
- [ ] **Adım 2:** `survive_frac`'ı 0,5'e en yakın olan ölçeği seç ve `configs/life_world.yaml`'a
  `film_scale` olarak yaz. `desc_dist` bu ölçekte sıfıra yakınsa (genom davranışı
  değiştirmiyorsa) bir sonraki büyük ölçeği seç.
- [ ] **Adım 3:** Saniye/adım ölçümüyle 100k adımlık koşu süresini hesapla. 8 saati aşıyorsa
  `steps`/`anneal_steps`'i orantılı küçült ve tasarım belgesine not düş.
- [ ] **Adım 4: Commit** — `git commit -m "feat(life): FiLM ölçeği kalibrasyonu"`

---

### Task 9: İlk evrim koşuları ve karar noktası raporu (deney görevi)

**Dosyalar:**
- Oluştur: `experiments/y_run.py`, `analysis/plots_life.py`
- Çıktı: `runs/lifeworld_<koşul>_s<tohum>/` (`history.csv`, `events.csv`, `orgs.json`,
  her 5000 adımda `snap_<t>.png`), `analysis/figures/life_*.png`, `analysis/summary_life.md`

**Arayüzler:**
- Üretir:
  - `experiments/y_run.py --config configs/life_world.yaml --cond {mu1,mu0} --seed N`.
    `mu0` için `mu = 0`. Tükenmede erken biter ve `history.csv`'ye `extinct` bayrağı yazar.
  - `analysis/plots_life.py`: koşul başına zaman serileri (`n_org`, `speed`, `pred_share`,
    `fission_self` oranı, `cut_rate`). Gölge: tohumlar arası standart hata.
    `summary_life.md`'de kesmeler söndükten sonraki son %20'lik dilimin ortalamaları
    karşılaştırılır.

- [ ] **Adım 1:** `y_run.py`'yi yaz. Kısa koşuyla (`--set steps=2000 anneal_steps=1000`) dosyaları
  doğrula.
- [ ] **Adım 2:** Dört koşu: `mu1` ve `mu0`, tohum 0 ve 1. Arka planda sırayla çalıştır
  (`run_in_background`), bitişi bildirimle bekle.
- [ ] **Adım 3:** `plots_life.py`'yi çalıştır. `summary_life.md`'ye şunları yaz:
  - Kesmeler söndükten sonra popülasyon sürüyor mu?
  - `mu1`'de `speed`, `pred_share` ve `fission_self` atadan ve `mu0`'dan farklı mı?
  - Demoda gösterilebilecek bir an var mı (snapshot)?
- [ ] **Adım 4: Karar raporu:** tasarım belgesine "Karar noktası (9 Aralık)" bölümünü ekle:
  devam / olumsuz sonuç olarak yaz. Kullanıcıya sun. Plan 2 bu rapora göre yazılır.
- [ ] **Adım 5: Commit** — `git commit -m "feat(life): ilk evrim koşuları ve karar noktası raporu"`
