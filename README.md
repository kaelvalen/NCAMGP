# NCA Morfogenez

Tek bir öğrenilmiş yerel kural (Neural Cellular Automata, Mordvintsev vd. 2020) ile tek hücreden
planarya benzeri bir organizma büyütme, hasar onarımı ve kesilen parçalardan tam organizma
yenilenmesi. Özgün kısım: parça boyutu–yenilenme eğrisi, kesme eğitimi olmadan bölünme ve
gizli kanallardaki baş–kuyruk bilgisine müdahaleyle iki başlı organizma.

## Kurulum

```bash
uv sync                       # Python 3.13 + CUDA 12.8 PyTorch
uv run pytest -q tests        # çekirdek testler + JS/Python parite testi (node gerekir)
```

NixOS notu: `nca/__init__.py`, `libcuda`'yı `/run/opengl-driver/lib`'den önceden yükler; başka
bir dağıtımda bu adım sessizce atlanır.

## Akış

```bash
uv run python targets/make_target.py            # hedef organizma (40x40 RGBA + bölge/eksen meta)
bash experiments/train_all.sh core              # 4 rejim x 3 tohum (model başına ~15 dk, RTX 5060)
bash experiments/train_all.sh ablations         # ablasyonlar (ABL_SEEDS="0 1 2" ile 3 tohum)
uv run python -m nca.train_noise configs/noise.yaml   # Genişletme 2 (esnek)
bash experiments/run_experiments.sh             # D1-D6, şekiller, demo dışa aktarımı
```

Tek bir koşu: `uv run python -m nca.train configs/r3.yaml --seed 1 --set iterations=4000`.
Her koşu `runs/<ad>_s<tohum>/` altına `config.yaml`, `log.csv`, `model.pt`, `growth.gif`,
`summary.yaml` ve anlık görüntüler yazar; config dosyasıyla birebir tekrarlanabilir.

## Demo

```bash
uv run python export/export_weights.py   # demo/models.js üretir (R0-R3, tohum 0)
xdg-open demo/index.html                 # sunucu gerekmez
```

Araçlar: **Sil** (fırça), **Kes** (çizgi çiz; parçalar otomatik ayrılır), **Müdahale** (gizli kanal
boyama; "Baş sinyali" prob kanallarını baş değerlerine ayarlar), **Tohum**. Model seçimi durumu
korur; **Kaydet / Geri yükle** ile aynı kesme farklı modellerde denenir.

## Yapı

```
configs/       base.yaml, r0-r3.yaml, ablations/, noise.yaml, smoke.yaml
targets/       make_target.py, planaria.png/.npz
nca/           model.py, train.py, damage.py, metrics.py, evaluate.py, train_noise.py, utils.py
experiments/   d1-d6 betikleri, train_all.sh, run_experiments.sh, results/
analysis/      plots.py -> figures/, summary.md
export/        export_weights.py
demo/          index.html, nca.js (çalışma zamanı), app.js (arayüz), models.js
paper/         main.tex (IEEEtran), refs.bib
tests/         test_core.py, test_js_parity.py
```

## Deneyler

| Betik | Soru | Çıktı |
|---|---|---|
| `d1_reproduce.py` | Büyüme / dairesel onarım / 1000 adım kararlılığı | `results/d1.csv`, büyüme dizileri |
| `d2_fragment_curve.py --mode inplace` | Parça boyutu yenilenmeyi nasıl etkiler? | `results/d2_inplace.csv` (ana şekil) |
| `d2_fragment_curve.py --mode split` | Ayrılan iki parçanın ikisi de birey olur mu? | `results/d2_split.csv` |
| `d3_cut_direction.py` | Baş/kuyruk/orta parça farkı | `results/d3.csv` |
| `d4_probe.py` | Gizli kanallar eksen konumunu kodluyor mu? | `runs/*/probe.json`, `results/d4.csv` |
| `d5_intervention.py` | Müdahaleyle iki baş; yeniden kesmede kalıcılık | `results/d5.csv` |
| `d6_noise.py` | Gürültüden kaç birey çıkar? | `results/d6.csv` |

Metrik tanımları `nca/metrics.py` içindedir. Onarım başarısı: hasardan 200 adım sonra IoU ≥ 0,9
olan örneklerin oranı; onarım süresi IoU'nun 0,9'un üstünde kalıcı olarak kalmaya başladığı adım
(planın "ilk kez geçtiği adım" tanımı geçici dokunuşları da başarı saydığı için sıkılaştırıldı).
Değerlendirmede dairesel hasarın merkezi rastgele bir canlı hücredir, yarıçap 0,1–0,25
(normalize; eğitimde Mordvintsev'deki gibi 0,1–0,4 ve düzgün dağılımlı merkez). Bölünme modunda parçalar başka konuma taşındığı için
ağırlık merkezine hizalanmış IoU kullanılır.

## Evrimsel ekosistem (yeni ana katkı)

Tasarım ve sonuçlar: `docs/evrim_tasarim.md`. Hedefli NCA (aşağıdaki çekirdek katman) taban
çizgisi olarak kalır.

```bash
uv run python targets/creature_space.py                    # ata canlı uzayı galerisi
uv run python -m nca.train_ancestor configs/ancestor.yaml   # ata fiziği (~25 dk / 10k iterasyon)
uv run python -m nca.train_ancestor configs/ancestor.yaml --set init_from=runs/ancestor_stage1_s0/model.pt iterations=20000 lr=1.0e-3 lr_drop_iter=12000 damage.start_iter=0
uv run python experiments/e1_diversity.py --runs ancestor_s0          # E1/E4
(cd experiments && uv run python e3_evolution.py --run ancestor_s0)   # E3 (~2 saat)
(cd analysis && uv run python plots_evo.py)                           # şekiller + summary_evo.md
uv run python export/export_evo.py runs/ancestor_s0                   # demo/evo_models.js
xdg-open demo/evo.html                                                # ekosistem demosu
```

- `nca/evo.py` — genom koşullu kural, yara potansiyeli V, kalıtım, mutasyon
- `nca/world.py` — kaynak/enerji, alanla orantılı yırtıcı kesmeleri, kimlik etiketleri (bireyler
  kaynaşmaz), aynı bireyle sınırlı genom homojenleşmesi, soy ağacı
- `nca/train_ancestor.py`, `targets/creature_space.py` — ata eğitimi ve ata canlı uzayı
- `nca/train_evo.py` — ilk, hedefsiz yaklaşım (karşılaştırma; çeşitliliği renge yazdı)
- `demo/evo.html`, `evo.js`, `evo_gpu.js`, `evo_app.js` — tarayıcı ekosistemi. Kural WebGL2 ile GPU'da çalışır
  (`#cpu` adresiyle veya GPU yoksa JS'te), kalan fizik JS'te (adım fiziği Python ile parite testli;
  kesme demoda yumuşatıldı: kenar söner, parçalar adım adım ayrılır, kesilen doku altındaki kaynak
  serbest kalır. Bunlar `nca/world.py`de yok, yani E3/E5 sonuçları demonun kesmesiyle üretilmedi)

Özet: E1 ve E4 tuttu (görülmemiş genomların %100'ü yaşayabilir, hedefine IoU 0,967; kesilen
canlıların %100'ü yenilenir, yara mutasyonu genomu kaydırır). E3 (evrimsel iyileşme) tutmadı:
torunlar atalarıyla rekabette 0,50 ± 0,24 (n = 12). Tek-soy uygunluk testi (E5) kalıtsal fark
buldu (H² = 0,72) ve popülasyon uygunluğu 30k adımda %30–40 artıyor. μ taramasında μ = 1,0
uyumu hızlandırıyor (15k adımda 29,2'ye karşı mutasyonsuz 24,2). Mutasyonla doğan bir genom en iyi
kurucuyu %20 aşıyor (31,3 ± 2,2'ye karşı 26,0 ± 1,9) ama anlamlı değil (p ≈ 0,08). Yani en iyi
kurucunun ötesine geçen bir yenilik henüz kanıtlanmadı. Ayrıntılar: `docs/evrim_tasarim.md`.

```bash
(cd experiments && bash mu_sweep.sh)                                       # μ taraması (~5 saat)
(cd experiments && uv run python e5_fitness.py verify --reps 6)            # aday taraması
(cd experiments && uv run python e5_fitness.py confirm --reps 24 --top 5)  # kesin test
```

## Sonuçlar (çekirdek katman)

Güncel tablolar ve şekiller: `analysis/summary.md`, `analysis/figures/`. Tohum 3, model başına
100 örnek. Özet:

- Çekirdek: tüm rejimlerde büyüme IoU ≥ 0,99, 1000 adımda bozulma yok; R1–R3 dairesel onarım %100.
- D2 (yerinde): R0 parça küçüldükçe hızla çöküyor; R1 kısmen genelleşiyor (%50 parçada 0,86,
  %10'da 0,31 — H1); R2/R3 %20'ye kadar ≥ 0,96, %10'da keskin alt sınır (H2).
- D2 (bölünme): kesme eğitimi almamış R1'de %30–50 kesmelerde iki parça da 0,92–0,95
  oranında tam bireye dönüşüyor — bölünme dairesel onarım eğitiminden kendiliğinden çıkıyor.
- D4: eksen konumu gizli kanallardan doğrusal olarak R² ≈ 0,76–0,79 ile okunuyor (H3'ün 0,8
  eşiğinin hemen altı); 2. derece prob R² ≈ 0,97–0,99 — bilgi var ama doğrusal değil.
- D5: Janus (iki uçta baş) hiçbir koşulda çıkmadı; yüksek güçlü müdahaleler yaradan bağlı
  ikinci bir birey ("ikiz", %27–74) üretiyor. Baş dokusu içermeyen parça yeniden kesildiğinde
  bu morfoloji çoğu koşulda korunmuyor (%0–8); tek istisna R1'de tüm gizli kanallara 30 adımlık
  müdahale (%29). Janus olmadığı için H4 tutmuyor; kalıcılıkta zayıf, koşula bağlı bir iz var.
- D6 (esnek): tutmadı. Sıfırdan eğitim ölüm minimumuna düştü; R3'ten ince ayar ise birey yerine
  yarı saydam bulanık lekeler üretti (iyi birey 0). Karşılaştırma: gürültüyle hiç eğitilmemiş R3,
  seyrek gürültüde her hücreyi tohum gibi büyütüyor (ortalama 3,9 bileşen, birey IoU 0,53) ama
  bireyler çarpışıyor. Makalede gelecek çalışma (OpenAI-ES veya çarpışma cezası).

## Plandan bilinçli sapmalar ve notlar

- Analiz notebook yerine `analysis/plots.py` betiği; aynı şekilleri tekrarlanabilir üretir.
- Pool kapalı ablasyonunda her iterasyon tohumdan başladığı için hasar uygulanamaz; bu
  ablasyon "onarım pool olmadan güvenilmez" bulgusunu test eder.
- D5'te "baş değeri" D4 probunun baş bölgesinde ölçtüğü ortalama gizli kanal değerleridir;
  açgözlü seçimde ilk k kanal kullanılır. Kesme orta gövde parçasıdır (canlı kütlenin dikey
  %30–70 aralığı), Durant vd. (2017) gövde parçasına karşılık gelir.
- D6'da birey sayısı sapma cezası türevlenebilir değildir; eksik bireyler için boş yere konan
  "sanal" hedef pencereler türevlenebilir vekil olarak kullanılır. Sıfırdan eğitim (v1,
  `runs/noise_v1_failed_s0`) her şeyi öldüren gradyansız yerel minimuma düştü; v2 R3
  ağırlıklarından ince ayar yapar ve ölen havuz örneklerini yeniler. OpenAI-ES yedeği uygulanmadı.
- D2 bölünme modunda parçalar toplam 44 piksel ayrılır: daha az ayırmada (24 px) enine kesilen
  parçalar yeniden büyürken çarpışıp birleşiyor ve bölünme başarısı yapay olarak düşük ölçülüyordu.
- D5 sonuç sınıflarına "ikiz" eklendi (iki baş + kuyruk: yaradan ikinci tam birey). Sobel
  algılamalı kural yöne duyarlıdır; "başın altında gövde" sabit yönünü öğrenir, bu yüzden ters
  dönmüş bir baş (Janus) yerine yaradan aynı yönlü yeni bir birey başlatır. Kalıcılık testinde
  yeniden kesilen parça baş dokusu içermeyen gövde bölgesinden alınır; aksi halde parçada zaten
  bulunan ikinci baş sahte bir "kalıcılık" üretir.
- Yalnız Laplace (izotropik) ablasyonu sabit yönlü hedefi öğrenemez (IoU ≈ 0,42); izotropik
  NCA döndürmeye duyarsız bir kayıp gerektirir (Mordvintsev vd. 2022) — kapsam dışı.
