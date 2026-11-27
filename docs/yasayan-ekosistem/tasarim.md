# Yaşayan Ekosistem — Tasarım

Tarih: 25 Kasım 2026. Durum: kullanıcı tarafından onaylandı (25 Kasım 2026). Önceki yaklaşım (planarya atası):
`docs/evrim_tasarim.md`.

## Amaç ve neden

Ekosistemdeki canlıların yaratıcısı ağ ve evrim olmalı, biz değil. Planarya ata uzayı
(`targets/creature_space.py`) ve her türlü hedef görüntü kalkar. Ölçüt şekil çeşitliliği değil,
**canlılık**: demoya bakan biri gerçekten yaşayan bir ekosistem görmeli. Bugünkü dünyada
canlıların yaptığı hiçbir şey kendi işleri değildir. Hareket etmezler, üreme dışarıdan bir zar
kesmesidir (`World.predation`), yırtıcı bir canlı değildir.

Kullanıcı kararı: büyük risk bilerek alınır ("adam gibi bir proje"). Yaklaşım C (melez) seçildi.

## Hedefler

Aşağıdaki dört davranış elle yazılmadan, ağın yerel kuralından ve evrimden çıkmalı:

1. **Hareket**: canlı bedeni kaynağa doğru kayar ya da kalabalıktan kaçar.
2. **Kendiliğinden bölünme**: zar olmadan, canlının kendi kuralıyla ikiye ayrılması.
3. **Birbirini yeme**: bir canlının hücreleri başka bir canlının hücrelerini tüketip enerjisini alır.
4. **Ölüm ve yaşam döngüsü**: aç kalan küçülür ve ölür, maddesi ekosisteme geri döner.

**Başarı (makale için):** atada olmayan ya da nadir olan en az bir davranış, mutasyonlu
koşullarda mutasyonsuz kontrole göre anlamlı biçimde artmalı, kalıtsal olmalı ve demoda
görülebilmeli. Hiçbiri çıkmazsa sonuç olumsuz olarak dürüstçe raporlanır (katmanlı kapsam).

## Hedef olmayanlar

3B, biyolojik gerçekçilik, eşeyli üreme, türe özel ayrı ağlar, tarayıcıda eğitim, çekirdek
katmanın (planarya NCA, D1–D6) değiştirilmesi. Çekirdek katman taban çizgisi olarak aynen kalır.

## Anti-hedefler (çalışsa bile başarısızlık sayılır)

- Elle yazılmış davranış: hareket, saldırı ya da bölünmeyi tetikleyen bizim yazdığımız bir
  kural. Fizik yalnızca davranışı *mümkün* kılar, *ne zaman* yapılacağına kural karar verir.
- Şekil, renk veya davranış hedefi içeren bir eğitim kaybı.
- Üremenin asıl yolunun dış kesmeler olarak kalması.
- Hiçbir şeyin kıpırdamadığı bir demo.

## Kısıtlar

- Teslim: plan yaklaşık 22 Aralık 2026'da bitiyor, sunum için canlı tarayıcı demosu ve IEEE
  makalesi gerekiyor. Kalan süre yaklaşık 4 hafta.
- Donanım: RTX 5060 Laptop (8 GB). `torch.compile` kullanılamaz. Paralel süreçler hızlandırmaz.
- Demo sunucusuz çalışmalı ve kural WebGL2 ile GPU'da koşmalı (bugünkü `evo_gpu.js` gibi).
  Python ile JS arasında adım paritesi test edilir (`tests/test_js_parity.py` deseni).
- Kod yorumları ve belgeler Türkçe yazılır.

## Yaklaşımlar

| | Yaklaşım | Artısı | Eksisi |
|---|---|---|---|
| A | Saf nöroevrim: genom kuralın tüm ağırlıkları | En dürüst "ağ yarattı" hikayesi | Soğuk başlangıç; rastgele ağlar ölür veya dünyayı doldurur; haftalarca sonuç çıkmama riski |
| B | Ortak kural gradyanla "yaşamayı" öğrenir, genom yalnızca girdi | En düşük risk, kodun çoğu kalır | Eski hedefsiz denemenin çöktüğü yer: genom ucuz bir eksene (renk) yazılır |
| **C** | **Melez: önce gradyanla şekilsiz "en basit canlı", sonra kural ağırlıkları kalıtsal ve mutasyona açık** | Soğuk başlangıcı B çözer, yaratıcılığı A verir | Hücre başına farklı kural, simülasyonu ve gölgelendiriciyi karmaşıklaştırır |

**Seçilen: C.** 4 haftada bir şey göstermemizi sağlayan tek yol bu.

## Tasarım

### 1. Durum

Hücre başına `(B, C, H, W)` kanalları:

| Kanal | Kim yazar | Anlamı |
|---|---|---|
| `m` kütle (≥ 0) | fizik (taşınım, büyüme, yeme) | Bugünkü alfanın yerini alır. Hücre `m > m_min` iken canlıdır, görünür alfa `clamp(m)` |
| `RGB` | kural | Görünürdür ve tüm komşular algılar (uyarı rengi gibi sinyaller evrimleşebilir) |
| `E` enerji (kütle başına) | fizik | Kaynaktan alınır, bakım ve eylemlerle harcanır |
| `V` yara potansiyeli | fizik | Bugünkü `V_CH` (`nca/evo.py`) aynen kalır |
| gizli `h` | kural | Bugünkü gizli kanallar |
| `tag` kimlik | fizik | Bugünkü kimlik etiketi ve segmentasyon (`World.segment`, `new_tag`) |
| `g` genom (d_g = 32) | fizik (kalıtım, mutasyon) | Kuralı değiştirir (bkz. 3) |

Dünya alanı: kaynak `R`.

### 2. Algı

Kuralın girdisi bugünkü algılama filtreleridir (`sobel_lap`), şu kanallar üzerinde:
`[m, RGB, V, h]`. Buna fiziğin sağladığı üç algı daha eklenir:

- yerel `R`,
- **yabancı kütle alanı** `F`: farklı `tag`'li komşu kütlesinin toplamı ve gradyanı,
- hücrenin kendi `E` değeri.

Hareket ve saldırı ancak kural yiyeceği ve yabancıyı algılayabiliyorsa evrimleşebilir.

### 3. Kural ve kalıtsal ağırlıklar

Kural bugünkü `EvoCA` ile aynı iki katmanlı 1x1 ağdır. Genom, gizli katmanı FiLM ile
değiştirir:

```
z  = W1 · p + b
z' = z ⊙ (1 + P_s g) + P_b g        # P_s, P_b: sabit, tohumlanmış rastgele izdüşümler
y  = W2 · relu(z')
```

- `W1`, `b`, `W2` ön eğitimde öğrenilir. `P_s` ve `P_b` hiç eğitilmez.
- Evrim, ağırlık uzayının 2 × 32 boyutlu rastgele bir alt uzayında arama yapar (alt uzay
  nöroevrimi; Li vd. 2018 "intrinsic dimension"). Hücre başına farklı kural olur ama soy
  tablosu gerekmez. `g` bugünkü gibi bir doku olarak gölgelendiriciye gider.
- Ön eğitimde `g = 0` kullanılır. Kural `g`'ye dayanıklı olacak biçimde eğitilmez, çünkü
  dayanıklılık `g`'yi yok sayan bir kural demektir.
- İzdüşüm ölçeği `σ_P` ayarlanır: `|g| ≈ 1` canlıyı ölçülebilir biçimde değiştirmeli ama
  çoğunu öldürmemeli. Ölçüt: rastgele `g` örneklerinin yaklaşık yarısı 2000 adım yaşamalı.

Kural çıktıları:

- `Δ[RGB, h]`,
- büyüme `γ ≥ 0`: enerjiyi kütleye çevirme isteği,
- akış logitleri (9 tane: kal + 8 yön): kütlenin nereye gideceği, softmax kesirleri olarak,
- saldırı `a ∈ [0, 1]`, `a = sigmoid(y_a − 4)` biçiminde. Ön eğitimde yabancı yoktur ve
  `a`'ya gradyan gelmez. Bu kaydırma olmasaydı sıfır başlatılmış `W2` herkesi `a = 0,5` ile
  saldırgan başlatırdı. Böylece ata saldırmaz, saldırıyı evrim bulur.

### 4. Adım fiziği (sıra önemli)

1. **Kural**: bugünkü gibi stokastik ateşleme maskesiyle uygulanır.
2. **Taşınım**: her hücre kütlesini akış kesirlerine göre dağıtır. `RGB`, `h`, `g` ve `E`
   kütle ağırlıklı olarak taşınır. Alıcı hücre en büyük gelen katkının `tag`'ini alır. Başka
   bir `tag`'in tuttuğu hücreye giren kütle geri döner, yani bireyler kaynaşmaz (bugünkü
   `self_tol` mantığı). Boş hücreye kütle girerse orada yeni bir hücre doğar, genomu kütle
   ağırlıklı ortalamadır. Bu, bugünkü `inherit`'in yerini alır. Gölgelendirici için toplama
   (gather) biçiminde iki geçiş kullanılır: önce çıkış kesirleri yazılır, sonra alıcı toplar.
3. **Yeme**: saldırısı `a` olan hücre, her yabancı komşusundan `k_eat · a · m_kurban`
   alır. Birden çok saldırgan varsa pay bölünür. Alınan kütle `η < 1` verimle saldırganın
   enerjisine eklenir, saldırgan adım başına `c_attack · a` öder. Kurbanın kaybettiği kütle
   `V`'yi yükseltir, yani yenen canlının yarası mutasyona yol açar.
4. **Metabolizma ve büyüme**: alım `∝ m · R`, bakım `∝ m`, hareket maliyeti `∝` taşınan
   kütle, büyüme `γ`'ye göre `E`'yi `m`'ye çevirir. Yeni hücreler büyüyen kütlenin boş
   komşulara akmasıyla oluşur.
5. **Ölüm**: `E ≤ 0` olan hücrenin kütlesi `R`'ye çürür. `m < m_min` olan hücre ölür,
   içeriği `R`'ye döner.
6. **Bölünme ve segmentasyon**: aynı `tag`'li beden iki ya da daha çok bileşene ayrılırsa ve
   her biri `min_viable` boyutundaysa küçük parçalara yeni `tag` verilir. Bölünme olayı
   nedeniyle birlikte kaydedilir: `attack` (son K adımda o bedene saldırı kaybı oldu),
   `disturbance` (dış kesme) ya da `self` (hiçbiri).
7. **Mutasyon ve homojenleşme**: bugünkü `mutate` (`g += μ V ξ`) ve beden içi
   `homogenize` aynen kalır.
8. **Dış karışıklık, kademeli söndürme (kullanıcı kararı)**: bugünkü `cut`, düşük oranda bir
   "hava olayı" olarak başlar ve koşu boyunca doğrusal olarak sıfıra iner (örneğin koşunun
   ilk yarısında). Ata bölünmeyi bilmediği için başlangıçta üreme yolu budur. Evrim, sönen
   kesmenin yerine kendi üreme yolunu (kendiliğinden bölünme ya da yeme yoluyla parçalanma)
   koymak zorundadır. Popülasyonun kesmeler sıfırlandıktan sonra sürmesi makalenin ana
   iddiasıdır. Sönmesi de olumsuz sonuç olarak raporlanır.

**Korunum**: besin kapalı bir döngüdür, `R ↔ E ↔ m`. Bakım ve hareket maliyeti `R`'ye
geri döner. `ΣR + Σ(m·(1+E))` (tam biçimi uygulamada sabitlenecek) her adımda sabit kalır ve
testle doğrulanır. Patlama ve çöküş döngüleri bu kısıttan çıkar.

### 5. Ön eğitim: "en basit canlı"

Bugünkü `train_ancestor.py` deseni kullanılır (pool ve hasar), ama hedef görüntü yoktur.
Dünyada yalnızca tek bir canlı ve düzgün kaynak bulunur, yeme etkin değildir, `g = 0`'dır.
Taşınım softmax olduğu için türevlenebilir. Tek canlı olduğundan `tag` sabittir.

Kayıp yalnızca canlılık ölçütlerinden oluşur:

- kütle `[A_min, A_max]` aralığında,
- döküntü yok: `min_viable`'dan küçük bileşenlerdeki kütle cezalandırılır. `train_evo.py`'deki
  "tek beden" terimi kullanılmaz, çünkü o terim ataya kendiliğinden bölünmemeyi öğretir.
  Yaşayabilir iki bedene ayrılmak cezasızdır.
- enerji pozitif,
- homeostaz **beden çerçevesinde**: ağırlık merkezine hizalanmış betimleyiciler (alan,
  ortalama renk, uzama) durağan olmalı. Böylece hareket cezalandırılmaz ama ödüllendirilmez de.
- hasardan sonra yine bu ölçütlere dönüş.

Şekil, renk ve davranış hedefi yoktur. Atanın neye benzeyeceğini eğitim belirler.

### 6. Ekosistem

Bugünkü `World` yeni fiziğe taşınır. 192², toroidal olmayan dünya, 12 tohum korunur.
Kayıtlara birey başına şu betimleyiciler eklenir:

- hız (ağırlık merkezinin yer değiştirmesi / adım),
- avcılık payı (enerji alımında yemenin payı),
- `self` bölünme oranı,
- ömür, boyut, şekil (alan, uzama).

Soy ağacı bugünkü gibi kaydedilir.

### 7. Deneyler

| Kod | Soru | Kontroller |
|---|---|---|
| Y1 | Ata yaşayabilir mi (tek beden, 2000 adım, hasardan dönüş)? | — |
| Y2 | Davranışlar evrimsel zamanda değişiyor mu? Hız, avcılık payı, `self` bölünme, ömür | μ = 0; `g` rastgele örnekleme (seçilim mi, sürüklenme mi?) |
| Y3 | Ekolojik dinamik: popülasyon ve avcı oranı zaman serileri, salınım var mı? | μ = 0 |
| Y4 | Kesmeler söndükten sonra popülasyon sürüyor mu (üreme kendi işleri mi oldu)? Söndürme sonrası birey sayısı ve bölünme nedenleri (`self` / `attack`) | ata aynı söndürmeyle μ = 0; ata karışıklık kapalı |
| Y5 | Davranışlar kalıtsal mı? Bugünkü E5 tek-soy testi, davranış betimleyicileriyle | — |

Tohum sayısı en az 2 olmalı, mümkünse 3. Uzun koşular en az 100 bin adım (süre 1. haftada
ölçülecek).

### 8. Demo

`demo/evo.*` yeni fiziğe taşınır:

- Gölgelendirici FiLM'li kuralı `g` dokusuyla çalıştırır. Taşınım iki geçişle yapılır.
- Kalan fizik (yeme, metabolizma, segmentasyon) bugünkü gibi JS'tedir.
- Araçlar korunur: kes, sil, tohum. "Bir canlıyı izle" görünümü eklenir: birey seçilince hızı,
  avcılık payı ve soyu gösterilir.
- Python ile JS arasında parite testi yapılır.

### 9. Eski yaklaşımın durumu

Planarya atalı ekosistem (E1–E5, μ taraması) silinmez. Kodu ve sonuçları depoda kalır ve
makalede "ilk yaklaşım ve neden terk edildi" olarak kısaca anlatılır. Onaydan sonra
`docs/evrim_tasarim.md`'nin başına ve README'nin "Evrimsel ekosistem" bölümüne bu belgeye
bağlantı eklenir.

## Takvim ve karar noktası

Uygulama planı (karar noktasına kadar): `docs/yasayan-ekosistem/plan-1-fizik.md`.
Plan 2 (Y2–Y5, demo, makale): `docs/yasayan-ekosistem/plan-2-deney-demo-makale.md`.

| Hafta | Bitiş | İş |
|---|---|---|
| 1 | 2 Aralık | Fizik (taşınım, yeme, enerji döngüsü), korunum testleri, ön eğitim, Y1 |
| 2 | 9 Aralık | Ekosistem, `σ_P` ayarı, ilk evrim koşuları. **Karar noktası** |
| 3 | 16 Aralık | Y2–Y5 ve kontroller, şekiller, demo taşıma |
| 4 | 22 Aralık | Makale ve sunum |

**Karar noktası (9 Aralık):** atada olmayan bir davranış işareti yoksa yeni deneylere girilmez.
Elde ne varsa olumsuz sonuç olarak yazılır. E1–E5 ve çekirdek katman makaleyi taşır.

## Uygulayıcıya bırakılanlar

Sabitlerin sayısal değerleri (`k_eat`, `η`, `c_attack`, maliyetler, `m_min`, `A_min/A_max`),
kanal sayıları, kütle ağırlıklı taşınımın kesin sayısal biçimi, korunum ifadesinin tam biçimi,
betimleyicilerin kesin tanımları, gölgelendirici geçişlerinin düzeni.

## Tasarımcının kararları (kullanıcı onayladı)

- FiLM ile rastgele alt uzay (soy başına ağırlık tablosu yerine), d_g = 32, ön eğitimde `g = 0`.
- Algıya `R`, yabancı kütle alanı ve `E` eklenir.
- Bireyler kaynaşmaz: yabancı hücreye giren kütle geri döner.
- Kapalı besin döngüsü ve korunum testi.
- Homeostaz beden çerçevesinde ölçülür (hareket cezalandırılmaz).
- Yenen canlının kaybı `V`'yi yükseltir (yara mutasyonu yemeye bağlanır).
- Eski planarya ekosistemi silinmez, makalede karşılaştırma olarak kalır.
- Ön eğitimde "tek beden" yerine "döküntü yok" terimi kullanılır (ata bölünmeye karşı eğitilmez).
- Saldırı çıktısı sıfıra yakın başlatılır (ata saldırmaz).

## Kullanıcı kararları

- Yaklaşım C (melez).
- Dış kesmeler kademeli söndürülür (4.8, Y4).

## Karar noktası (9 Aralık) — ön rapor, 26 Kasım

Ayrıntılar: `analysis/summary_life.md`, `analysis/figures/life_timeseries.png`, `runs/lifeworld_*`.

**Çalışan kısım.**
- Ata (`runs/life_s0`, checkpoint 9000) Y1'i geçiyor: 2000 adımda canlılık 1,00, kesmeden sonra
  yeniden büyüme 1,00. Checkpoint Y1'e bakılarak seçildi, bu yüzden Y1 bağımsız test sayılmaz.
- Elle yazılmamış hareket var: ata, kuyruklu bir damla olarak sürekli yer değiştiriyor.
- Dört koşunun hiçbiri tükenmedi. Kapalı besin döngüsü 100k adım korundu.
- Evrim koşusu mu1 s0'da ~10k–45k arasında renkleri ayrışmış serbest bir popülasyon oluştu,
  kendiliğinden bölünme arttı (~20/1000 adım). ~33k'da avcılık payı 0,23'e sıçradı; mu0'da bu değer
  0,01 düzeyinde kaldı.
- Ön eğitimin 8000. checkpoint'i boş arenada kendiliğinden çoğalıyor (~5 beden). Bölünmenin ağın
  ulaşabileceği bir davranış olduğunu gösteriyor.

**Çalışmayan kısım.**
1. *Geometri uyumsuzluğu.* Ata sarmal arenada eğitildi, dünya kenarlı. Hareket mutlak bir yöne
   sabit, bu yüzden bireyler köşeye birikiyor. Geç dönemde canlı maddenin %86–100'ü duvar şeridinde,
   geç pencere metrikleri büyük ölçüde artefakt.
2. *Mutasyon kaynağı zayıf.* μ = 1 kurucuları öldürdü, μ 0,1'e indirildi. Genom çeşitliliği
   (|g| ≈ 0,2) ilk büyümede oluşup sonra sabit kalıyor: dengede V ≈ 0,05. Kesmeler sönünce yeni
   varyasyon pek oluşmuyor.
3. *Tek tohum.* mu1 ile mu0 arasındaki fark tek bir tohumdan geliyor (n = 2).

**Ön karar: koşullu devam.** Fizik ve öğrenilmiş ata çalışıyor; olumsuz sonuç ilan etmek için
erken. Asıl engel geometri ve doğrudan düzeltilebilir. Plan 2'nin ilk işleri:
(a) sarmal dünya: kenardan sarmalanan segmentasyon, kesme ve ekim; (b) koşuları yeniden
yapmak ve tohum sayısını 4'e çıkarmak; (c) gerekirse mutasyonu yara potansiyelinden bağımsız küçük
bir taban terimle beslemek. (c) elle yazılmış davranış değildir, ancak tasarım değişikliği olduğu
için kullanıcı onayı gerekir.

**Kullanıcı kararı (26 Kasım):** (a), (b) ve (c) onaylandı. Uygulama: dünya `phys.torus` ile sarmal
(`label_wrap`, dairesel merkez, sarmal kesme ve taşıma, yeme ve V yayılımı sarmal);
`LifeParams.mu_base = 0,03` (`evo.mutate` `base`: g += μ(V + base)ξ). 5000 adımlık sınama:
base 0 -> 49 birey, base 0,03 -> 51 birey ve |g| 0,30 -> 0,35, base 0,1 -> t = 1360 tükenme.

### Sarmal dünya sonuçları (27 Kasım)

Ayrıntılar `analysis/summary_life_torus.md` ve `analysis/figures/life_timeseries_torus.png`. Duvar
sorunu çözüldü. Mutasyonun erken etkisi (daha çok bölünme, hız ve avcılık) üç tohumda tutarlı. Ancak
kesmeler söndükten sonra dünya doyuyor: ata ~2000 adımdan sonra büyüyüp bölünmeyi sürdürüyor ve
besin bitene kadar arenayı bir halı gibi kaplıyor. Ata, besin kıtlığında (R0 ≤ 0,35) tek hücreden
büyüyemiyor. Önerilen sonraki adım ata ön eğitimini kıtlığa ve uzun ufka dayanıklı hale getirmek:
örnek başına rastgele başlangıç besini, daha büyük arena ve uzun ufuklu bir Y1. Kullanıcı kararı bekleniyor.

**Kullanıcı kararı (27 Kasım):** Plan A, yani kıtlığa ve uzun ufka dayanıklı ata (v2). Seçilen ata
`runs/life2ft_s0` checkpoint 3000. Dünya ayarları: film_scale 0,5 ve μ 0,05.

### Ata v2 evrim sonuçları (27 Kasım)

Ayrıntılar `analysis/summary_life_life2.md` ve `analysis/figures/life_timeseries_life2.png`.
Sekiz koşunun (2 koşul × 4 tohum, 100k adım) hiçbiri tükenmedi. Halı sorunu mutasyonsuz koşulda
çözüldü: ~300 birey, kapsama %31. Mutasyon dört tohumda aynı yöne seçilim üretiyor
(tohum çiftleri arası genom kosinüsü 0,73, p < 5e-5). Seçilen yön K-stratejisi: büyük (~180 hücre),
yavaş, seyrek bölünen ve uzun yaşayan bedenler. Bu bedenler besini daha düşük düzeye indiriyor
(R* 0,08, mutasyonsuzda 0,18) ve dünyanın %74'ünü kaplayan bir bölge mozaiği kuruyor.

Makale için bu temiz ve tekrarlanmış bir sonuç: evrim kendiliğinden ortaya çıkıyor, paralel
evrim gözleniyor ve R* ilkesi geçerli. Ancak "yaşayan ekosistem" hedefi açısından geç dönem
durağan (hız 0,007). Açık sorular: seçilim birey düzeyinde mi hücre düzeyinde mi işliyor
(mutasyon somatik), ve bozulma rejimi (kesmeler) söndürülmeseydi dinamik bir denge kalır mıydı?
Kullanıcı kararı: soy ağacı analizi, ardından Plan 2.

**Soy düzeltmeli tekrar (life2l, 27 Kasım).** Dört yeni mu1 koşusu aynı yöne evrildi (life2 ortak yönüyle
kosinüs 0,75–0,84). Price ayrıştırmasında kaymanın %60–68'i birey düzeyi seçilimden, kalanı aynı yönde
beden içi değişimden geliyor. Düzeltilmiş ölçülerle evrimleşen bedenler kişi başı 4,7 kat seyrek bölünüyor
ve 1,7 kat uzun yaşıyor (life2'deki 5,7 kat izleme hatasıydı). Ayrıntılar `analysis/summary_life_life2l.md`.
