# Yaşayan ekosistem — karar noktası özeti

Erken pencere: ilk 5000 adım (ataya yakın). Geç pencere: kesmeler söndükten (t > 50000) sonraki
son %20. Kendiliğinden bölünme 1000 adım başına; avcılık payı = yemeyle alınan / (yeme + kaynak alımı).

| koşul | tohum | son adım | tükendi | birey erken → geç | hız erken → geç | avcılık erken → geç | kendiliğinden bölünme erken → geç |
|---|---|---|---|---|---|---|---|
| mu0 | 0 | 100000 | hayır | 11.7 → 312 | 0.0894 → 0.0308 | 0.00652 → 0.00864 | 7.8 → 17.6 |
| mu0 | 1 | 100000 | hayır | 22.1 → 306 | 0.102 → 0.0313 | 0.00415 → 0.00867 | 19.2 → 17.1 |
| mu0 | 2 | 100000 | hayır | 18.9 → 312 | 0.102 → 0.0306 | 0.00233 → 0.00785 | 13.6 → 17.4 |
| mu0 | 3 | 100000 | hayır | 32.8 → 319 | 0.0953 → 0.0307 | 0.005 → 0.00922 | 23.4 → 16.9 |
| mu1 | 0 | 100000 | hayır | 75 → 180 | 0.0779 → 0.00695 | 0.0154 → 0.0213 | 55.8 → 4.9 |
| mu1 | 1 | 100000 | hayır | 65.6 → 149 | 0.0754 → 0.00728 | 0.0169 → 0.0203 | 42.2 → 3.8 |
| mu1 | 2 | 100000 | hayır | 41.5 → 144 | 0.106 → 0.00692 | 0.009 → 0.0184 | 45.4 → 3.15 |
| mu1 | 3 | 100000 | hayır | 81.7 → 145 | 0.103 → 0.00652 | 0.012 → 0.0205 | 91.2 → 2.9 |

## Yorum (27 Kasım) — ata v2, 2 koşul × 4 tohum, 100k adım

Geç pencere t ≥ 90000. Kapsama = canlı hücre / 192².

| | mu0 (mutasyonsuz) | mu1 (μ = 0,05, taban 0,03) |
|---|---|---|
| tükenme | 0/4 | 0/4 |
| birey (20–30k → geç) | ~282 → ~309 | ~258 → ~145 |
| ortalama beden (hücre) | 38 → 37 | 75 → 180 |
| kapsama | %29 → %31 | %52 → %74 |
| ortalama besin R | 0,18 | 0,08 |
| hız (hücre/adım) | 0,031 | 0,007 |
| avcılık payı | 0,009 | 0,020 |
| ortalama yaş (adım) | ~5400 | ~30800 |
| toplam besin kayması | < 1e-4 | < 1e-4 |

**Önceden yazılan beklentiye göre:** tükenme yok ✓. mu0 besin sınırında ~300 bireyde, kapsama
%31 ✓ (halı yok). mu1 mu0'dan açıkça farklı ✓, ama beklenen yönde değil: nüfus 145'e iniyor ve
kapsama %74'e çıkıyor ✗. Bu tekdüze bir halı değil, sınırlarla ayrılmış ~145 bireyden oluşan bir
bölge mozaiği (`runs/life2_mu1_s0/snap_100000.png`). mu0 ise atanın sabit yönelimiyle aynı
köşegen yönde akan ince şeritlerden oluşuyor (`runs/life2_mu0_s0/snap_100000.png`).

**Paralel evrim.** mu1'de nüfusun ortalama genomu dört bağımsız tohumda aynı yöne kayıyor:
32 bileşenin 18'i dört tohumda da aynı işaretli (yansız sürüklenmede beklenen 4). Tohum çiftleri
arasındaki kosinüs benzerliği ortalama 0,73 (rastgele yönlerde ±0,18). İşaret çevirmeli sıfır
modelinde p < 5e-5 (20000 permütasyon). Ortak bileşenler g12 > 0, g21 < 0, g27 < 0 ve g29 > 0.
Mutasyon gürültüsü işaretçe simetrik olduğundan ortak yön seçilimden geliyor.

**Seçilen strateji: r'den K'ya.** Evrimleşen soylar daha büyük, daha yavaş, daha seyrek bölünen
ve daha uzun yaşayan bedenler. Besini daha düşük bir düzeye (R 0,08, mu0'da 0,18) indirebiliyor
ve bölge tutuyorlar. Bu Tilman'ın R* ilkesine uyuyor: kıt kaynak rekabetini, kaynağı en düşük
düzeyde sürdürebilen tip kazanır. Beden büyümesinin çoğu dış kesmeler söndükten sonra geliyor
(50k'da 102 hücre, 100k'da 160). Yani bozulma rejimi büyük bedeni frenliyordu.

**Açık sorular ve sınırlar:**
1. Mutasyon somatik: yaşayan bedenin hücrelerinde yerinde işliyor. Gözlenen seçilim kısmen
   beden içi hücre soyları arasında da olabilir (büyümeyi artıran genom beden içinde yayılır).
   Birey düzeyi ile hücre düzeyi seçilimini ayırmak için soy ağacı analizi gerekir; yapılmadı.
2. Geç dönem durağan: mu1'de hız 0,007, kendiliğinden bölünme 1000 adımda ~4. "Yaşayan
   ekosistem" hedefi açısından evrim hareketli bir dünyaya değil, yerleşik bir mozaiğe yakınsıyor.
3. `mean_generation` ölçüsü ~2000 bölünmeye rağmen 1'in altında kalıyor. Olası neden
   (doğrulanmadı): `min_org_area` altında kopan tomurcuklar sonradan büyüyünce ebeveynsiz
   doğmuş sayılıyor ve soy takibi kopuyor. Ölçü yorumda kullanılmadı; soy ağacı analizi
   (madde 1) bu düzeltmeyi gerektirir.
4. `orgs.json` kayıtları boyut nedeniyle gzip'lendi (`orgs.json.gz`).

**Düzeltme (27 Kasım, `6ee8adc`):** Bu koşulardaki soy izleme, 20 hücreden küçük tomurcukları ve 10 adımda
boyundan fazla kayan küçük bedenleri kaybediyordu (mu1_s0: 18828 doğumun 16074'ü ebeveynsiz). Bu yüzden
bu tablodaki bölünme sayısı, ortalama yaş ve kuşak ölçüleri yanlı; nüfus, alan, kapsama, R, hız ve genom
ölçüleri etkilenmez. Düzeltilmiş ölçüler `runs/life2l_*` ve `summary_life_life2l.md`'de.
