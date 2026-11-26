# Yaşayan ekosistem — karar noktası özeti

Erken pencere: ilk 5000 adım (ataya yakın). Geç pencere: kesmeler söndükten (t > 50000) sonraki
son %20. Kendiliğinden bölünme 1000 adım başına; avcılık payı = yemeyle alınan / (yeme + kaynak alımı).

| koşul | tohum | son adım | tükendi | birey erken → geç | hız erken → geç | avcılık erken → geç | kendiliğinden bölünme erken → geç |
|---|---|---|---|---|---|---|---|
| mu0 | 0 | 100000 | hayır | 13.8 → 10.7 | 0.0203 → 0.012 | 0.00161 → 0.00815 | 1.2 → 0.15 |
| mu0 | 1 | 100000 | hayır | 13.2 → 23.1 | 0.0292 → 0.0107 | 0.0018 → 0.00315 | 1.8 → 0.1 |
| mu1 | 0 | 100000 | hayır | 7.02 → 29.8 | 0.0295 → 0.00652 | 0.00318 → 0.227 | 0.8 → 0.4 |
| mu1 | 1 | 100000 | hayır | 9.7 → 20.5 | 0.0261 → 0.0112 | 0.00794 → 0.00584 | 2.2 → 0.2 |

## Uyarı: duvar artefaktı (geç pencere güvenilmez)

Ata sarmal arenada eğitildi, dünya ise kenarlı (192², `LifeWorld` `torus=False`). Atanın hareketi
mutlak bir yöne sabit (galerideki kuyruklar hep aynı köşegende). Bu yüzden bütün bireyler sol üst
köşeye sürükleniyor ve duvar boyunca yayılan, yaşayan şeritlere dönüşüyor. Anlık görüntülerde,
kenara 16 hücreden yakın şeridin (alanın ~%31'i) taşıdığı canlı piksel payı:

| koşul/tohum | 5k | 25k | 50k | 100k |
|---|---|---|---|---|
| mu1 s0 | %48 | %29 | %45 | %86 |
| mu1 s1 | %100 | %95 | %100 | %100 |
| mu0 s0 | %79 | %100 | %88 | %100 |
| mu0 s1 | %52 | %90 | %97 | %89 |

Tablodaki geç pencere değerleri büyük ölçüde duvardaki yapıları ölçüyor. Serbest yaşayan bir
popülasyon yalnızca mu1 s0'da ~10k–45k arasında var (`snap_035000.png`: renkleri ayrışmış, ~30
ayrı beden). Avcılık payındaki artış (0,23) ve kendiliğinden bölünme patlaması (~10k) da tek bir
tohumdan geliyor (mu1 s1'de görülmüyor).
