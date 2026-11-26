# Yaşayan ekosistem — karar noktası özeti

Erken pencere: ilk 5000 adım (ataya yakın). Geç pencere: kesmeler söndükten (t > 50000) sonraki
son %20. Kendiliğinden bölünme 1000 adım başına; avcılık payı = yemeyle alınan / (yeme + kaynak alımı).

| koşul | tohum | son adım | tükendi | birey erken → geç | hız erken → geç | avcılık erken → geç | kendiliğinden bölünme erken → geç |
|---|---|---|---|---|---|---|---|
| mu0 | 0 | 100000 | hayır | 12.6 → 34.7 | 0.0199 → 0.0109 | 0.000572 → 0.00202 | 0.8 → 3.05 |
| mu0 | 1 | 100000 | hayır | 11.8 → 25.7 | 0.0172 → 0.011 | 0 → 0.00218 | 0 → 2.35 |
| mu0 | 2 | 100000 | hayır | 11.8 → 30.1 | 0.0179 → 0.011 | 3.97e-07 → 0.00235 | 0 → 2.95 |
| mu0 | 3 | 100000 | hayır | 11.8 → 36.5 | 0.0175 → 0.0117 | 0 → 0.00271 | 0 → 3.1 |
| mu1 | 0 | 1700 | evet | 4.39 → nan | 0.225 → nan | 0.00612 → nan | 11.8 → nan |
| mu1 | 1 | 100000 | hayır | 6.09 → 50.4 | 0.258 → 0.0113 | 0.00282 → 0.00413 | 32.8 → 13.2 |
| mu1 | 2 | 100000 | hayır | 10.7 → 18.8 | 0.139 → 0.00727 | 0.00155 → 0.000802 | 25.8 → 2.35 |
| mu1 | 3 | 100000 | hayır | 23.6 → 35 | 0.171 → 0.00985 | 0.00776 → 0.00734 | 57.4 → 7.5 |

## Yorum (sarmal dünya, 4 + 4 tohum)

- **Duvar sorunu çözüldü.** Bedenler arenaya dağılıyor; 20k–45k arasında ~90–100 serbest birey var.
- **Mutasyonun erken etkisi tutarlı.** Mutasyonlu koşulda ilk ~40k adımda kendiliğinden bölünme
  (~60–75/1000 adım; mu0'da 0–25), hız (0,1–0,3; mu0'da 0,02) ve avcılık payı (~0,02; mu0'da ~0,01)
  daha yüksek, birey sayısı daha hızlı artıyor. Üç tohumun üçünde de görülüyor.
- **Tükenme.** mu1 tohum 0, t = 1700'de başlangıç darboğazında tükendi (taban mutasyonu ilk büyümede de etkili).
- **Doygunluk (asıl sorun).** Kesmeler söndükten sonra iki koşul aynı noktada buluşuyor: bedenler
  400–1000 hücreye büyüyor ve canlı doku arenanın ~%70'ini kaplıyor (anlık görüntüler: halı). Tanı
  koşusu (kesme ve mutasyon yok, 20k adım): ilk 2000 adım kararlı (alan ~40, Y1'in ölçtüğü aralık),
  sonra bedenler büyüyüp bölünüyor (12 → 60 birey) ve besin tükenene kadar sürüyor (ortalama R 0,97 → 0,40).
  Taşıma kapasitesini yalnızca toplam besin belirliyor.
- **Ata dar.** Başlangıç besini 0,1, 0,2 ya da 0,35 olunca tek hücreden hiç büyüyemiyor. Ata yalnız
  eğitildiği bol besinli (R = 1) koşulda yaşıyor. Besini kısarak taşıma kapasitesini düşürmek bu
  ata ile mümkün değil.
- Küçük not: toplam besin 100k adımda ~%0,01 kayıyor (36888 → 36893); kesme sırasında geri dağıtımdan
  geldiğini düşünüyorum, doğrulamadım.
