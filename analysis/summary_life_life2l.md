# Yaşayan ekosistem — karar noktası özeti

Erken pencere: ilk 5000 adım (ataya yakın). Geç pencere: kesmeler söndükten (t > 50000) sonraki
son %20. Kendiliğinden bölünme 1000 adım başına; avcılık payı = yemeyle alınan / (yeme + kaynak alımı).

| koşul | tohum | son adım | tükendi | birey erken → geç | hız erken → geç | avcılık erken → geç | kendiliğinden bölünme erken → geç |
|---|---|---|---|---|---|---|---|
| mu0 | 0 | 100000 | hayır | 21.1 → 317 | 0.0961 → 0.0302 | 0.00384 → 0.00838 | 87 → 92.9 |
| mu0 | 1 | 100000 | hayır | 33 → 320 | 0.0976 → 0.0305 | 0.00471 → 0.00774 | 156 → 89.7 |
| mu0 | 2 | 100000 | hayır | 18.4 → 315 | 0.1 → 0.0305 | 0.00243 → 0.00802 | 82.8 → 94.2 |
| mu0 | 3 | 100000 | hayır | 17.1 → 306 | 0.101 → 0.0313 | 0.00253 → 0.0083 | 69 → 97.8 |
| mu1 | 0 | 100000 | hayır | 76.2 → 170 | 0.108 → 0.00787 | 0.0153 → 0.0204 | 325 → 13.2 |
| mu1 | 1 | 100000 | hayır | 55.5 → 186 | 0.104 → 0.00628 | 0.0112 → 0.0268 | 234 → 9.55 |
| mu1 | 2 | 100000 | hayır | 69.8 → 170 | 0.101 → 0.00724 | 0.0131 → 0.0217 | 254 → 11.8 |
| mu1 | 3 | 100000 | hayır | 70.8 → 193 | 0.0967 → 0.00747 | 0.0162 → 0.0209 | 246 → 14.6 |

Geç pencere (t ≥ 90000), koşul başına tohum ortalaması. Bölünme eşik altı tomurcukları da sayar
(track_propagules); kişi başı = bölünme / birey.

| koşul | birey | beden (hücre) | kapsama | R | yaş (adım) | kuşak | bölünme / 1000 adım | kişi başı |
|---|---|---|---|---|---|---|---|---|
| mu1 | 172 | 152 | 0.70 | 0.078 | 32123 | 13.9 | 10.7 | 0.062 |
| mu0 | 316 | 36 | 0.31 | 0.178 | 18507 | 13.1 | 92.8 | 0.294 |

Y3: t > 50000 için eğilimden arındırılmış otokorelasyonun ilk negatif minimumu (gecikme adım,
değer). |değer| < 2/√n ise salınım yok.

| koşul | tohum | birey: gecikme, ACF | avcılık: gecikme, ACF | 2/√n |
|---|---|---|---|---|
| mu0 | 0 | salınım yok (-0.01) | 2980, -0.13 | 0.028 |
| mu0 | 1 | salınım yok (-0.00) | salınım yok (-0.03) | 0.028 |
| mu0 | 2 | salınım yok (-0.00) | 7290, -0.08 | 0.028 |
| mu0 | 3 | 5730, -0.12 | 4680, -0.17 | 0.028 |
| mu1 | 0 | 18340, -0.54 | 6810, -0.17 | 0.028 |
| mu1 | 1 | 6010, -0.40 | salınım yok (-0.00) | 0.028 |
| mu1 | 2 | 20050, -0.45 | salınım yok (-0.00) | 0.028 |
| mu1 | 3 | 14670, -0.40 | 5750, -0.23 | 0.028 |

## Yorum (27 Kasım) — soy düzeltmeli tekrar, 2 koşul × 4 tohum, 100k adım

**Denetimler.** Tükenme 0/8. Ebeveynsiz doğum (t > 0) her koşuda ≤ %0,08 (life2 mu1_s0'da %85'ti).
Price analizinde ata zincirine bağlanamayan birey oranı 0.

**Önceki bulgular tekrarlandı.** Bu dört mu1 koşusu life2'den bağımsız rastgele akışlarla koştu.
Geç pencerede birey 172 (life2: 150), beden 152 hücre (183), kapsama 0,70 (0,74), R 0,078 (0,078).
Son ortalama genomun life2 ortak yönüyle kosinüsü 0,75–0,84. 32 bileşenin 18'i dört tohumda aynı
işaretli. Yani dört bağımsız koşu daha aynı yöne evrildi.

**Düzeltilmiş bölünme ve yaş.** Kişi başı bölünme mu1'de 0,062, mu0'da 0,294 / 1000 adım (4,7 kat az):
"seyrek bölünen" geçerli. Ortalama yaş 32100'e karşı 18500 adım (1,7 kat). life2'deki 5,7 kat izleme
hatasından gelen abartıydı. Kuşak derinliği iki koşulda benzer (~13–14). mu0 çok daha sık tomurcuk
veriyor ama tomurcukların daha azı 20 hücreye ulaşıyor (mu1 %46, mu0 %32 doğum).

**Seçilim düzeyi (Price, `analysis/figures/life_price_life2l.png`).** Ortak yön u üzerinde 100k adımda
toplam kayma: birey düzeyi seçilim 1,74–1,79, aktarım (yaşam içi değişim) 0,82–1,22. Seçilim payı
%60–68 ve dört tohumda da seçilim aktarımdan büyük: **birey düzeyi seçilim baskın.** Aktarım da pozitif
ve küçük değil (kaymanın ~1/3'ü). Mutasyon gürültüsü işaretçe simetrik olduğundan, yaşam içi değişimin
u yönünde birikmesi beden içi hücre soyları arasında da aynı yöne seçilim olduğunu gösteriyor. İki
düzey aynı yönde çalışıyor. Kaymanın ~%85'i ilk 50k adımda, dış kesmeler sürerken oluyor (seçilim·u
1,51 ± 0,10; sonrasında 0,25 ± 0,12).

**Y3 ekolojik dinamik (yukarıdaki tablo).** Düzenli bir av-avcı salınımı yok. mu0'da birey sayısında
anlamlı bir negatif lob yalnız bir tohumda var. mu1'de birey sayısının ACF'si -0,40…-0,54'e iniyor, ama
gecikmeler tohumdan tohuma 6000–20000 adım arasında dağınık ve 50k adımlık pencereye en çok 2–3 tur sığıyor.
Bu, periyodik salınımdan çok yavaş dalgalanmaya (ve 2/√n eşiğinin otokorelasyonlu seride gevşek olmasına)
uyuyor. Avcılık payındaki lob zayıf (|ACF| ≤ 0,23) ve tutarsız. Sonuç: iki koşul da gürültülü, kararlı bir
denge etrafında; mu1'deki daha yavaş dalgalanma büyük ve uzun ömürlü bedenlerin daha yavaş nüfus devriyle uyumlu.

**Y5 ortak bahçe (`analysis/figures/life_garden.png`, `experiments/results/y5_garden.csv`).** Her genom kendi
taze dünyasında: 128², aynı genomlu 6 kurucu, mutasyon ve kesme yok, 15k adım, ölçüm son 5k adım, 3 tekrar.
Birim genomdur (tekrarlar ortalanır); tükenen tekrarlar ölçüye girmez. Gruplar: ata, 4 evrim genomu, 8 rastgele.

| grup | genom | tükenen tekrar | beden (hücre) | hız | bölünme / birey / 1000 adım | R |
|---|---|---|---|---|---|---|
| ata | 1 | 0 | 79 | 0,050 | 1,7 | 0,60 |
| evrim | 4 | 0 | 1110 ± 720 | 0,0035 ± 0,0014 | 0,02 ± 0,03 | 0,048 ± 0,009 |
| rastgele | 7 (+1 tükendi) | %17 | 74 ± 34 | 0,126 ± 0,050 | 6,0 ± 6,8 | 0,80 ± 0,22 |

Evrim ile rastgele arasında kesin permütasyon testi. Rastgele genomlar kaynak evrim genomunun normuyla eşleşik
olduğundan etiketler yalnız kaynak içinde değiştirilir (54 atama, ulaşılabilecek en küçük p = 1/54 ≈ 0,019):
beden p = 0,019, R p = 0,019 (her kaynakta tam ayrışma), hız p = 0,037, bölünme p = 0,11. Dört ölçü için
Bonferroni eşiği (0,0125) bu tasarımda ulaşılamaz; sonucun gücünü 4 kaynak sınırlıyor. Havuzlanmış test
(eşleşmeyi yok sayar) beden ve hız için 0,006, R için 0,003 verir; düzeltme öncesi bu raporlanmıştı. Evrim genomları mutasyonsuz taze dünyada da büyük, yavaş ve
besini düşük düzeye indiren bedenler üretiyor: fenotip kalıtsal. Aynı normdaki rastgele genomlar ataya benzer
ya da daha hızlı ve küçük kalıyor, biri tükeniyor: etki genomun yönünden geliyor, büyüklüğünden değil.
Önceden yazılan beklenti tuttu. Bahçede evrim bedenleri evrim dünyasındakinden (152) çok daha büyük; olası
neden (doğrulanmadı): rakip ve kesme olmadan beden besin sınırına kadar büyüyor. Bir evrim genomu (tohum 2)
~110 hücrede kalıyor ama o da yavaş (0,005) ve besini düşük tutuyor. Tekrarlanabilirlik: gid'siz ilk koşunun
ata ve evrim satırları bu koşuyla birebir aynı.
