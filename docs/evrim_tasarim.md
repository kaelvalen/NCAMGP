# Evrimsel NCA Ekosistemi — Tasarım (yeni ana katkı)

Karar (16 Ekim 2026): hedefli NCA sonuçları (büyüme, onarım, parça eğrisi, prob/müdahale)
zorunlu **Çekirdek** katman ve taban çizgisi olarak kalır. Ana katkı, hedef görüntüsü olmayan,
açık uçlu bir evrimsel ekosistemdir. İyileşme ölçütü: **ekolojik hayatta kalma**.

## Fikir

Biçimler için bir kural yoktur. Tüm hücreler aynı yerel kuralı ("fizik") paylaşır; her hücre
kalıtsal bir **genom** taşır ve kural bu genomu okuyarak büyür. Genom uzayı sürekli olduğu için
canlı biçimleri sonsuz kombinasyondadır. Bir canlı parçalandığında yara kenarında bir
**potansiyel** (biyoelektrik yara sinyali) oluşur; potansiyelin yüksek olduğu yerde genom mutasyona
uğrar. Böylece yenilenen canlı eskisinin aynısı olmaz. Parçalanma üremedir: parçalar ayrı
bireyler olarak yenilenir, sınırlı kaynak için rekabet eder; daha uzun yaşayan ve daha çok
çoğalan soylar yayılır.

## Bileşenler

| Bileşen | Öğrenilen mi? | Tanım |
|---|---|---|
| Hücre durumu | — | RGBA (4) + potansiyel V (1) + gizli (12) |
| Genom g | — | Hücre başına 8 boyut; kuralın girdisi, kural değiştiremez |
| Fizik kuralı f_θ | evet | Algılama [RGBA, V, gizli] + hücrenin genomu → durum güncellemesi |
| Kalıtım | hayır | Yeni canlanan hücre, canlı komşularının genom ortalamasını alır; canlı hücreler komşularına doğru yavaşça homojenleşir; ölü hücrenin genomu silinir |
| Yara potansiyeli V | hayır | Komşu hücre ölümüyle artar, söner ve canlı doku içinde yayılır; kural V'yi algılar |
| Mutasyon | hayır (yalnız ekosistemde) | g ← g + μ · V · ξ, ξ uzayca düzgünleştirilmiş gürültü |
| Enerji / kaynak | hayır (yalnız ekosistemde) | Kaynak alanı R yavaşça yenilenir; canlı hücre R tüketir, bakım maliyeti öder; enerjisi biten hücre ölür (bu da yara ve potansiyel yaratır) |
| Parçalanma | hayır | Rastgele kesme olayları (yırtıcı); parçalar uzaklaştırılır, yer yoksa ezilir |

## Ata planarya + evrim (16 Ekim 2026 kararı)

Saf hedefsiz eğitim (aşağıda) teknik olarak çalıştı — rastgele genomların %91'i yaşayabilir,
tek bedenli, kararlı canlılar üretti ve genom fenotipten R² 0,79 ile okunabildi — ama model
çeşitliliği hep renge yazdı, şekil neredeyse sabit kaldı (düz neon kapsüller; altı kayıp
tasarımı denendi, `runs/_evo_v*`). Bu yüzden fizik iki aşamalıdır:

1. **Ata eğitimi** (`nca/train_ancestor.py`): genom koşullu kural, ata canlı uzayındaki
   (`targets/creature_space.py`) her canlıyı tek hücreden büyütmeyi ve parçadan yenilemeyi
   öğrenir. Uzay organik planaryalardan oluşur ve genomla sürekli değişir: boy, en, kulakçık,
   baş ve göz, doğal pigment rengi, benek, desen (düz/çizgili/bantlı), kuyruk çatalı ve farinks.
   Bu yalnız *atanın* vücut planlarıdır, tek bir hedef değildir.
2. **Ekosistem**: kural sabittir; biçimleri genomlar belirler. Yara potansiyeline bağlı
   mutasyon genomları ata uzayının dışına taşıyabilir; kuralın bu genomlardaki davranışı
   eğitilmemiştir — yeni biçimler buradan çıkar, seçilim yaşayabilir olanları tutar.

Biyolojik anlatı: tüm canlılar ortak bir atanın vücut planını paylaşır; evrim bu planın
üzerinden ilerler.

## Fiziğin hedefsiz eğitimi (ilk yaklaşım, karşılaştırma)

Fizik, "nasıl bir canlı" ile değil "canlı olmak" ölçütleriyle eğitilir. Her örnek rastgele bir
genomla (g ~ N(0, I)) tek hücreden başlar.

- **Kimlik (çeşitlilik):** küçük bir çözücü ağ, büyümüş canlının görüntüsünden genomu tahmin
  eder; NCA ile çözücü birlikte bu hatayı küçültür. Fenotip genomu taşımak zorundadır, yani
  farklı genom farklı canlı demektir (morfolojik karşılıklı bilgi; DIAYN benzeri). İkinci bir
  çözücü yalnız alfa maskesine bakar ve şekil çeşitliliğini zorlar.
- **Canlılık:** canlı alan [A_min, A_max] içinde kalır, tek beden olur (en büyük bileşen dışındaki
  kütle cezalandırılır), uzun açılımda durağanlaşır (homeostaz), değerler sınırlı kalır.
- **Yapı:** beden içi renk çeşitliliği alt sınırı (düz renkli leke olmasın).
- **Yenilenme:** sample pool içinde dairesel hasar ve yarım düzlem kesmesi; parça yine canlılık
  ve kimlik ölçütlerini karşılamalıdır. Aynı biçime dönmesi gerekmez, yalnızca kendi genomunun
  canlısına dönmesi gerekir.

## Hipotezler

- **E1 (çeşitlilik):** rastgele genomlar ayırt edilebilir, kararlı ve tek bedenli canlılar üretir;
  morfolojik tanımlayıcıların genom uzayındaki dağılımı geniştir.
- **E2 (yara potansiyeli):** kesmeden sonra V yara kenarında yükselir ve yenilenme bittiğinde
  söner; V algısı kapatıldığında (ablasyon) yenilenme kötüleşir.
- **E3 (evrimsel iyileşme):** ekosistemde nesiller boyunca popülasyonun ortalama ömrü ve parça
  boyutu–yenilenme başarısı artar (mutasyonsuz kontrol ve rastgele genom kontrolüne göre).
- **E4 (yenilenme yeniliği):** parçadan yenilenen birey ebeveyninden ölçülebilir biçimde farklıdır
  (fenotip mesafesi > 0) ve fark potansiyelin büyüklüğüyle artar.

## İlk sonuçlar (16 Ekim 2026, ata modeli `runs/ancestor_s0`, 30k iterasyon)

- **E1:** eğitimde görülmemiş 64 genomun %100'ü yaşayabilir, tek bedenli, 1000 adım kararlı
  (IoU 0,99); kendi ata hedefine ortalama IoU 0,967 (%98'i ≥ 0,9). Şekil çeşitliliği hedefsiz
  modelden belirgin biçimde yüksek (fenotipler arası IoU 0,71'e karşı 0,84; uzama std 0,69'a karşı 0,19).
- **E4:** yarısı kesilen canlıların %100'ü tek beden olarak yenilenir (ebeveyne IoU 0,96); tek
  bir yara genomu 0,04 (μ = 0,15) – 0,08 (μ = 0,3) kaydırır.
- **E3 (tutmadı):** 30k adım, 2 tohum; torunlar kendi atalarıyla aynı dünyada yarıştırıldığında
  hücre payı evrimde 0,50 ± 0,24, mutasyonsuz kontrolde 0,57 ± 0,16 (n = 12). Genomlar 8–9
  nesilde 1,54 uzaklaştı ama rekabette fark çıkmadı. Rekabet ölçüsü çok gürültülü (yerleşim
  piyangosu) ve hücre payını ölçüyor. Zaman serileri: mutasyon genom çeşitliliğini korur (2,2'ye
  karşı 1,0) ve dünya ~%20 daha fazla, ~%25 daha küçük birey taşır.
- **E5 (tek-soy uygunluk testi, 24 ata genomu x 4 tekrar, 4000 adım):** kalıtsal uygunluk farkı
  *vardır* — birey sayısı için H² = 0,72 (hücre sayısı 0,37). Uygunluk beden eniyle (g1) güçlü
  ters ilişkili (r = −0,90), ortalama beden alanıyla r = −0,81: ince/küçük bedenli soylar aynı
  kaynakla daha çok birey üretir. (E3 sonrası "kalıtsal fark yok" yorumu bu ölçümle yanlışlandı.)
  Seçilim yönü belli olduğu için asıl test, popülasyon uygunluğunun zaman içindeki eğrisidir
  (`e5_fitness.py curve`).
- **E5 uygunluk eğrisi (μ = 0,15, 2 tohum, zaman noktası başına 8 genom x 2 tekrar):** popülasyon
  uygunluğu 30k adımda her iki koşulda ~20'den 26–28 bireye çıkar (seçilim işliyor). Mutasyonsuz
  kontrol daha hızlı yükselir (27,9'a karşı 25,6) ve hiçbir koşul en iyi kurucuyu (28,6) aşmaz:
  iyileşme şimdilik mevcut çeşitlilik içinden seçilimden gelir, μ = 0,15'teki mutasyon net bir yük
  getirir. `analysis/figures/e5_fitness_curve.png`.
- **μ taraması (0 / 0,05 / 0,15 / 0,5 / 1,0; 2 tohum x 30k adım; `experiments/mu_sweep.sh`):**
  popülasyon uygunluğu 15k adımda sırasıyla 24,2 / 22,5 / 23,7 / 23,2 / **29,2**, 30k adımda
  27,9 / 20,3 / 25,6 / 28,6 / 28,0. μ = 1,0 uyumu yaklaşık iki kat hızlandırıyor; μ = 0,05 en
  kötüsü (mutasyon yükü var, yararlı varyant üretmeye yetmiyor). Ortalamada hiçbir koşul en iyi
  kurucuyu aşmıyor. `analysis/figures/e5_fitness_curve_mu.png`.
- **Doğrulama (`e5_fitness.py verify`, 12 kurucunun tamamı + her koşulun son anındaki en iyi 4
  genomu, 6 taze tekrar):** gerçek en iyi kurucu 24,8 / 25,0 (tohum 0 / 1). Eğrideki 29,5'lik
  çizgi 8 kurucudan ve 2 tekrardan gelen şişkin bir tahmindi. Test GPU'da deterministik değil:
  aynı genom aynı tohumlarla 23,5 ve 28,2 verdi, yani ~5 bireylik fark gürültüdür. Mutasyonsuz
  koşulun son genomları kuruculardan birinin aynısı (uzaklık 0); evrimleşmiş olanlar en yakın
  kurucudan 3,5–4 uzakta, yani gerçekten yeni genomlar. En yüksek değer μ = 1,0'da: 37,3.
- **Kesin test (`e5_fitness.py confirm`, en iyi 5 evrimleşmiş genom + tohum başına en iyi 3
  kurucu, 24 taze tekrar):** 37,3'lük genom 31,3 ± 2,2'ye indi (kazananın laneti). Bu, kendi
  dünyasının en iyi kurucusundan (26,0 ± 1,9) %20 yüksek ama anlamlı değil (Welch p = 0,08,
  çoklu karşılaştırma düzeltmesinden önce). Tüm kurucuların en iyisine (28,1 ± 1,9) karşı p = 0,29.
  Diğer evrimleşmiş genomlar 27–29 bandında, yani kurucu düzeyinde. **Sonuç:** seçilim işliyor ve
  yüksek μ uyumu hızlandırıyor; yara mutasyonu kurucu düzeyinde *yeni* genomlar üretiyor, ama 30k
  adımda en iyi kurucunun ötesine geçen bir yenilik istatistiksel olarak gösterilemedi.
- **Sonraki adım (açık):** μ = 1,0 ile daha uzun (≥ 100k adım) ve daha çok tohumlu koşular; ya da
  en iyi kurucu dışarıda bırakılarak "kaybedilen uygunluğun mutasyonla geri kazanımı" testi.

## Metrikler

Birey sayısı, ömür, soy ağacı (parça-ebeveyn bağı), genom ve fenotip çeşitliliği, nesil başına
parça eğrisi (çekirdek katmandaki D2 metriği, evrimleşmiş genomlarla), V dinamikleri.

## Kod

- `nca/evo.py` — genom koşullu kural, kalıtım, potansiyel, mutasyon, enerji/kaynak fiziği
- `nca/train_evo.py` — fiziğin hedefsiz eğitimi
- `nca/world.py` — ekosistem simülasyonu ve kayıt
- `experiments/e*.py` — E1–E4
