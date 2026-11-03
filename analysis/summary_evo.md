# Evrimsel katman — sonuç özeti

`analysis/plots_evo.py` tarafından üretilir.

## E1 – Çeşitlilik ve canlılık

| run | viable | area_in_range | single_body | stable_iou_200_1000 | pairwise_shape_iou_mean | pairwise_shape_iou_min | decoder_r2 | elongation_std | fill_std | hue_spread | target_iou_mean | target_iou_ge_0.9 | target_mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ancestor_s0 | 1.000 | 1.000 | 1.000 | 0.991 | 0.712 | 0.428 | nan | 0.692 | 0.023 | 0.102 | 0.967 | 0.984 | 0.000 |

## E4 – Parçadan yenilenme ve yara mutasyonu

| run | mu | regrown_single | shape_iou_to_parent | genome_drift | colour_change | V_peak |
|---|---|---|---|---|---|---|
| ancestor_s0 | 0.000 | 1.000 | 0.962 | 0.000 | 0.002 | 0.502 |
| ancestor_s0 | 0.150 | 1.000 | 0.963 | 0.038 | 0.002 | 0.518 |
| ancestor_s0 | 0.300 | 1.000 | 0.957 | 0.076 | 0.002 | 0.492 |

## E3 – Torun ile kendi atasının rekabeti (0,5 = fark yok)

| cond | evolved_cell_frac_mean | evolved_cell_frac_std | evolved_cell_frac_count | genome_distance_mean | genome_distance_std | genome_distance_count | mean_generation_mean | mean_generation_std | mean_generation_count |
|---|---|---|---|---|---|---|---|---|---|
| evrim | 0.495 | 0.242 | 12 | 1.539 | 0.208 | 12 | 8.650 | 0.157 | 12 |
| mutasyonsuz | 0.571 | 0.161 | 12 | 0.000 | 0.000 | 12 | 5.550 | 1.097 | 12 |

## E5 – Ata genomları arasında kalıtsal uygunluk farkı (H², tek yönlü ANOVA)

| metric | mean | std | H2 |
|---|---|---|---|
| n_org | 24.438 | 11.510 | 0.723 |
| cells | 4164.219 | 1425.527 | 0.366 |
| births | 73.240 | 21.119 | 0.282 |
| mean_area | 192.191 | 82.132 | 0.502 |

## E5 – Popülasyon uygunluğu zaman içinde (tek-soy testi, birey)

| cond | t | mean | std | count |
|---|---|---|---|---|
| evrim | 0 | 19.970 | 7.470 | 32 |
| evrim | 5000 | 20.160 | 6.620 | 32 |
| evrim | 10000 | 21.440 | 7.700 | 32 |
| evrim | 15000 | 23.690 | 8.600 | 32 |
| evrim | 20000 | 22.310 | 5.690 | 32 |
| evrim | 25000 | 22.340 | 7.100 | 32 |
| evrim | 30000 | 25.560 | 6.610 | 32 |
| mutasyonsuz | 0 | 19.720 | 6.750 | 32 |
| mutasyonsuz | 5000 | 22.250 | 8.700 | 32 |
| mutasyonsuz | 10000 | 22.970 | 8.350 | 32 |
| mutasyonsuz | 15000 | 24.160 | 5.890 | 32 |
| mutasyonsuz | 20000 | 24.560 | 6.180 | 32 |
| mutasyonsuz | 25000 | 26.880 | 7.520 | 32 |
| mutasyonsuz | 30000 | 27.880 | 4.720 | 32 |

## E5 – μ taraması: popülasyon uygunluğu (2 tohum ortalaması, birey)

| cond | 0 | 5000 | 10000 | 15000 | 20000 | 25000 | 30000 |
|---|---|---|---|---|---|---|---|
| evrim | 20.000 | 20.200 | 21.400 | 23.700 | 22.300 | 22.300 | 25.600 |
| evrim_mu0.05 | 21.300 | 18.500 | 23.500 | 22.500 | 23.000 | 23.200 | 20.300 |
| evrim_mu0.5 | 21.700 | 20.800 | 22.900 | 23.200 | 24.400 | 24.700 | 28.600 |
| evrim_mu1.0 | 20.100 | 24.300 | 25.900 | 29.200 | 27.000 | 28.300 | 28.000 |
| mutasyonsuz | 19.700 | 22.200 | 23.000 | 24.200 | 24.600 | 26.900 | 27.900 |

## E5 – Doğrulama: genom başına 6 taze tekrar (kurucular: 12'nin tamamı; koşullar: son anın en iyi 4'ü)

| cond | seed | max | mean | count |
|---|---|---|---|---|
| evrim | 0 | 27.000 | 23.920 | 4 |
| evrim | 1 | 28.330 | 25.580 | 4 |
| evrim_mu0.05 | 0 | 25.330 | 23.500 | 4 |
| evrim_mu0.05 | 1 | 28.170 | 17.040 | 4 |
| evrim_mu0.5 | 0 | 24.500 | 23.040 | 4 |
| evrim_mu0.5 | 1 | 30.670 | 27.960 | 4 |
| evrim_mu1.0 | 0 | 37.330 | 27.880 | 4 |
| evrim_mu1.0 | 1 | 29.500 | 23.420 | 4 |
| kurucu | 0 | 24.830 | 18.690 | 12 |
| kurucu | 1 | 25.000 | 19.100 | 12 |
| mutasyonsuz | 0 | 24.330 | 22.620 | 4 |
| mutasyonsuz | 1 | 28.170 | 25.380 | 4 |

## E5 – Kesin test: en iyi evrimleşmiş genomlar ve en iyi kurucular, 24 taze tekrar

| cond | seed | genome | mean | std | count | se |
|---|---|---|---|---|---|---|
| evrim | 1 | 2 | 27.250 | 9.510 | 24 | 1.940 |
| evrim_mu0.5 | 1 | 4 | 28.120 | 9.460 | 24 | 1.930 |
| evrim_mu0.5 | 1 | 6 | 28.000 | 10.870 | 24 | 2.220 |
| evrim_mu1.0 | 0 | 0 | 31.290 | 10.850 | 24 | 2.210 |
| evrim_mu1.0 | 1 | 6 | 28.580 | 8.260 | 24 | 1.690 |
| kurucu | 0 | 0 | 25.790 | 11.520 | 24 | 2.350 |
| kurucu | 0 | 2 | 25.960 | 9.550 | 24 | 1.950 |
| kurucu | 0 | 4 | 21.040 | 6.220 | 24 | 1.270 |
| kurucu | 1 | 0 | 25.000 | 9.260 | 24 | 1.890 |
| kurucu | 1 | 5 | 28.120 | 9.430 | 24 | 1.930 |
| kurucu | 1 | 8 | 25.250 | 8.850 | 24 | 1.810 |
