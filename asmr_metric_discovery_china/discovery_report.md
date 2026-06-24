# ASMR metric discovery report

- Cases used: 100
- Input root: `E:\simuChina\cartesianBDD_original`
- Pattern: `**/pressureMotion.txt`
- Warmup samples skipped: 4
- Static library degree: 1

This report ranks pressure-derived scalar descriptors and fits sparse
static expressions. It is not a standard dynamical SINDy fit because
the target is a scenario parameter, not a time derivative.

## Target `log10_Re`

Sparse expression: `log10_Re ~= 4.6859938 + 0.50040873*log10_abs(dominant_frequency_min) + 0.49927188*log10_abs(dominant_frequency_q25)`

- In-sample R2: 1
- In-sample MAE: 0.0004159
- Cross-validated R2: 1 +/- 1.976e-06

Best single descriptors:

| feature | Spearman | Pearson | MI | single-feature CV R2 |
|---|---:|---:|---:|---:|
| dominant_frequency_min | 0.995 | 0.9516 | 1.834 | 0.8855 |
| dominant_frequency_q25 | 0.995 | 0.9516 | 1.834 | 0.8855 |
| log10_t_max_min | -0.995 | -0.9996 | 1.696 | 0.999 |
| china_wall_log10_t_min_corrected_max | -0.9864 | -0.9941 | 1.592 | 0.9812 |
| log10_t_min_max | -0.9864 | -0.9941 | 1.592 | 0.9812 |
| china_wall_log10_t_min_exact_max | -0.9864 | -0.994 | 1.592 | 0.9808 |
| china_wall_log10_t_min_corrected_q75 | -0.9275 | -0.9476 | 1.507 | 0.8352 |
| log10_t_min_q75 | -0.9275 | -0.9476 | 1.507 | 0.8352 |
| log10_t_max_max | -0.9851 | -0.9921 | 1.483 | 0.9802 |
| china_wall_log10_t_min_exact_q75 | -0.9259 | -0.9452 | 1.442 | 0.8281 |
| log10_t_max_median | -0.9765 | -0.9883 | 1.386 | 0.9713 |
| china_wall_log10_t_min_exact_mean | -0.8158 | -0.7077 | 1.381 | 0.2316 |

## Target `log10_h`

Sparse expression: `log10_h ~= 1.7153265 - 11.451613*signed_log10(pressure_mean_abs_median) - 17.714343*signed_log10(pressure_mean_abs_mean) + 20.504115*signed_log10(pressure_mean_abs_q25) - 2.479015*signed_log10(svd_energy_fraction_5) - 70.173585*signed_log10(spatial_gradient_median)`

- In-sample R2: 0.9972
- In-sample MAE: 0.009529
- Cross-validated R2: 0.9965 +/- 0.0013

Best single descriptors:

| feature | Spearman | Pearson | MI | single-feature CV R2 |
|---|---:|---:|---:|---:|
| spatial_std_signal_rms | -0.995 | -0.9938 | 2.025 | 0.9853 |
| spatial_std_signal_mean | -0.995 | -0.9893 | 2.025 | 0.9757 |
| pressure_mean_abs_q25 | -0.4523 | -0.3311 | 2.025 | 0.05942 |
| log10_pressure_absmax_q75 | -0.995 | -0.9942 | 2.025 | 0.987 |
| highpass_010_log10_inv_range4_median | 0.995 | 0.9718 | 1.992 | 0.9383 |
| highpass_030_log10_inv_range4_q75 | 0.9552 | 0.903 | 1.93 | 0.8121 |
| pressure_mean_abs_mean | -0.9806 | -0.9545 | 1.922 | 0.9088 |
| log10_pressure_range_q75 | -0.995 | -0.9966 | 1.91 | 0.9925 |
| china_wall_raw_log10_inv_range4_q25 | 0.995 | 0.9966 | 1.91 | 0.9925 |
| china_wall_highpass_log10_inv_range4_q25 | 0.995 | 0.9966 | 1.91 | 0.9925 |
| log10_inv_pressure_range4_q25 | 0.995 | 0.9966 | 1.91 | 0.9925 |
| spatial_range_signal_log10_range | -0.995 | -0.9918 | 1.91 | 0.9812 |

## Classic relationships for `Re`

These fits compare identity/log/square-root/inverse/signed-log
transforms and optional quadratic terms. R2 is measured in the
transformed target space shown in the expression.

| expression | CV R2 | in-sample R2 | MAE |
|---|---:|---:|---:|
| `log10(Re) ~= 4.6859938 + 0.99968061*log10(dominant_frequency_min)` | 1 +/- 1.98e-06 | 1 | 0.0004159 |
| `log10(Re) ~= 4.6859938 + 0.99968061*log10_abs(dominant_frequency_min)` | 1 +/- 1.98e-06 | 1 | 0.0004159 |
| `log10(Re) ~= 4.6859938 + 0.99968061*log10(dominant_frequency_q25)` | 1 +/- 1.98e-06 | 1 | 0.0004159 |
| `log10(Re) ~= 4.6859938 + 0.99968061*log10_abs(dominant_frequency_q25)` | 1 +/- 1.98e-06 | 1 | 0.0004159 |
| `log10(Re) ~= 4.68604 + 0.99976713*log10(dominant_frequency_min) + 3.719587e-05*(log10(dominant_frequency_min))^2` | 1 +/- 2.05e-06 | 1 | 0.0004162 |
| `log10(Re) ~= 4.68604 + 0.99976713*log10_abs(dominant_frequency_min) + 3.719587e-05*(log10_abs(dominant_frequency_min))^2` | 1 +/- 2.05e-06 | 1 | 0.0004162 |
| `log10(Re) ~= 4.68604 + 0.99976713*log10(dominant_frequency_q25) + 3.719587e-05*(log10(dominant_frequency_q25))^2` | 1 +/- 2.05e-06 | 1 | 0.0004162 |
| `log10(Re) ~= 4.68604 + 0.99976713*log10_abs(dominant_frequency_q25) + 3.719587e-05*(log10_abs(dominant_frequency_q25))^2` | 1 +/- 2.05e-06 | 1 | 0.0004162 |
| `cuberoot(Re) ~= 0.0049874151 + 36.473148*cuberoot(dominant_frequency_min)` | 1 +/- 2e-06 | 1 | 0.005305 |
| `cuberoot(Re) ~= 0.0049874151 + 36.473148*cuberoot(dominant_frequency_q25)` | 1 +/- 2e-06 | 1 | 0.005305 |
| `cuberoot(Re) ~= -0.0024352026 + 36.508921*cuberoot(dominant_frequency_min) - 0.04081254*(cuberoot(dominant_frequency_min))^2` | 1 +/- 2e-06 | 1 | 0.005278 |
| `cuberoot(Re) ~= -0.0024352026 + 36.508921*cuberoot(dominant_frequency_q25) - 0.04081254*(cuberoot(dominant_frequency_q25))^2` | 1 +/- 2e-06 | 1 | 0.005278 |

## Classic relationships for `h`

These fits compare identity/log/square-root/inverse/signed-log
transforms and optional quadratic terms. R2 is measured in the
transformed target space shown in the expression.

| expression | CV R2 | in-sample R2 | MAE |
|---|---:|---:|---:|
| `log10(h) ~= 1.7056262 - 14.218911*signed_log10(pressure_max_q75) + 54.603231*(signed_log10(pressure_max_q75))^2` | 0.9981 +/- 0.000458 | 0.9983 | 0.007976 |
| `sqrt(h) ~= 9.3510254 - 22.83326*sqrt(pressure_max_q75) + 18.855256*(sqrt(pressure_max_q75))^2` | 0.9975 +/- 0.000836 | 0.9978 | 0.03382 |
| `sqrt(h) ~= 9.3510254 - 22.83326*sqrt_abs(pressure_max_q75) + 18.855256*(sqrt_abs(pressure_max_q75))^2` | 0.9975 +/- 0.000836 | 0.9978 | 0.03382 |
| `cuberoot(h) ~= 4.6452322 - 8.9189926*sqrt(pressure_max_q75) + 6.9622443*(sqrt(pressure_max_q75))^2` | 0.9973 +/- 0.000918 | 0.9976 | 0.01461 |
| `cuberoot(h) ~= 4.6452322 - 8.9189926*sqrt_abs(pressure_max_q75) + 6.9622443*(sqrt_abs(pressure_max_q75))^2` | 0.9973 +/- 0.000918 | 0.9976 | 0.01461 |
| `log10(h) ~= 1.6807073 - 5.502499*pressure_max_q75 + 8.4097157*(pressure_max_q75)^2` | 0.9973 +/- 0.000473 | 0.9976 | 0.009323 |
| `1/(h) ~= 0.3220005 + 0.3876757*log10(pressure_max_q75) + 0.12928407*(log10(pressure_max_q75))^2` | 0.9972 +/- 0.00103 | 0.9976 | 0.00156 |
| `1/(h) ~= 0.3220005 + 0.3876757*log10_abs(pressure_max_q75) + 0.12928407*(log10_abs(pressure_max_q75))^2` | 0.9972 +/- 0.00103 | 0.9976 | 0.00156 |
| `sqrt(h) ~= 2.43309 - 0.78731979*sqrt_abs(log10_pressure_range_q75) + 4.9347361*(sqrt_abs(log10_pressure_range_q75))^2` | 0.997 +/- 0.000806 | 0.9974 | 0.04249 |
| `sqrt(h) ~= 2.43309 - 0.3936599*sqrt_abs(china_wall_raw_log10_inv_range4_q25) + 1.233684*(sqrt_abs(china_wall_raw_log10_inv_range4_q25))^2` | 0.997 +/- 0.000806 | 0.9974 | 0.04249 |
| `sqrt(h) ~= 2.43309 - 0.3936599*sqrt_abs(china_wall_highpass_log10_inv_range4_q25) + 1.233684*(sqrt_abs(china_wall_highpass_log10_inv_range4_q25))^2` | 0.997 +/- 0.000806 | 0.9974 | 0.04249 |
| `sqrt(h) ~= 2.43309 - 0.3936599*sqrt_abs(log10_inv_pressure_range4_q25) + 1.233684*(sqrt_abs(log10_inv_pressure_range4_q25))^2` | 0.997 +/- 0.000806 | 0.9974 | 0.04249 |
