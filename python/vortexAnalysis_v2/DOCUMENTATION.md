# VortexAnalysis_v2 Documentation (Excluding Kinematics Regression)

**Scope**
This document covers the code and generated artifacts in `python/vortexAnalysis_v2`, excluding `train_regression_kinematics.py` and `vortex_outputs/kinematics_regression/` per request.

**Overview**
`vortexAnalysis_v2` is a pipeline for vortex detection and tracking from Q-criterion maps, building pressure snapshots aligned to vortex visibility at sensors, computing pressure-mode bases (SVD), training a linear classifier, and testing it against real pressure signals. It also includes spine/kinematic preprocessing (non-regression side).

**Data Conventions And Assumptions**
- Grid size defaults to `256 x 128` (`num_cols x num_rows`). Flattened frames are stored in i-major order and reshaped with `.reshape(num_cols, num_rows).T`.
- Dataset names are consistently `nadia`, `thomas`, `boai`, `full`.
- Default sensor coordinates are `146,80`, `198,80`, `200,39` with labels `S2`, `S1`, and `S4` respectively.
- Many scripts assume dataset roots on `D:/` (for example `D:/crop_nadia`, `D:/thomas_files`, `D:/boai_files`, `D:/full_files`, or `D:/crop_{dataset}`).
- Pressure snapshots are contiguous frame segments where a vortex is visible at a sensor; these are resampled, zero-meaned, and max-abs normalized before classification.

**Scripts (Purpose, Inputs, Outputs)**
| Script | Purpose | Key Inputs | Key Outputs |
| --- | --- | --- | --- |
| `Q_criterion_computation.py` | Compute Q-criterion from velocity maps across experiments/snapshots. | `ux_map.csv`, `uy_map.csv` (and circle variants) in dataset folders. | Writes `Q_criterion_map.csv` and `circle_Q_criterion_map.csv` into each snapshot folder. |
| `view_Q_criterion.py` | Visual compare reference Q vs recomputed Q, frame-by-frame with overlap metrics. | `Q-criterion.csv`, `Q_criterion_test_map.csv`. | Interactive display (no files). |
| `vortex_extraction.py` | Detect vortices from Q maps, apply body/sensor/wall filters, track over time, optional animation. | Q maps, body contours (`x.csv`, `y.csv` or snapshot-specific), sensor list. | `vortex_tracks.csv` or `circle_vortex_tracks.csv` per snapshot; optional `vortex_tracking.gif`. |
| `build_pressure_from_vortex.py` | Build sensor pressure snapshots where vortices are visible, using vortex tracks. | `vortex_tracks.csv` + `pressure_map.csv` per snapshot. | `vortex_outputs/pressure_snapshots/pressure_snapshots_{dataset}_{model}.csv` and `_meta.csv`. |
| `compute_pressure_modes.py` | Resample/normalize snapshots and compute SVD pressure modes at an energy target. | Pressure snapshot matrices + meta CSVs. | `vortex_outputs/pressure_modes/pressure_modes_{dataset}_{energy}.npz`. |
| `train_classifier_pressure.py` | Train linear classifier on mode coefficients (Optuna or fixed params). | Modes + pressure snapshots. | Weights `classifier_best_weights_{dataset}_{energy}.pt` and results CSVs. |
| `test_classifier_real_pressure.py` | Apply trained classifiers to real pressure signals, generate prediction CSVs and plots. | TDMS pressure signals, timestamps CSVs, snapshot meta, mode files, weights. | `real_pressure_predictions_{arch}_{dataset}_{energy}.csv` and optional distribution/threshold plots. |
| `run_classifier_pipeline.py` | Orchestrate compute-modes → train → test across energies/thresholds. | Same as above scripts. | Runs the pipeline; generates outputs from sub-scripts. |
| `train_classifier_vortex.py` | Match vortices to confident real-pressure predictions, build vortex-feature dataset, optionally train a vortex classifier with SHAP. | Prediction CSVs, snapshot meta, vortex tracks. | `matching_vortices_*.csv`, vortex classifier weights/results, SHAP and loss plots. |
| `preprocess_spines.py` | Extract COM position, orientation, tail angle, speed, acceleration from `spines_interpolated.csv`. | `spines_interpolated.csv` per experiment/sub-experiment. | Per-file `spine_*.csv` or aggregated matrices + `spine_meta.csv`. |

**Pipeline Flow (Typical)**
1. Compute Q-criterion maps with `Q_criterion_computation.py`.
2. Extract and track vortices with `vortex_extraction.py` to produce `vortex_tracks.csv` in each snapshot folder.
3. Build pressure snapshots tied to vortex visibility with `build_pressure_from_vortex.py` into `vortex_outputs/pressure_snapshots/`.
4. Compute pressure modes via `compute_pressure_modes.py` into `vortex_outputs/pressure_modes/`.
5. Train the linear pressure classifier using `train_classifier_pressure.py`.
6. Test on real pressure via `test_classifier_real_pressure.py` and generate prediction CSVs and plots.
7. Match vortices to confident predictions and train vortex-feature classifiers via `train_classifier_vortex.py`.

**Current Outputs Observed (Non-Kinematics)**
- `vortex_outputs/pressure_snapshots/pressure_snapshots_{dataset}_{model}.csv` and `_meta.csv` exist for `boai`, `full`, `nadia`, `thomas` and models `fish`, `cylinder`.
- `vortex_outputs/pressure_modes/pressure_modes_{dataset}_{energy}.npz` exist for datasets above with energy tags `95` and `99`.
- `vortex_outputs/pressure_modes/classifier_best_weights_{dataset}_{energy}.pt` and `classifier_results_{dataset}_{energy}.csv` exist for energy tags `95` and `99`.
- `vortex_outputs/pressure_modes/real_pressure_predictions_linear_{dataset}_{energy}.csv` exist for energy tags `95` and `99`.
- `vortex_outputs/pressure_modes/real_pressure_threshold_curves_{energy}.png/.pdf/.eps` exist for energy tags `95` and `99`.
- `vortex_outputs/matching_vortices/matching_vortices_linear_{dataset}_{energy}_thr{threshold}.csv` exist for `energy=95, threshold=0.95` and `energy=99, threshold=0.75`.
- `vortex_outputs/vortex_classifier/` contains `vortex_classifier_best_weights_{dataset}_95_thr0.95.pt`, `vortex_classifier_results_summary_95_thr0.95.csv`, SHAP plots (`vortex_shap_*`), and loss-curve plots (`vortex_loss_curves.*`).
- `vortex_outputs/vortex_tracks.csv` exists from a single-run extraction.
- `vortex_outputs/pressure_modes/README.md` is a legacy summary and may still reference architectures that are no longer in this folder.

**Dependencies (By Script Usage)**
- Core: `numpy`, `pandas`, `matplotlib`.
- Vortex detection and masking: `scipy` (filters, interpolation, morphology, KD-tree).
- Training/inference: `torch`, `optuna`.
- Real pressure I/O: `nptdms` for TDMS.
- SHAP plots for vortex classifier: `shap`.

**Excluded By Request**
- `python/vortexAnalysis_v2/train_regression_kinematics.py`.
- `python/vortexAnalysis_v2/vortex_outputs/kinematics_regression/`.
