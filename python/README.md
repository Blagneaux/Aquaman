# Python Folder Guide

This folder collects small, mostly single-purpose scripts for the Aquaman project: data acquisition from sensors, Lilypad/CFD post-processing, YOLO dataset prep, vortex analysis, and assorted visualization utilities. Many scripts are designed to be run directly and use hard-coded Windows paths (e.g., `D:/crop_nadia/...`, `E:/simuChina/...`); update those before running.

## Dependencies (commonly used)
- Core: `numpy`, `pandas`, `matplotlib`, `scipy`, `opencv-python`
- Data files: `nptdms`, `openpyxl`, `serial`, `nidaqmx`
- ML/metrics: `ultralytics` (YOLOv8), `scikit-learn`, `dtw`, `similaritymeasures`, `shapely`, `shap`
- Other: `pydmd`, `pysindy`, `tkinter`, `Pillow`

## Subfolders
- `circleDataset/`: YOLO dataset skeleton (images/masks split into train/test with `dataset.yaml`).
- `python_extras/`: UIs and utilities (live pressure display, band-pass filtering helper, panorama stitching).

## Data acquisition & sensor utilities
- `add_sensor2video.py`: Overlay sensor position on a video and prep filtered pressure traces from TDMS/CSV for comparison.
- `check_Chine_exp.py`: Low-pass filter and plot wall-pressure Excel dumps.
- `find_crop_start.py`: Locate where a cropped clip begins inside a full-length video by frame differencing.
- `forceSensor_connect.py`: Read NI force/pressure channels, apply transfer matrix, and log forces/pressures to CSV.
- `readPressure.py`: Arduino wall-pressure reader that logs to Excel for a given Re/h combo.
- `readPressureCorentin.py`: Tkinter GUI to start/stop serial logging, save to Excel, and plot results.
- `readPressureFile.py`: Interactive low-pass filter and plot an existing Excel pressure file.
- `readSensors.py`: Run wall-pressure and force-sensor readers in parallel threads.
- `tdmsRead.py`: Inspect TDMS files against digital-twin pressures with filtering and Frechet distance checks.

## Simulation, correlation, and digital-twin tooling
- `ASMR.py`: Core “Automatic Scenarization of Multi-fidelity Representation” loop—extract metrics from Lilypad outputs, read sensors, and queue next Re/h parameters (uses NIDAQmx + GPR).
- `ASMR_rassurerGurvan.py`: Toy GPR disagreement example to illustrate the ASMR sampling idea.
- `auto_simu.py`: Launch multiple Processing/Lilypad sketches in parallel from a CSV parameter list.
- `similarity_computation.py`: Compute DTW, Frechet distance, and correlation between TDMS sensor signals and Lilypad simulations; writes per-sample CSVs.
- `similarity_mean_computation.py`: Aggregate similarity metrics across runs (mid/circle/offset datasets) and plot cumulative means.
- `compute_mean_similarity_test.py`: Minimal DTW/Frechet demo on synthetic sine/cosine signals.
- `cross_correlation.py`: Align experimental TDMS data with simulated pressure maps (windowing, filtering) and compute correlations per sample.
- `auto_cross_correlation.py`: Cross-correlate every experimental TDMS run against every other run.
- `auto_cross_correlation_cylinder.py`: Cross-correlate cylinder simulations with each other; can animate flow reconstruction.
- `analysis_correlation_matrix.py`: KDE/peak analysis of saved cross-correlation matrices.
- `chinaPressureExtraction.py`: Parse Lilypad `pressureMotion.txt`/`FullMap.csv` across Re/h folders, filter Cp, and plot/regress cylinder/wall pressure trends.
- `extract_pressure_gurvan.py`: Build per-case Cp/pressure metrics vs distance/time for the Gurvan simulation set.
- `compareMaps.py`: Turn reference and YOLO-derived pressure maps into grayscale videos and difference visuals.
- `quick_visualization.py`: Quick overlay of filtered experimental pressure vs digital-twin pressures at sensor locations (with nearby offsets).

## YOLO/dataset preparation
- `dataAugmentation.py`: Crop/rescale fish images around masks or YOLO detections to build training tiles.
- `dataAugmentationLabel.py`: Map cropped YOLO labels back to the original image via template matching/homography.
- `mask2Yolo.py`: Convert binary masks to YOLO polygon labels (filters tiny contours).
- `labelStudioBrush2Yolo.py`: Convert Label Studio brush exports into YOLO polygon text files.
- `lilypad2Yolo.py`: Convert Lilypad pressure grids into YOLO-style synthetic labels.
- `seedYolo.py`: Split `all/images` + `all/labels` into train/test folders (80/20).
- `checkLabels.py`: Visualize YOLO polygon labels over images and compute shape/tail heuristics.
- `reformatYoloOutput.py`: Plot raw YOLO label text outputs.
- `quickYOLOtest.py`: Run YOLOv8 segmentation on video, spline-interpolate contours, and draw rescaled polylines.
- `ultraQuickYOLOtest.py`: One-liner YOLOv8 video inference with saving/display enabled.
- `labelsIntersection.py`: IoU/area comparison between manual and OpenCV-generated labels (Shapely).

## Video & segmentation helpers
- `ezSegmentation.py`: Background-median subtraction to produce masks/images from a video.
- `contourDetection.py`: Simple contour detection demo that saves annotated frames.
- `crop_video.py`: Trim frames from the start/end of a video and save a cropped copy.
- `find_crop_start.py`: (See above) locate crop offset; useful when syncing clips.
- `lookForVortex.py`: Deprecated helper to render pressure-map videos with YOLO contours and sensor markers.

## Vortex analysis
- `vortexDetection.py`: Detect/track vortices in vorticity or pressure fields, filter near sensors, and animate overlays; includes labeling UI hooks.
- `vortexFeatures.py`: Build/append vortex feature vectors (distance, vorticity stats, speeds, duration) from labeled tracks into `vortex_features.csv`.
- `vortexClassification.py`: Train/evaluate classifiers (RandomForest/GB) on `vortex_features.csv`; optional hyperparameter search and SHAP analysis.
- `vortexAnalysis.py`: Deprecated plotting of labeled vortex statistics.

## Miscellaneous analysis/sandboxes
- `imageMetrics.py`: Compute offset, MSE, PSNR, and SSIM utilities.
- `justification_normalisation.py`: Low-pass filtering example on Lilypad pressure logs.
- `find_crop_start.py`: Video alignment utility (duplicated above for visibility).
- `sandboxInterpo.py`: Spline interpolation/duplicate-removal sandbox for contour points using `x.csv`/`y.csv`.
- `sandboxTimeInterpo.py`: Sandbox for time derivatives and signal reconstruction with SINDy.
- `redrawFromCSV.py`: Rebuild videos/animations from pressure CSVs and compare against experimental traces.
- `testDMD.py`: Dynamic Mode Decomposition experiments on pressure maps (BOPDMD).
- `testGPR.py`: Toy multisource GPR sampler demonstrating ASMR-style active learning.

## python_extras
- `dynamic_display_UI_04092025.py`: Live/replay UI that tails a CSV of pressures and animates them on a custom sensor layout.
- `review_UI.py`: Apply a Butterworth band-pass to ESP sensor CSVs and write filtered output.
- `stitch_video.py`: SIFT-based panorama stitching across videos; includes helpers for homography and batch jobs.
