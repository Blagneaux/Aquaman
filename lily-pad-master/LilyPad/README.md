LilyPad folder guide
====================

This folder contains the Processing sketch and utilities for running 2D fluid–structure interaction simulations (BDIM) and the custom HAACHAMA digital-twin pipeline used in this repo.

Prerequisites
-------------
- Processing IDE (tested with the PDE runner) available on your machine.
- Open the sketch by launching `LilyPad.pde` from this folder in Processing.
- CSV inputs referenced in the HAACHAMA pipeline need to exist locally (see “Digital-twin pipeline”).

Folder layout (local)
---------------------
- `LilyPad.pde` — main Processing sketch; currently set up for the HAACHAMA pipeline (fish or circle digital twin).
- `*.pde` classes — solver and geometry building blocks (`BDIM.pde`, `Field.pde`, `Body.pde`, `NACA.pde`, `TestLine.pde`, etc.).
- `CustomCircle.pde` — circle twin driven by spine CSV data.
- `CSV2DigitalTwin.pde` — builds a deforming fish body from CSV coordinate and velocity files.
- `data/` — Processing resources (font, logo).
- `chinaBenchmark/`, `plesiosaur/`, `testDataSave/`, `saved/` — example outputs and saved frames/fields.

Running the sketch
------------------
1) Open `LilyPad.pde` in Processing.
2) Verify file paths inside `LilyPad.pde` (defaults point to `D:/thomas_files/...` and a local download for `y_dot`). Update them to your paths before running.
3) Click Run. The window size and grid are set in `setup()`. Vorticity is plotted by default.
4) Frames save to `saved/frame-####.png` when `automated == false`. Data CSV output lines are currently commented out; enable the relevant `createWriter` calls if you want files emitted.

Digital-twin pipeline (current defaults)
----------------------------------------
- Inputs: three CSVs (`x.csv`, `y.csv`, `y_dot_lowpass_30Hz_fs100.csv`) and a spine CSV (`spines_interpolated.csv`). Paths are set near the top of `setup()`.
- Flags:
  - `fish` (boolean) — true to run deforming fish (`CSV2DigitalTwin`); false to run circle (`CSV2CircleTwin`).
  - `pressure` — false plots vorticity; true switches to pressure maps (output writer is commented).
  - `automated` — when true, parses command-line args (`Uturn`, `startIndex`, `haato`, `haachama`, `fish`) after a `--args` offset.
- Walls: two `TestLine` bodies create channel walls; bodies combine into a `BodyUnion` before solving.
- Outputs: vorticity/pressure field per frame; optional CSV maps (uncomment `createWriter` blocks). Data accumulation uses `pressureDataList`; `dataAdd()` writes a time-major CSV when the simulation ends.

Other examples
--------------
Many `.pde` files contain short commented example setups at the top. To try one, copy the example `setup()/draw()` into `LilyPad.pde` (or temporarily comment out the HAACHAMA block) and run.

Tips and gotchas
----------------
- Grid sizes: `numCols = 2^8`, `numRows = 2^7` in the current pipeline; update these if you resize the domain.
- Absolute paths: replace `D:/...` and `C:/Users/...` with your own before running, otherwise the sketch will fail to load CSVs.
- Steady state solver: `BDIM.solveSteadyState()` exists for sandboxed baseflow runs (see commented blocks at the top of `LilyPad.pde`).
