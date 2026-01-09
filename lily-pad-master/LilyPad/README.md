# LilyPad folder

This folder is a Processing sketch that implements the LilyPad 2D
fluid-structure interaction solver (BDIM) plus a set of example
simulations, geometry builders, visualization helpers, and data I/O
utilities. Each `.pde` file is a Processing tab; all tabs compile
together into a single sketch.

## Quick start

1) Open the `LilyPad` folder in Processing (the tab list is built from
   the `.pde` files).
2) Run the sketch.
3) Only one `setup()`/`draw()` pair can be active at a time. Many tabs
   include example code blocks at the top - copy them into
   `LilyPad.pde` or comment/uncomment the block you want to run.

`LilyPad.pde` is the current main tab. In this repo it is configured
to run a CSV-driven "digital twin" case and save pressure/vorticity
maps to disk.

## Typical simulation flow

Most examples follow the same pattern:

```java
int n = (int)pow(2, 7);
Window view = new Window(n, n);
Body body = new CircleBody(n/3, n/2, n/8, view);
BDIM flow = new BDIM(n, n, 0, body, n/8000., true);
FloodPlot flood = new FloodPlot(view);

void draw() {
  body.follow();
  flow.update(body);
  flow.update2();
  flood.display(flow.u.curl());
  body.display();
}
```

Key concepts:
- `n`, `m` are grid sizes (often with 2 ghost cells).
- `Window` maps grid coordinates to screen pixels.
- `Body` or `BodyUnion` defines the geometry.
- `BDIM` or `TwoPhase` updates the flow; use `update()` and `update2()`.
- `FloodPlot`, `StreamPlot`, or `ParticlePlot` render the result.

## File guide (by category)

### Core numerics and fields
- `BDIM.pde`: Base solver for velocity/pressure using the BDIM method,
  including `update()`, `update2()`, CFL checks, and steady-state solver.
- `TwoPhase.pde`: Extends BDIM with free-surface tracking and variable density.
- `Field.pde`: Scalar field storage, interpolation, advection, and operators.
- `VectorField.pde`: Staggered vector field (u,v) with divergence/curl, advection.
- `SharpField.pde`: Sharp interface advection for two-phase modeling.
- `FreeInterface.pde`: SharpField with density ratio handling.
- `PoissonMatrix.pde`: Sparse Poisson operator representation.
- `MG.pde`: Multigrid solver for Poisson problems.

### Geometry and bodies
- `Body.pde`: Base rigid body class plus `EllipseBody` and `CircleBody`.
- `BodyUnion.pde`: Combines multiple `Body` instances into one union.
- `LineSegBody.pde`: Polyline body with thickness (1D body).
- `PixelBody.pde`: Body defined by a per-cell solid fraction (WIP).
- `NACA.pde`: Parametric NACA foil geometry.
- `FlexNACA.pde`: NACA foil with a traveling-wave deformation.
- `ChaoticEllipse.pde`: Ellipse with custom `react()` dynamics.
- `Rectangle.pde`, `Triangle.pde`: Custom polygon bodies used in experiments.
- `VWall.pde`, `TestLine.pde`: Simple wall/line bodies for fixtures.
- `Sensor.pde`: Small sensor geometry used in the digital-twin setup.
- `CircleArray.pde`: Ringed cylinder bundle arrangement.
- `CounterRotatingCylinders.pde`: Pair of counter-rotating control cylinders.
- `CustomCircle.pde`: CSV-driven circle motion (`CSV2CircleTwin`).
- `DigitalTwin.pde`: Hard-coded body union matching a physical experiment.
- `CSV2DigitalTwin.pde`: Builds a time-varying body from CSV trajectories.

### Visualization and diagnostics
- `FloodPlot.pde`: Scalar field color map with legend support.
- `StreamPlot.pde`: FloodPlot plus streamlines from a flow field.
- `Particle.pde`: Lagrangian tracer particle.
- `Swarm.pde`: Particle manager (streaklines, sources, inlets).
- `ParticlePlot.pde`: Particle rendering colored by a scalar field.
- `CirculationFinder.pde`: Detects vortex cores and estimates circulation.
- `Window.pde`: Screen-to-grid mapping utilities (also includes `Scale`).
- `OrthoNormal.pde`: Geometry helper for body segment normals.

### Data I/O
- `SaveData.pde`: Writes time series data at body points (pressure, forces).
- `SaveVectorField.pde`: Writes full-field u/v/p grids to file.
- `ReadData.pde`: Reads tabular time series for actuation or motion inputs.

### Example and test cases
- `LilyPad.pde`: Main sketch tab (currently CSV-driven digital twin + map saving).
- `AncientSwimmer.pde`: Two-foil swimmer (plesiosaur) with thrust logging.
- `BlindFish.pde`: Cylinder passes a foil; saves pressure snapshots.
- `InlineFoilTest.pde`: Inline flapping foil with optional control inputs.
- `Duncan.pde`: Foil beneath a free surface (two-phase flow).
- `Vortex.pde`: Rankine vortex pair evolution in a quiescent field.
- `UnsteadyLiftControl.pde`: Lift controller with estimator/observer.
- `TestSaving.pde`: Example of saving full pressure maps.

## Data folders

- `data/`: Processing assets (font and image); required for some displays.
- `saved/`: Default output directory used by many examples.
- `testDataSave/`: Reference pressure map CSVs.
- `plesiosaur/`: Output from the `AncientSwimmer` example.
- `chinaBenchmark/`: Output datasets for digital-twin benchmarks.

## Notes

- Many tabs contain example code blocks at the top. They are meant to
  be pasted into the active `LilyPad.pde` `setup()`/`draw()` and run one
  at a time.
- If you run via `processing-java`, `LilyPad.pde` supports optional
  command-line arguments when `automated` is set to `true`.
