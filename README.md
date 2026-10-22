# Numerical precision and reproducibility study: JAX phase-field models

Edward Gyasi Anokye. Screening task for the CAMMP research positions, October 2026.

## In plain language
The package simulates phase separation: a mixture that starts nearly uniform "un-mixes" into two regions, like oil and
water, by repeatedly updating the value at every point of a grid. Computers store numbers with limited digits (float32:
about 7 significant digits, float64: about 16), so every update carries a tiny rounding error. This study runs the same
simulations in both precisions, with everything else identical, and measures how much the results differ: the final
patterns, the energy over time, conservation of the total amount of material (mass) and numerical stability.
Float64 is the reference.

## Main findings
- With the shipped Landau example and explicit Euler (dt = 0.001), float32 ends up with a clearly different pattern
  (relative field difference 0.43 at t = 500, energy 1.9% off). Perturbing the float64 start by 1e-14 to 1e-7 changes the
  result by at most 1.7e-4, so the cause is float32 arithmetic and not general sensitivity.
- Smaller time steps make float32 worse (field error at t = 200: 1.9e-2 at dt = 0.001, 2.0e-4 at dt = 0.01).
- Both precisions are stable up to dt = 0.015 and unstable from dt = 1/64; float32 fails earlier.
- Float32 does not conserve the mean density and loses monotone energy decrease; its in-run energy is accurate to ~4e-7.
- The semi-implicit spectral integrator stays closer to float64 (field difference <= 1.1e-2 at dt = 0.01).
- Simple Wertheim model (first t = 0.2 only): no stalling; the float32 energy error is almost entirely mass drift.
- Float32 results were bit-identical on two machines (same library versions); float64 differed at the 1e-13 level.

## What is in this folder
    README.md           this file
    report/             precision_report.docx (the short report)
    requirements.txt    exact package versions
    run_all.ps1         reruns every experiment and analysis (about 20-30 minutes)
    *.py                the scripts (below)
    tables/             comparison tables (CSV) written by the analysis scripts
    figures/            plots (PNG) used in the report
    runs/<run>/         per simulation: config.toml (settings), meta.json (versions, timing, status),
                        log.csv (energy and mean density over time)
The large snapshot files (fields.npy) are not included; run_all.ps1 regenerates them.

## Scripts
- run_sim.py         one Landau run (64 x 64) with full-precision logging; also the perturbation controls
- run_cfg.py         runs any example TOML of the repository (used for the Wertheim model)
- compare_euler.py   Euler float32 vs float64 vs perturbed float64 (tables, figures 1-4)
- analyze_sweep.py   time-step sweep: stability limits and float32 error versus dt
- stagnation.py      one real float32 step versus the exact float64 step (how well float32 holds an increment)
- compare_pair.py    reusable pair comparison (used for the spectral integrator)
- wert_analysis.py   Wertheim float32 vs float64 (energies re-evaluated in float64)
- make_report.py     builds the Word report from the tables and figures

## Source of the code and environment
The shared Drive folder ("2. PF JAX") contained only the repository README and images, not the source. The code is the public
repository https://github.com/ajvetturini/phase-field-models at commit c90978331a1f83dbf056e70e624dbe098d667d9a, used
unmodified. Windows, CPU only, Python 3.12.10, JAX/jaxlib 0.11.2, NumPy 2.2.4 (see requirements.txt and each meta.json).

## Experiments
Common Landau settings: Cahn-Hilliard, N = 64, dx = 1, M = 1, k = 1, interface_scalar = 2, epsilon = 0.9, initial field
loaded from Examples/landau/standard_input.dat (identical for both precisions; seed unused).

| Group | Runs | Model / integrator | dt | Length |
|---|---|---|---|---|
| Euler main | euler_f32, euler_f64 | Landau CH, explicit Euler | 0.001 | 500000 steps (t = 500) |
| Controls | euler_f64_roundinit, euler_f64_p1e-14, _p1e-10, _p1e-7 | same | 0.001 | 500000 steps |
| dt sweep | sweep_dt*_f32 / _f64 (11 values, 0.001 to 0.02) | explicit Euler | see runs | to t = 200 |
| Spectral | spec_f32, spec_f64 | semi-implicit spectral | 0.01 | 50000 steps (t = 500) |
| Wertheim | wert_f32, wert_f64 | simple Wertheim CH, Euler, N = 512 | 1e-5 | 20000 steps (t = 0.2) |

## Method notes
- Precision is set with float_type in the TOML. It is a process-wide JAX setting, so every run is a separate process.
- Float64 is the reference. Differences: relative L2 norm of (float32 field - float64 field), maximum pointwise
  difference, fraction of pixels in opposite phase, relative energy difference.
- Energies are re-evaluated in float64 from the stored fields (snapshots of float32 runs are exact float32 values).
  For Landau an independent NumPy implementation is used (agrees with the package to 1.5e-16 in float64); for Wertheim
  the package's own functional in a float64 process.
- Mass = mean density, summed with math.fsum in float64 from the stored field.
- Controls: float64 from float32-rounded start (a no-op for Landau, whose start is exactly float32-representable) and
  float64 with random relative perturbations of the start (1e-14, 1e-10, 1e-7).

## Limits of the study
One machine pair for bit-level reproducibility (same JAX/NumPy versions, CPU only); a 64 x 64 Landau grid and one initial
condition; Wertheim covers only the first t = 0.2 of the example's 1e8 steps; no GPU runs. Hypotheses that were not
tested are marked as such in the report.

## How to reproduce
Tested end to end on the author's Windows laptop (CPU) with Python 3.12.
1. Create a folder, e.g. precision-study, and in it: python -m venv .venv ; activate it ; git clone
   https://github.com/ajvetturini/phase-field-models ; in that clone run: git checkout
   c90978331a1f83dbf056e70e624dbe098d667d9a ; pip install -e .
2. pip install -r requirements.txt (if pip cannot install the pfm line, the clone from step 1 is enough).
3. Create a folder named study next to phase-field-models and copy the scripts and run_all.ps1 into it.
4. From the precision-study folder run:  .\study\run_all.ps1   then   python study\make_report.py
