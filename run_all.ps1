# run_all.ps1 -- reproduces every run, table and figure of the precision study.
# Run from the folder containing study/ and phase-field-models/ with the virtual environment active:
#     .\study\run_all.ps1
# About 20-30 minutes on a laptop CPU (the float64 Wertheim run is the longest).

function Run { & python @args; if ($LASTEXITCODE -ne 0) { throw "python $args failed" } }

# 1. Landau Cahn-Hilliard, explicit Euler, dt = 0.001, t = 500, plus the controls
Run study\run_sim.py --name euler_f32 --precision float32 --integrator euler --dt 0.001 --steps 500000 --log-every 2000
Run study\run_sim.py --name euler_f64 --precision float64 --integrator euler --dt 0.001 --steps 500000 --log-every 2000
Run study\run_sim.py --name euler_f64_roundinit --precision float64 --integrator euler --dt 0.001 --steps 500000 --log-every 2000 --round-init-f32
foreach ($pert in "1e-14", "1e-10", "1e-7") {
  Run study\run_sim.py --name "euler_f64_p$pert" --precision float64 --integrator euler --dt 0.001 --steps 500000 --log-every 2000 --perturb $pert
}
Run study\compare_euler.py

# 2. dt sweep, explicit Euler, t = 200
$dts = 0.001, 0.0025, 0.005, 0.01, 0.0125, 0.015625, 0.016, 0.02
foreach ($dt in $dts) {
  $tag = $dt.ToString([cultureinfo]::InvariantCulture)
  $le = [int][math]::Round(2 / $dt)
  $steps = [int][math]::Round(200 / $dt)
  foreach ($p in "float32", "float64") {
    $short = $p.Replace("float", "f")
    Run study\run_sim.py --name "sweep_dt${tag}_${short}" --precision $p --integrator euler --dt $dt --steps $steps --log-every $le
  }
}
foreach ($dt in 0.013, 0.014, 0.015) {
  $tag = $dt.ToString([cultureinfo]::InvariantCulture)
  $steps = 100 * [int][math]::Round(200 / $dt / 100)
  foreach ($p in "float32", "float64") {
    $short = $p.Replace("float", "f")
    Run study\run_sim.py --name "sweep_dt${tag}_${short}" --precision $p --integrator euler --dt $dt --steps $steps --log-every 100
  }
}
Run study\analyze_sweep.py
Run study\stagnation.py

# 3. Semi-implicit spectral integrator, dt = 0.01, t = 500
Run study\run_sim.py --name spec_f32 --precision float32 --integrator spectral --dt 0.01 --steps 50000 --log-every 200
Run study\run_sim.py --name spec_f64 --precision float64 --integrator spectral --dt 0.01 --steps 50000 --log-every 200
Run study\compare_pair.py --ref spec_f64 --test spec_f32 --label spectral_dt0.01

# 4. Simple Wertheim model, N = 512, dt = 1e-5, 20000 steps
$cfg = "phase-field-models\Examples\simple wertheim\euler_noautodiff.toml"
Run study\run_cfg.py --base-config $cfg --name wert_f32 --precision float32 --steps 20000 --log-every 2000
Run study\run_cfg.py --base-config $cfg --name wert_f64 --precision float64 --steps 20000 --log-every 2000
Run study\wert_analysis.py --ref wert_f64 --test wert_f32 --label wertheim_euler
