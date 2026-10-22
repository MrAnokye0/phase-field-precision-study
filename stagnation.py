"""stagnation.py -- does float32 drop explicit-Euler increments? (Cahn-Hilliard, Landau)

For saved float32 snapshots of the dt-sweep runs: apply ONE real float32 step (package code)
and compare the realised increment with the exact increment computed in float64 (NumPy)
from the very same state.

Run from the precision-study folder:  python study/stagnation.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)
TARGETS = [20, 50, 100, 150, 200]

import jax.numpy as jnp
from pfm.manager import SimulationManager


def lap(a):
    return np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a


rows = []
folders = sorted(glob.glob(os.path.join(RES, "sweep_dt*_f32")))
for p in folders:
    name = os.path.basename(p)
    dt = float(name.split("_")[1][2:])
    if dt > 0.01:
        continue
    cfg = toml.load(os.path.join(p, "config.toml"))
    cfg["write_path"] = os.path.join(RES, "_scratch")
    eps = cfg["landau"]["epsilon"]
    kappa = cfg["interface_scalar"] * cfg.get("k", 1.0)
    M = cfg.get("M", 1.0)
    mgr = SimulationManager(cfg)
    log = pd.read_csv(os.path.join(p, "log.csv"))
    fields = np.load(os.path.join(p, "fields.npy"))
    t = log["t"].values
    for T in TARGETS:
        i = int(np.argmin(np.abs(t - T)))
        if abs(t[i] - T) > 0.05 or not bool(log["finite"].iloc[i]):
            continue
        rho32 = fields[i].astype(np.float32)
        assert np.array_equal(rho32.astype(np.float64), fields[i])  # snapshot is exactly float32
        new32 = np.asarray(mgr._evolve_n_steps(jnp.asarray(rho32)[None], 1))[0]
        d32 = new32.astype(np.float64) - rho32.astype(np.float64)      # realised increment
        r = rho32.astype(np.float64)
        mu = -eps * r + r**3 - kappa * lap(r)
        d64 = M * dt * lap(mu)                                          # exact increment
        ulp = np.spacing(np.abs(rho32)).astype(np.float64)
        abs64 = np.abs(d64)
        stag = d32 == 0
        rows.append(dict(
            dt=dt, t=T, max_abs_rho=float(np.abs(r).max()),
            median_incr_over_ulp=float(np.median(abs64 / ulp)),
            frac_incr_below_half_ulp=float((abs64 < 0.5 * ulp).mean()),
            frac_stagnated=float(stag.mean()),
            share_of_movement_lost=float(abs64[stag].sum() / abs64.sum()),
            rel_step_error=float(np.linalg.norm(d32 - d64) / np.linalg.norm(d64)),
            mean_dr32_per_step=float(d32.mean()),
            mean_dr64_per_step=float(d64.mean()),
        ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, "stagnation.csv"), index=False)
pd.options.display.float_format = lambda x: f"{x:.4g}"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False))
