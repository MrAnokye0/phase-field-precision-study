"""analyze_sweep.py -- dt sweep: Euler stability limits and float32-vs-float64 differences.

Run from the precision-study folder:  python study/analyze_sweep.py
Reads study/results/sweep_dt*_f32|f64; writes tables to study/tables/, figures to study/figures/
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
TARGETS = [20, 50, 100, 200]


def load(name):
    d = os.path.join(RES, name)
    log = pd.read_csv(os.path.join(d, "log.csv"))
    fields = np.load(os.path.join(d, "fields.npy"))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    amax = np.maximum(log["rho_min"].abs(), log["rho_max"].abs()).values
    healthy = log["finite"].astype(bool).values & (amax < 1e3)
    n_h = len(healthy) if healthy.all() else int(np.argmin(healthy))
    t_bad = float(log["t"].iloc[n_h]) if n_h < len(log) else np.nan
    return dict(log=log.iloc[:n_h].reset_index(drop=True), fields=fields[:n_h],
                meta=meta, n_h=n_h, t_first_bad=t_bad)


def stats(d):
    log, fields = d["log"], d["fields"]
    n = d["n_h"]
    N = fields.shape[1]
    sign = (-1.0) ** np.add.outer(np.arange(N), np.arange(N))
    checker = np.abs((fields * sign).mean(axis=(1, 2)))  # (pi,pi) grid-scale mode amplitude
    E = log["E_ref64"].values
    amax = np.maximum(log["rho_min"].abs(), log["rho_max"].abs()).values
    thr = d["meta"]["args"]["log_every"] / d["meta"]["median_chunk_s"]
    return dict(
        status="ok" if d["meta"]["status"] == "ok" else "blowup",
        t_last_ok=float(log["t"].iloc[-1]), t_first_bad=d["t_first_bad"],
        final_E=float(E[-1]), final_max_abs_rho=float(amax[-1]),
        E_increases=int((np.diff(E) > 1e-12 * np.abs(E[1:])).sum()),
        checker_max=float(checker.max()), checker_final=float(checker[-1]),
        steps_per_s=thr,
    )


tags = {}
for p in glob.glob(os.path.join(RES, "sweep_dt*_f32")):
    s = os.path.basename(p).split("_")[1][2:]
    tags[float(s)] = s
dts = sorted(tags)
runs = {}
for dt in dts:
    for short in ("f32", "f64"):
        runs[(dt, short)] = load(f"sweep_dt{tags[dt]}_{short}")

# ---- Table A: stability --------------------------------------------------------
rowsA = []
for dt in dts:
    for short in ("f32", "f64"):
        rowsA.append(dict(dt=dt, prec=short, **stats(runs[(dt, short)])))
A = pd.DataFrame(rowsA)
A.to_csv(os.path.join(TAB, "sweep_stability.csv"), index=False)

# ---- Table B: float32 vs float64 at each dt --------------------------------------
rowsB = []
for dt in dts:
    a, b = runs[(dt, "f32")], runs[(dt, "f64")]
    n = min(a["n_h"], b["n_h"])
    t = b["log"]["t"].values[:n]
    row = dict(dt=dt)
    for T in TARGETS:
        i = int(np.argmin(np.abs(t - T)))
        if abs(t[i] - T) <= 0.7:
            fa, fb = a["fields"][i].ravel(), b["fields"][i].ravel()
            row[f"relL2_t{T}"] = float(np.linalg.norm(fa - fb) / np.linalg.norm(fb))
            row[f"tused_t{T}"] = float(t[i])
        else:
            row[f"relL2_t{T}"] = np.nan
            row[f"tused_t{T}"] = np.nan
    sa, sb = stats(a), stats(b)
    both_ok = sa["status"] == "ok" and sb["status"] == "ok"
    row["both_ok"] = both_ok
    row["final_dE_rel"] = (sa["final_E"] - sb["final_E"]) / abs(sb["final_E"]) if both_ok else np.nan
    row["f32_mass_drift"] = float(a["log"]["M_ref64"].iloc[-1] - a["log"]["M_ref64"].iloc[0])
    row["speed_f32_over_f64"] = sa["steps_per_s"] / sb["steps_per_s"]
    rowsB.append(row)
B = pd.DataFrame(rowsB)
B.to_csv(os.path.join(TAB, "sweep_precision.csv"), index=False)

pd.options.display.float_format = lambda x: f"{x:.4g}"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print("\n=== A. Stability (per dt and precision) ===")
print(A.to_string(index=False))
print("\n=== B. float32 vs float64 at the same dt (relative L2 field difference) ===")
cols = ["dt"] + [f"relL2_t{T}" for T in TARGETS] + ["both_ok", "final_dE_rel", "f32_mass_drift", "speed_f32_over_f64"]
print(B[cols].to_string(index=False))

# ---- scaling of the float32 error with dt ---------------------------------------
print("\n=== C. Scaling: log-log slope of relL2 vs dt (float32 vs float64) ===")
for T in (20, 50):
    ok = B[np.abs(B[f"tused_t{T}"] - T) < 0.05].dropna(subset=[f"relL2_t{T}"])
    for label, sub in (("all dt", ok), ("dt<=0.01", ok[ok["dt"] <= 0.01])):
        if len(sub) >= 3:
            slope = np.polyfit(np.log(sub["dt"]), np.log(sub[f"relL2_t{T}"]), 1)[0]
            print(f"t={T:>3d}  {label:>9s}  n={len(sub)}  slope={slope:+.2f}")
        else:
            print(f"t={T:>3d}  {label:>9s}  not enough points")

# ---- Figure 5: divergence vs time for each dt ------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
cmap = plt.get_cmap("viridis")
for k, dt in enumerate(dts):
    a, b = runs[(dt, "f32")], runs[(dt, "f64")]
    n = min(a["n_h"], b["n_h"])
    if n < 3:
        continue
    t = b["log"]["t"].values[:n]
    fa, fb = a["fields"][:n].reshape(n, -1), b["fields"][:n].reshape(n, -1)
    y = np.linalg.norm(fa - fb, axis=1) / np.linalg.norm(fb, axis=1)
    ax.semilogy(t[1:], np.where(y[1:] > 0, y[1:], np.nan), color=cmap(k / max(1, len(dts) - 1)), label=f"dt={dts[k]:g}")
ax.set_xlabel("time t")
ax.set_ylabel("relative L2 difference, float32 vs float64")
ax.set_title("Euler dt sweep (curves end where a run stops being finite)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_sweep_divergence.png"), dpi=150)
plt.close(fig)

# ---- Figure 6: error at fixed time vs dt -----------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for T, mk in ((20, "o"), (50, "s")):
    ok = B[np.abs(B[f"tused_t{T}"] - T) < 0.05].dropna(subset=[f"relL2_t{T}"])
    ax.loglog(ok["dt"], ok[f"relL2_t{T}"], mk + "-", label=f"t={T}")
ax.set_xlabel("dt")
ax.set_ylabel("relative L2 difference, float32 vs float64")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_error_vs_dt.png"), dpi=150)
plt.close(fig)

print("\nTables written to", TAB)
print("Figures written to", FIG)
