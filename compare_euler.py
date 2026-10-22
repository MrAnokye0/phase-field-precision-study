"""compare_euler.py -- compare the Euler runs: float32 vs the float64 reference,
plus perturbed float64 runs that show how sensitive the system is to ANY tiny change.

Run from the precision-study folder:  python study/compare_euler.py
Writes tables to study/tables/ and figures to study/figures/
"""
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

REF = "euler_f64"
RUNS = ["euler_f32", "euler_f64_p1e-14", "euler_f64_p1e-10", "euler_f64_p1e-7"]
LABEL = {
    "euler_f64": "float64 (reference)",
    "euler_f32": "float32",
    "euler_f64_p1e-14": "float64, 1e-14 perturbation",
    "euler_f64_p1e-10": "float64, 1e-10 perturbation",
    "euler_f64_p1e-7": "float64, 1e-7 perturbation",
}
TIMES = [20, 50, 100, 150, 200, 300, 500]


def load(name):
    d = os.path.join(RES, name)
    log = pd.read_csv(os.path.join(d, "log.csv"))
    fields = np.load(os.path.join(d, "fields.npy"))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return dict(log=log, fields=fields, meta=meta)


def first_cross(t, y, thr):
    idx = np.where(y > thr)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def fmtcell(v):
    return f"{v:.3e}" if isinstance(v, (float, np.floating)) else str(v)


runs = {n: load(n) for n in [REF] + RUNS}
ref = runs[REF]
t = ref["log"]["t"].values
fr = ref["fields"].reshape(len(t), -1)
Eref = ref["log"]["E_ref64"].values

series, summary, timepoints = {}, [], []
for name, d in runs.items():
    log = d["log"]
    f = d["fields"].reshape(len(t), -1)
    assert f.shape == fr.shape and (log["step"].values == ref["log"]["step"].values).all()
    E = log["E_ref64"].values
    E_in = log["E_inproc"].values
    drift = log["M_ref64"].values - log["M_ref64"].values[0]
    row = dict(
        run=name, status=d["meta"]["status"], dtype=d["meta"]["rho_dtype"],
        steps_per_s=d["meta"]["steps_done"] / d["meta"]["wall_s"],
        final_E_ref64=E[-1],
        final_mass_drift=drift[-1], max_abs_mass_drift=float(np.abs(drift).max()),
        E_increases_ref64=int((np.diff(E) > 1e-12 * np.abs(E[1:])).sum()),
        E_increases_inproc=int((np.diff(E_in) > 1e-12 * np.abs(E_in[1:])).sum()),
        max_E_readout_rel_err=float(np.max(np.abs(E_in - E) / np.abs(E))),
    )
    if name != REF:
        diff = f - fr
        rel_l2 = np.linalg.norm(diff, axis=1) / np.linalg.norm(fr, axis=1)
        maxabs = np.abs(diff).max(axis=1)
        mism = (np.sign(f) != np.sign(fr)).mean(axis=1)
        dE = (E - Eref) / np.abs(Eref)
        series[name] = dict(rel_l2=rel_l2, dE=dE, drift=drift)
        row.update(
            final_rel_L2=rel_l2[-1], final_sign_mismatch=mism[-1], final_dE_rel=dE[-1],
            t_relL2_gt_1e_2=first_cross(t, rel_l2, 1e-2),
            t_relL2_gt_1e_1=first_cross(t, rel_l2, 1e-1),
        )
        for T in TIMES:
            i = int(np.argmin(np.abs(t - T)))
            timepoints.append(dict(run=name, t=int(round(float(t[i]))), rel_L2=rel_l2[i],
                                   max_abs_diff=maxabs[i], sign_mismatch=mism[i], dE_rel=dE[i]))
    else:
        series[name] = dict(drift=drift)
    summary.append(row)

tp_df = pd.DataFrame(timepoints)
sdf = pd.DataFrame(summary)
tp_df.to_csv(os.path.join(TAB, "euler_timepoints.csv"), index=False)
sdf.to_csv(os.path.join(TAB, "euler_summary.csv"), index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print("\n=== Differences to the float64 reference at selected times ===")
print(tp_df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
print("\n=== Per-run summary ===")
print(sdf.set_index("run").astype(object).T.map(fmtcell).to_string())

# ---- Figure 1: field difference vs time -------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in RUNS:
    y = series[name]["rel_l2"]
    ax.semilogy(t, np.where(y > 0, y, np.nan), label=LABEL[name],
                lw=2.2 if name == "euler_f32" else 1.3)
ax.set_xlabel("time t")
ax.set_ylabel("relative L2 difference to float64 reference")
ax.set_title("Field divergence: float32 vs perturbed float64")
ax.grid(alpha=0.3, which="both")
ax.legend()
savefig(fig, "fig1_field_divergence.png")

# ---- Figure 2: energy ---------------------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
a1.plot(t, ref["log"]["E_ref64"], label="float64 (reference)")
a1.plot(t, runs["euler_f32"]["log"]["E_ref64"], label="float32 (energy recomputed in float64)")
a1.plot(t, runs["euler_f32"]["log"]["E_inproc"], "--", lw=0.9, label="float32 (energy as computed in float32)")
a1.set_ylabel("average free energy")
a1.legend()
a1.grid(alpha=0.3)
for name in RUNS:
    y = np.abs(series[name]["dE"])
    a2.semilogy(t, np.where(y > 0, y, np.nan), label=LABEL[name],
                lw=2.2 if name == "euler_f32" else 1.3)
a2.set_xlabel("time t")
a2.set_ylabel("|relative energy difference to reference|")
a2.grid(alpha=0.3, which="both")
a2.legend()
savefig(fig, "fig2_energy.png")

# ---- Figure 3: mass drift -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in [REF] + RUNS:
    ax.plot(t, series[name]["drift"], label=LABEL[name], lw=2.2 if name == "euler_f32" else 1.2)
ax.set_yscale("symlog", linthresh=1e-12)
ax.set_xlabel("time t")
ax.set_ylabel("mean density minus its initial value")
ax.set_title("Mass conservation (mean computed in float64 from the stored field)")
ax.grid(alpha=0.3)
ax.legend()
savefig(fig, "fig3_mass_drift.png")

# ---- Figure 4: final fields and differences ---------------------------------
last = {n: runs[n]["fields"][-1] for n in runs}
tfin = int(round(float(t[-1])))
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for ax, n in zip(axs[0], [REF, "euler_f32", "euler_f64_p1e-7"]):
    im0 = ax.imshow(last[n], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(LABEL[n] + f", t={tfin}")
    ax.axis("off")
fig.colorbar(im0, ax=axs[0], shrink=0.8)
for ax, n in zip(axs[1], ["euler_f32", "euler_f64_p1e-10", "euler_f64_p1e-7"]):
    im1 = ax.imshow(last[n] - last[REF], cmap="PuOr", vmin=-2, vmax=2)
    ax.set_title("difference: " + LABEL[n])
    ax.axis("off")
fig.colorbar(im1, ax=axs[1], shrink=0.8)
fig.savefig(os.path.join(FIG, "fig4_final_fields.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nTables written to", TAB)
print("Figures written to", FIG)
