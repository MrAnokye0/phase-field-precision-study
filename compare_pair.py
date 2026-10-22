"""compare_pair.py -- compare any two runs (a float32 test run against a float64 reference).

Run from the precision-study folder, e.g.
  python study/compare_pair.py --ref spec_f64 --test spec_f32 --label spectral_dt0.01
Writes tables/pair_<label>_*.csv and figures/pair_<label>.png
"""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--test", required=True)
p.add_argument("--label", required=True)
p.add_argument("--times", type=float, nargs="+", default=[20, 50, 100, 150, 200, 300, 500])
a = p.parse_args()


def load(name):
    d = os.path.join(RES, name)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return dict(log=pd.read_csv(os.path.join(d, "log.csv")), fields=np.load(os.path.join(d, "fields.npy")), meta=meta)


R, T = load(a.ref), load(a.test)
n = min(len(R["log"]), len(T["log"]))
t = R["log"]["t"].values[:n]
assert np.allclose(t, T["log"]["t"].values[:n]), "the two runs must share the same time grid"
fr, ft = R["fields"][:n].reshape(n, -1), T["fields"][:n].reshape(n, -1)
diff = ft - fr
rel = np.linalg.norm(diff, axis=1) / np.linalg.norm(fr, axis=1)
maxabs = np.abs(diff).max(axis=1)
mism = (np.sign(ft) != np.sign(fr)).mean(axis=1)
Er, Et = R["log"]["E_ref64"].values[:n], T["log"]["E_ref64"].values[:n]
dE = (Et - Er) / np.abs(Er)

rows = []
for tt in a.times:
    i = int(np.argmin(np.abs(t - tt)))
    if abs(t[i] - tt) <= 0.7:
        rows.append(dict(t=float(t[i]), rel_L2=rel[i], max_abs_diff=maxabs[i], sign_mismatch=mism[i], dE_rel=dE[i]))
tp = pd.DataFrame(rows)


def summ(d, name):
    lg = d["log"]
    E, Ei = lg["E_ref64"].values, lg["E_inproc"].values
    drift = lg["M_ref64"].values - lg["M_ref64"].values[0]
    return dict(run=name, status=d["meta"]["status"], dtype=d["meta"]["rho_dtype"],
                steps_per_s=d["meta"]["args"]["log_every"] / d["meta"]["median_chunk_s"],
                final_E=E[-1], final_max_abs_rho=float(np.abs(d["fields"][-1]).max()),
                final_mass_drift=drift[-1], max_abs_mass_drift=float(np.abs(drift).max()),
                E_increases=int((np.diff(E) > 1e-12 * np.abs(E[1:])).sum()),
                max_E_readout_rel_err=float(np.max(np.abs(Ei - E) / np.abs(E))))


sm = pd.DataFrame([summ(R, a.ref), summ(T, a.test)])
tp.to_csv(os.path.join(TAB, f"pair_{a.label}_timepoints.csv"), index=False)
sm.to_csv(os.path.join(TAB, f"pair_{a.label}_summary.csv"), index=False)

pd.options.display.float_format = lambda x: f"{x:.4g}"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(f"\n=== {a.test} vs {a.ref}: differences over time ===")
print(tp.to_string(index=False))
print("\n=== per-run summary ===")
print(sm.set_index("run").T.to_string())

fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
axs[0].semilogy(t[1:], np.where(rel[1:] > 0, rel[1:], np.nan))
axs[0].set_xlabel("time t")
axs[0].set_ylabel("relative L2 difference")
axs[0].set_title("field difference")
axs[1].plot(t, Er, label=a.ref)
axs[1].plot(t, Et, "--", label=a.test + " (energy re-evaluated in float64)")
axs[1].set_xlabel("time t")
axs[1].set_ylabel("average free energy")
axs[1].set_title("energy")
axs[1].legend(fontsize=7)
for d, nm in ((R, a.ref), (T, a.test)):
    m = d["log"]["M_ref64"].values[:n]
    axs[2].plot(t, m - m[0], label=nm)
axs[2].set_yscale("symlog", linthresh=1e-12)
axs[2].set_xlabel("time t")
axs[2].set_ylabel("mean density minus initial")
axs[2].set_title("mass drift")
axs[2].legend(fontsize=7)
for ax in axs:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"pair_{a.label}.png"), dpi=150)
print("\nWrote tables/pair_%s_*.csv and figures/pair_%s.png" % (a.label, a.label))
