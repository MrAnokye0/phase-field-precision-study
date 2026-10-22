"""wert_analysis.py -- compare a float32 and a float64 Wertheim run made with run_cfg.py.

Run from the precision-study folder:
  python study/wert_analysis.py --ref wert_f64 --test wert_f32 --label wertheim_euler
Energies are re-evaluated in float64 with the package's own free-energy functional
(snapshots of the float32 run are exactly float32 values, widened to float64).
"""
import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

from pfm.manager import SimulationManager

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
a = p.parse_args()


def load(name):
    d = os.path.join(RES, name)
    return dict(log=pd.read_csv(os.path.join(d, "log.csv")), fields=np.load(os.path.join(d, "fields.npy")),
                cfg=toml.load(os.path.join(d, "config.toml")))


R, T = load(a.ref), load(a.test)
n = min(len(R["log"]), len(T["log"]))
assert np.array_equal(R["log"]["step"].values[:n], T["log"]["step"].values[:n])
cfg = dict(R["cfg"])
cfg["float_type"] = "float64"
cfg["write_path"] = os.path.join(RES, "_scratch")
system = SimulationManager(cfg)._system


def e64(field):
    return float(system.average_free_energy(jnp.asarray(field[None, ...], dtype=jnp.float64)))


fr, ft = R["fields"][:n], T["fields"][:n]
Er = np.array([e64(f) for f in fr])
Et = np.array([e64(f) for f in ft])
step, t = R["log"]["step"].values[:n], R["log"]["t"].values[:n]
Mr, Mt = R["log"]["M_ref64"].values[:n], T["log"]["M_ref64"].values[:n]


def nrm(x):
    return float(np.linalg.norm(x))


rows = []
for i in range(n):
    r, s = fr[i].ravel(), ft[i].ravel()
    dr, ds = r - r.mean(), s - s.mean()
    row = dict(step=int(step[i]), t=t[i],
               relL2_field=nrm(s - r) / nrm(r),
               relL2_deviation=nrm(ds - dr) / nrm(dr),
               E64_ref=Er[i], E64_test=Et[i], dE_rel=(Et[i] - Er[i]) / abs(Er[i]),
               mass_drift_rel_ref=(Mr[i] - Mr[0]) / Mr[0], mass_drift_rel_test=(Mt[i] - Mt[0]) / Mt[0])
    if i > 0:
        u, v = r - fr[0].ravel(), s - ft[0].ravel()  # displacement from each run's own start
        row.update(disp_ratio=nrm(v) / nrm(u), disp_corr=float(np.corrcoef(u, v)[0, 1]),
                   Edrop_ratio=(Et[i] - Et[0]) / (Er[i] - Er[0]),
                   frozen_ref=float((fr[i] == fr[i - 1]).mean()), frozen_test=float((ft[i] == ft[i - 1]).mean()))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(TAB, f"{a.label}_comparison.csv"), index=False)

pd.options.display.float_format = lambda x: f"{x:.4g}"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False))
ro = lambda L, E: float(np.max(np.abs(L["log"]["E_inproc"].values[:n] - E) / np.abs(E)))
print("\nin-run energy readout error (max rel):  ref %.2e   test %.2e" % (ro(R, Er), ro(T, Et)))
print("energy increases (float64 re-evaluation): ref %d   test %d" % (int((np.diff(Er) > 1e-12 * abs(Er[1:])).sum()),
                                                                     int((np.diff(Et) > 1e-12 * abs(Et[1:])).sum())))
print("initial-data rounding: rel. L2 of (float32 start - float64 start) = %.3e" % (nrm(ft[0] - fr[0]) / nrm(fr[0])))

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
ax = axs[0, 0]
ax.plot(t, Er - Er[0], "o-", label="float64 run (re-evaluated)")
ax.plot(t, Et - Et[0], "s--", label="float32 run (re-evaluated in float64)")
ax.plot(t, T["log"]["E_inproc"].values[:n] - T["log"]["E_inproc"].values[0], "x:", label="float32 run (as computed in float32)")
ax.set_title("energy change since start")
ax.set_xlabel("time t")
ax.legend(fontsize=7)
ax = axs[0, 1]
ax.semilogy(t[1:], df["relL2_field"][1:], "o-", label="field")
ax.semilogy(t[1:], df["relL2_deviation"][1:], "s-", label="deviation from the mean")
ax.set_title("relative L2 difference, float32 vs float64")
ax.set_xlabel("time t")
ax.legend()
ax = axs[1, 0]
ax.plot(t[1:], df["disp_ratio"][1:], "o-", label="|displacement f32| / |displacement f64|")
ax.plot(t[1:], df["disp_corr"][1:], "s-", label="correlation of displacements")
ax.set_title("is float32 reproducing the dynamics?")
ax.set_xlabel("time t")
ax.legend(fontsize=7)
ax = axs[1, 1]
ax.plot(t, df["mass_drift_rel_ref"], "o-", label="float64")
ax.plot(t, df["mass_drift_rel_test"], "s-", label="float32")
ax.set_yscale("symlog", linthresh=1e-15)
ax.set_title("relative drift of the mean density")
ax.set_xlabel("time t")
ax.legend()
for ax in axs.ravel():
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, f"{a.label}.png"), dpi=150)
print("\nWrote tables/%s_comparison.csv and figures/%s.png" % (a.label, a.label))
