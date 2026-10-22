"""run_cfg.py -- run any example TOML of the repo with chosen precision / length, saving snapshots.

Run from the precision-study folder, e.g.
  python study/run_cfg.py --base-config "phase-field-models/Examples/simple wertheim/euler_noautodiff.toml" --name wert_f32 --precision float32 --steps 2000 --log-every 500

Only steps, precision, logging interval, output path and the (absolute) initial-data path are overridden.
Outputs in study/results/<name>/: config.toml, meta.json, log.csv, fields.npy
(the in-run energy is computed in the working precision; re-evaluate it in float64 afterwards).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "phase-field-models"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--precision", choices=["float32", "float64"], required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--log-every", type=int, required=True)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--dt", type=float, default=None, help="override dt (default: keep the config's)")
    p.add_argument("--blowup-threshold", type=float, default=1e30)
    return p.parse_args()


def main():
    a = parse_args()
    if a.steps % a.log_every != 0:
        sys.exit("--steps must be a multiple of --log-every")
    outdir = os.path.join(a.out, a.name)
    os.makedirs(outdir, exist_ok=True)

    cfg = toml.load(a.base_config)
    data_file = os.path.join(os.path.dirname(os.path.abspath(a.base_config)), os.path.basename(cfg["load_from"]))
    if not os.path.exists(data_file):
        sys.exit("initial-data file not found next to the config: " + data_file)
    cfg.update(steps=a.steps, print_every=a.log_every, print_trajectory_every=a.log_every,
               float_type=a.precision, write_path=outdir, load_from=data_file, seed=0, verbose=False)
    cfg.pop("use_CUDA", None)
    cfg.setdefault("integrator", "euler")
    if a.dt is not None:
        cfg["dt"] = a.dt
    with open(os.path.join(outdir, "config.toml"), "w") as f:
        toml.dump(cfg, f)

    import jax
    if a.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from pfm.manager import SimulationManager

    mgr = SimulationManager(cfg)
    system = mgr._system
    rho = getattr(system, "init_" + system.field_name)
    dt = cfg["dt"]
    rows, fields, chunk_times = [], [], []

    def record(step, rho_now):
        field = np.asarray(rho_now, dtype=np.float64)[0]
        with np.errstate(all="ignore"):
            finite = bool(np.isfinite(field).all())
            rows.append(dict(step=step, t=step * dt,
                             E_inproc=float(system.average_free_energy(rho_now)),
                             M_inproc=float(system.average_mass(rho_now)),
                             M_ref64=math.fsum(field.ravel().tolist()) / field.size if finite else float("nan"),
                             rho_min=float(field.min()), rho_max=float(field.max()), finite=finite))
        fields.append(field)
        return finite and float(np.abs(field).max()) < a.blowup_threshold

    status = "ok"
    record(0, rho)
    n_chunks = a.steps // a.log_every
    t_start = time.perf_counter()
    steps_done = 0
    for c in range(1, n_chunks + 1):
        t0 = time.perf_counter()
        rho = mgr._evolve_n_steps(rho, a.log_every)
        rho.block_until_ready()
        chunk_times.append(time.perf_counter() - t0)
        steps_done = c * a.log_every
        healthy = record(steps_done, rho)
        r = rows[-1]
        print(f"step {steps_done:>9d}  E={r['E_inproc']:.10e}  mean={r['M_ref64']:.12e}  "
              f"min={r['rho_min']:.4e} max={r['rho_max']:.4e}  chunk={chunk_times[-1]:.1f}s")
        if not healthy:
            status = f"blowup at step <= {steps_done}"
            break
    wall = time.perf_counter() - t_start

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "log.csv"), index=False, float_format="%.17g")
    np.save(os.path.join(outdir, "fields.npy"), np.stack(fields))
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(name=a.name, status=status, steps_requested=a.steps, steps_done=steps_done,
                rho_dtype=str(rho.dtype), x64_enabled=bool(jax.config.jax_enable_x64), wall_s=wall,
                first_chunk_s=chunk_times[0] if chunk_times else None,
                median_chunk_s=float(np.median(chunk_times)) if chunk_times else None,
                python=sys.version, jax=jax.__version__, numpy=np.__version__, devices=str(jax.devices()),
                repo_commit=commit, base_config=a.base_config, args=vars(a))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{a.name}] status={status} dtype={rho.dtype} steps_done={steps_done} wall={wall:.1f}s")


if __name__ == "__main__":
    main()
