"""run_sim.py -- one phase-field run with full-precision logging.

Run from the precision-study folder, e.g.
  python study/run_sim.py --name smoke_f32 --precision float32 --steps 20000 --log-every 1000

Outputs in study/results/<name>/: config.toml, meta.json, log.csv, fields.npy
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
INIT = os.path.join(REPO, "Examples", "landau", "standard_input.dat")


def energy_np(rho, eps, k, model, dx=1.0):
    """Independent float64 re-implementation of the package's average free energy (Landau)."""
    r = np.asarray(rho, dtype=np.float64)
    bulk = -0.5 * eps * r**2 + 0.25 * r**4
    gx = (np.roll(r, -1, axis=0) - r) / dx
    gy = (np.roll(r, -1, axis=1) - r) / dx
    coef = k if model == "ch" else 0.5 * k
    interface = coef * (gx**2 + gy**2)
    return float((bulk.sum() + interface.sum()) * dx**3 / r.size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--precision", choices=["float32", "float64"], required=True)
    p.add_argument("--integrator", choices=["euler", "spectral"], default="euler")
    p.add_argument("--model", choices=["ch", "ac"], default="ch")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--log-every", type=int, required=True)
    p.add_argument("--epsilon", type=float, default=0.9)
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--round-init-f32", action="store_true",
                   help="float64 run started from the float32-rounded initial field (control)")
    p.add_argument("--perturb", type=float, default=0.0,
                   help="relative gaussian perturbation of the initial field (control)")
    p.add_argument("--blowup-threshold", type=float, default=1e3)
    return p.parse_args()


def main():
    a = parse_args()
    outdir = os.path.join(a.out, a.name)
    os.makedirs(outdir, exist_ok=True)
    if a.steps % a.log_every != 0:
        sys.exit("--steps must be a multiple of --log-every")
    if a.round_init_f32 and a.precision != "float64":
        sys.exit("--round-init-f32 only makes sense with --precision float64")

    cfg = dict(
        steps=a.steps, print_every=a.log_every, print_trajectory_every=a.log_every,
        N=64, dt=a.dt, free_energy="landau", integrator=a.integrator, model=a.model,
        use_autodiff=False, interface_scalar=2.0, initial_density=0.01,
        float_type=a.precision, write_path=outdir, load_from=INIT, seed=0, verbose=False,
    )
    cfg["landau"] = {"epsilon": a.epsilon}
    with open(os.path.join(outdir, "config.toml"), "w") as f:
        toml.dump(cfg, f)

    # Precision must be fixed BEFORE pfm / jax.numpy are used (same order as pfm.main.run)
    import jax
    if a.precision == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from pfm.manager import SimulationManager

    mgr = SimulationManager(cfg)
    system = mgr._system
    rho = getattr(system, "init_" + system.field_name)

    if a.round_init_f32:
        rho = jnp.asarray(np.asarray(rho).astype(np.float32).astype(np.float64), dtype=rho.dtype)
    if a.perturb > 0:
        rng = np.random.default_rng(12345)
        base = np.asarray(rho, dtype=np.float64)
        base = base * (1.0 + a.perturb * rng.standard_normal(base.shape))
        rho = jnp.asarray(base, dtype=rho.dtype)

    k_int = cfg.get("k", 1.0)
    rows, fields, chunk_times = [], [], []

    def record(step, rho_now):
        field = np.asarray(rho_now, dtype=np.float64)[0]  # exact widening of the working dtype
        with np.errstate(all="ignore"):
            finite = bool(np.isfinite(field).all())
            e_in = float(system.average_free_energy(rho_now))
            m_in = float(system.average_mass(rho_now))
            e_ref = energy_np(field, a.epsilon, k_int, a.model)
            m_ref = math.fsum(field.ravel().tolist()) / field.size if finite else float("nan")
            rows.append(dict(step=step, t=step * a.dt, E_inproc=e_in, M_inproc=m_in,
                             E_ref64=e_ref, M_ref64=m_ref,
                             rho_min=float(field.min()), rho_max=float(field.max()),
                             finite=finite))
        fields.append(field)
        return finite and float(np.abs(field).max()) < a.blowup_threshold

    status = "ok"
    record(0, rho)
    n_chunks = a.steps // a.log_every
    progress_every = max(1, n_chunks // 20)
    t_start = time.perf_counter()
    steps_done = 0
    for c in range(1, n_chunks + 1):
        t0 = time.perf_counter()
        rho = mgr._evolve_n_steps(rho, a.log_every)
        rho.block_until_ready()
        chunk_times.append(time.perf_counter() - t0)
        steps_done = c * a.log_every
        healthy = record(steps_done, rho)
        if not healthy:
            status = f"blowup at step <= {steps_done}"
            break
        if c % progress_every == 0:
            r = rows[-1]
            print(f"step {steps_done:>9d}  E={r['E_inproc']:.10e}  max|rho|={max(abs(r['rho_min']), abs(r['rho_max'])):.4f}")
    wall = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "log.csv"), index=False, float_format="%.17g")
    np.save(os.path.join(outdir, "fields.npy"), np.stack(fields))

    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    meta = dict(
        name=a.name, status=status, steps_requested=a.steps, steps_done=steps_done,
        rho_dtype=str(rho.dtype), x64_enabled=bool(jax.config.jax_enable_x64),
        wall_s=wall, first_chunk_s=chunk_times[0] if chunk_times else None,
        median_chunk_s=float(np.median(chunk_times)) if chunk_times else None,
        euler_stability_term=2 * 1.0 * 1.0 * a.dt / 1.0**4,
        python=sys.version, jax=jax.__version__, numpy=np.__version__,
        devices=str(jax.devices()), repo_commit=commit, args=vars(a),
    )
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{a.name}] status={status} dtype={rho.dtype} steps_done={steps_done} wall={wall:.1f}s")


if __name__ == "__main__":
    main()

