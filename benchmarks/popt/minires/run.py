#!/usr/bin/env python3
"""Run the methods on the cases, and write the table in RESULTS.md.

    python run.py                       # every method, every case
    python run.py --method "EnSQP"      # the methods whose label contains this
    python run.py --quick               # a check while developing: a few cases

A run of fewer cases than the default is a check, printed and not written to results/:
its cases are the first of the official run's, so it is printed next to the official scores
on those same cases, and any difference is due to the code (the experiments are seeded).

Each (method, case) is one optimization, run in its own process and its own temporary
directory, so that what PET writes there (results, logs) goes nowhere. The scores of each are
kept in ``results/results.json``; RESULTS.md and convergence.png are generated from those.
A run of a subset of the methods replaces their entries and leaves the others'.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # the parallelism is over experiments, not within them

import argparse
import contextlib
import io
import json
import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from optimizers import COMMON, METHODS, REFERENCE
from popt import EnOpt, EnSQP, GaussianEnsemble
from problem import (
    CONTROLS,
    INTERVALS,
    RATE_MAX,
    RATE_START,
    Problem,
    evaluate,
    objective_alone,
    objective_with_constraints,
    objective_with_penalty,
)
from scoring import scores
from simulator.minires import MiniRes

HERE = Path(__file__).parent
MODEL_SEED, ENSEMBLE_SEED = 1000, 2000  # case i has the permeability drawn with MODEL_SEED + i
CASES = 10
QUICK = 3


class CountingMiniRes(MiniRes):
    """MiniRes, counting the simulations the method asks for."""

    runs = 0

    def run_fwd_sim(self, state, member_index=0, **kwargs):
        self.runs += 1
        return super().run_fwd_sim(state, member_index, **kwargs)


def experiment(method: dict, i: int, ne: int) -> dict:
    """One optimization: the scores of where it ended, the simulations spent, and its path."""
    problem = Problem(i, MODEL_SEED + i)
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.chdir(tmp),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        sim = CountingMiniRes(problem.simulator())
        referee = MiniRes(problem.simulator())  # scores the iterates, uncounted
        path = []  # (simulations so far, the schedule)

        if method["kind"] == "reference":
            x, message = reference(sim, method["options"], path)
        else:
            x, message = ensemble_method(sim, method, ne, ENSEMBLE_SEED + i, path)

        out = {"runs": sim.runs, "message": message, **scores(referee, x)}
        trace = [(runs, scores(referee, xk)) for runs, xk in path]
        out["trace"] = {
            "runs": [runs for runs, _ in trace],
            "npv": [s["npv"] for _, s in trace],
            "violation": [max(s["water"], s["capacity"]) for _, s in trace],
        }
        return out


def ensemble_method(sim, method: dict, ne: int, seed: int, path: list):
    """Run EnSQP or EnOpt on the popt ensemble; returns the final schedule and the stopping message."""
    kind, options = method["kind"], {**COMMON, **method["options"]}
    if kind == "ensqp":
        objective = objective_with_constraints
    else:
        objective = objective_with_penalty if "epf" in options else objective_alone
    ensemble = GaussianEnsemble(Problem.ensemble(ne, seed), sim, objective)
    x0, cov, bounds = ensemble.get_state(), ensemble.get_cov(), ensemble.get_bounds()

    def record(optimizer):
        path.append((sim.runs, np.array(optimizer.optimize_results["x"], dtype=float)))

    def callback(optimizer):
        optimizer._record_results()  # the result the callback sees is otherwise the previous iterate's
        record(optimizer)

    if kind == "ensqp":
        optimizer = EnSQP(x0, ensemble.function_and_constraints, jac=ensemble.gradient, args=(cov,), bounds=bounds,
                          con_jac=ensemble.constraint_jacobian, hess=ensemble.lagrangian_hessian,
                          callback=callback, **options)
    else:
        optimizer = EnOpt(x0, ensemble.function, jac=ensemble.gradient, args=(cov,), bounds=bounds,
                          callback=callback, **options)
    path.append((0, np.array(x0, dtype=float)))
    optimizer.run_optimization()
    return np.array(optimizer.optimize_results["x"], dtype=float), optimizer.optimize_results["message"]


def reference(sim, options: dict, path: list):
    """SLSQP in the unit cube with finite-difference derivatives; returns the final schedule and message."""
    n = len(CONTROLS) * INTERVALS
    cache = {}

    def at(u):
        key = u.tobytes()
        if key not in cache:
            cache[key] = evaluate(sim, u * RATE_MAX)
        return cache[key]

    start = np.full(n, RATE_START / RATE_MAX)
    path.append((0, start * RATE_MAX))
    res = minimize(
        lambda u: at(u)[0], start, method="SLSQP", bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "ineq", "fun": lambda u: at(u)[1]}],
        callback=lambda u: path.append((sim.runs, np.array(u) * RATE_MAX)), options=options,
    )
    return res.x * RATE_MAX, res.message


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
COLUMNS = {
    "npv": "NPV [1e7 USD]",
    "gap": "NPV − reference",
    "water": "water-cap excess",
    "capacity": "capacity excess",
    "feasible": "feasible runs",
    "runs": "simulations",
}


def cell(values) -> str:
    """Mean, and its standard error over the cases."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.3g}"
    return f"{values.mean():.3g} ± {values.std(ddof=1) / np.sqrt(len(values)):.2g}"


def render_table(title: str, res: dict) -> list:
    meta, methods = res["meta"], res["methods"]
    lines = [
        "",
        title,
        "",
        f"{meta['cases']} cases, {meta['ne']} members for {len(CONTROLS) * INTERVALS} controls. "
        f"PET at `{meta['commit']}`, {meta['date']}.",
        "",
        "| | " + " | ".join(COLUMNS.values()) + " |",
        "|---" * (len(COLUMNS) + 1) + "|",
    ]
    ref = methods.get(REFERENCE)
    for label, r in methods.items():
        row = []
        for key in COLUMNS:
            if key == "gap":
                row.append(cell(np.array(r["npv"]) - np.array(ref["npv"])) if ref else "")
            elif key == "feasible":
                row.append(f"{int(np.sum(r['feasible']))} of {len(r['feasible'])}")
            elif key == "runs":
                row.append(f"{np.mean(r['runs']):.0f}")
            else:
                row.append(cell(r[key]))
        lines += [f"| {label} | " + " | ".join(row) + " |"]
    return lines


def render(results: dict) -> str:
    lines = [
        "# popt on MiniRes: results",
        "",
        "Generated by `run.py`; do not edit by hand. See `problem.py` for the problem, `optimizers.py` for the",
        "methods and their settings, and `scoring.py` for what is scored. Each cell is the mean over the",
        "cases (permeability fields), ± its standard error. The net present value of a run that violates a",
        "constraint is not comparable with that of one that does not: read it together with the excess",
        "columns, which give the largest violation as a fraction of the limit. The reference is not an",
        "ensemble method: it differentiates the simulator by finite differences.",
    ]
    lines += render_table("## Injection rates for net present value, under a water-production cap and an injection capacity", results)
    lines += [
        "",
        "![NPV below the reference and constraint excess against simulations, all cases](convergence.png)",
        "",
        "Thin lines are the cases, the thick line their mean. A run that has stopped is held at where it",
        "ended, so each mean line ends, at the simulations of the method's longest run, on the table's",
        "\"NPV − reference\".",
    ]
    return "\n".join(lines) + "\n"


def first_cases(res: dict, cases: int, labels) -> dict:
    """The official results ``res``, of the methods ``labels``, cut down to their first ``cases``."""
    methods = {
        label: {k: v[:cases] for k, v in r.items() if k not in ("trace", "traces")}
        for label, r in res["methods"].items() if label in labels
    }
    return {"meta": {**res["meta"], "cases": cases}, "methods": methods}


def commit() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()

    # The last commit of PET's code, which is what the numbers depend on (not of the benchmark itself)
    src = str(HERE.parents[2] / "src")
    return git("log", "-1", "--format=%h", "--", src) + ("-dirty" if git("status", "--porcelain", "--", src) else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", help="label substring; default: all")
    parser.add_argument("--cases", type=int, help=f"default: {CASES}")
    parser.add_argument("--ne", type=int, default=10)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true", help=f"default to {QUICK} cases")
    args = parser.parse_args()
    if args.quick:
        args.cases = args.cases or QUICK

    methods = [m for m in METHODS if not args.method or any(s in m["label"] for s in args.method)]
    cases = args.cases or CASES
    quick = cases < CASES

    jobs = [(m, i) for m in methods for i in range(cases)]
    print(f"{len(methods)} method(s) x {cases} cases ...", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=logging.disable, initargs=(logging.WARNING,)) as pool:
        done = list(pool.map(experiment, *zip(*[(m, i, args.ne) for m, i in jobs])))

    res_file = HERE / "results" / "results.json"
    meta = {"cases": cases, "ne": args.ne, "commit": commit(), "date": str(date.today())}
    old = json.loads(res_file.read_text()) if res_file.exists() else None
    compatible = old and (old["meta"]["cases"], old["meta"]["ne"]) == (cases, args.ne)
    results = {"meta": meta, "methods": dict(old["methods"]) if compatible else {}}
    for m in methods:
        mine = [d for (mm, _), d in zip(jobs, done) if mm is m]
        results["methods"][m["label"]] = {
            **{k: [d[k] for d in mine] for k in ("runs", "npv", "water", "capacity", "feasible", "message")},
            "traces": [d["trace"] for d in mine],  # the path on every case, for the figure
        }
    results["methods"] = {m["label"]: results["methods"][m["label"]] for m in METHODS if m["label"] in results["methods"]}

    if quick:
        print("\n".join(render_table("## This run", results)))
        if old and old["meta"]["ne"] == args.ne and old["meta"]["cases"] >= cases:
            official = first_cases(old, cases, results["methods"])
            if official["methods"]:
                print("\n".join(render_table("## Official, on the same cases", official)))
        print()
        return

    res_file.parent.mkdir(exist_ok=True)
    res_file.write_text(json.dumps(results, indent=1) + "\n")
    (HERE / "RESULTS.md").write_text(render(results))
    from figure import convergence

    convergence(results, HERE / "convergence.png")
    print("Wrote RESULTS.md and convergence.png")


if __name__ == "__main__":
    main()
