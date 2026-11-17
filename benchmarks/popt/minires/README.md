# popt on MiniRes: constrained optimization

A constrained production-optimization problem on a small waterflood, simulated with
[MiniRes](https://github.com/patnr/MiniRes), to compare how popt's optimizers handle
constraints on equal terms: the same cases, the same start, the same scores. It was set up to
compare `EnSQP` (ensemble-based sequential quadratic programming) with `EnOpt` under the
exterior penalty function (EPF).

> [!WARNING]
> This is not a representative or authoritative benchmark. It has one reservoir and one set of
> limits; few cases; and methods run at plausible settings, not tuned ones. Its results say how
> the methods fare on *this* problem and *these* settings, and can change with either. Use it to
> catch regressions and to guide development, not to rank methods.

## The problem

The reservoir is that of `benchmarks/pipt/minires` with the well pattern turned round: a
1000 x 1000 x 20 m box of 25 x 25 cells, four injectors towards the corners on rate, one
producer at the centre on bottom-hole pressure, 40 steps of 60 days. The permeability is known,
one heterogeneous field per case, so the problem is deterministic.

The controls are the injectors' rates $q_{w,k}$ in each of 8 intervals of 300 days, 32
numbers, each between 0 and 500 m³/day. They maximise the net present value

$$ \mathrm{NPV} = \sum_{k=1}^{8} \frac{p_o V^o_k - c_w V^w_k - c_i V^i_k}{(1 + d)^{t_k}} $$

of the oil $V^o_k$ produced in interval $k$, less the cost of the water produced, $V^w_k$, and
injected, $V^i_k$ (500, 60 and 20 USD/m³, discounted at 8 % a year), subject to

- **a cap on the water production**, $V^w_k / 300\,\text{days} \le 300$ m³/day in every
  interval: 8 nonlinear constraints, known only through the simulation;
- **a capacity for the injection**, $\sum_w q_{w,k} \le 800$ m³/day in every interval:
  8 constraints that are linear in the controls.

Every injector starts at 150 m³/day throughout, which meets the capacity and violates the
water cap in the last intervals. At the reference solution the capacity binds early and the
water cap late.

`EnSQP` is given the constraints as such, returned by the objective next to its value. `EnOpt`
is given the objective plus the penalty $\tfrac{r}{2} \sum \max(-c, 0)^2$ on the scaled
violations, starting at $r = 10$, with $r$ multiplied by 4 in each of up to 6 outer passes. A step
that fails ends such a run, whatever the violation; the "advance on failure" row sets
`advance_on_failure` in `epf`, so that a failed step raises $r$ and the run goes on.

## Layout

| | |
|---|---|
| `problem.py` | The problem: the reservoir, its wells and controls, the economics, the constraints, and the objective functions handed to the optimizers. |
| `optimizers.py` | The methods compared, each a label and its settings. |
| `scoring.py` | What a run is scored on. |
| `run.py` | Runs the methods on the cases, and generates `RESULTS.md` and `convergence.png`. |
| `figure.py` | The convergence figure. |
| `results/` | The scores per method and case (JSON), from which `RESULTS.md` is generated. |
| `RESULTS.md` | The table. Generated; do not edit by hand. |

`tests/test_benchmark_popt_minires.py` checks that each method still runs, on one tiny
optimization. The simulator's well controls, which this benchmark relies on, are tested in
`tests/test_minires_controls.py`.

## Running

Needs the `minires` extra (`pip install -e .[minires]`) and matplotlib. From this directory:

```sh
python run.py --quick               # while developing: 3 cases
python run.py                       # the official run: every method on 10 cases
python run.py --method "EnSQP"      # only the methods whose label contains this
python run.py --ne 40               # more ensemble members than controls
```

One simulation takes about 0.15 s. An ensemble method spends a few hundred of them per case,
the finite-difference reference a few thousand, so the reference dominates the run time; leave
it out with `--method En` while developing. The jobs run in parallel, one per core.

A quick run (fewer cases than the official one) prints its table and writes nothing. It is
printed next to the official scores on the same cases; the experiments are seeded, so any
difference is due to the code.

## Reading the table

- The net present value of a run that violates a constraint is not comparable with that of a
  run that does not; read it with the two excess columns, which give the largest violation as
  a fraction of the limit (0.05 is 5 % over).
- "Feasible" means both excesses are below 0.1 % of the limit (`scoring.FEASIBLE`).
- The reference is scipy's SLSQP with finite-difference derivatives of the simulator. It is
  the best solution known for each case, not a proven optimum, and not an ensemble method.
- With 10 members for 32 controls (the default) each ensemble spans a third of the control
  space, which is the usual situation in practice. `--ne 40` gives the other regime.

## Adding a method or a score

- A method: an entry in `METHODS` of `optimizers.py` (and a colour in `figure.COLORS` to have it in the figure).
- A score: a key in the output of `scoring.scores`, and a column in `COLUMNS` of `run.py`.
