# MiniRes benchmark

Twin experiments on a small waterflood, simulated with [MiniRes](https://github.com/patnr/MiniRes),
to compare PET's update schemes, analyses and localizations on equal terms: the same cases, the
same truths, the same scores.

> [!WARNING]
> This is not (yet) a representative or authoritative benchmark. It has one reservoir, at two
> resolutions; few truths; and methods run at default or textbook settings, mostly untuned.
> Its results say how the methods fare on *this* case and *these* settings, and can change with
> either. Use it to catch regressions and to guide development, not to rank methods.

## Purpose

- **A score kept up to date.** `RESULTS.md` records how each method does, so that a change to PET's
  code can be checked against it.
- **Scores a misfit cannot give.** The truth is known, so an update is scored on the unknowns
  (the log-permeability field, and the residual oil saturation `sor`) and on a forecast that was
  not assimilated, as well as on how well it fits the data.

## Layout

| | |
|---|---|
| `cases.py` | The cases: the reservoir, its wells and schedule, the prior, the observations, and the twin experiments drawn from them. |
| `methods.py` | The methods compared, each a label and its PET settings (`[dataassim]`). |
| `scores.py` | What an ensemble is scored on. |
| `run.py` | Runs the methods on the cases, and generates `RESULTS.md`. |
| `results/` | The scores of each case, per method and truth (JSON), from which `RESULTS.md` is generated. |
| `RESULTS.md` | The tables. Generated; do not edit by hand. |

`tests/test_benchmark_minires.py` checks that each method still runs, on one tiny experiment.

## Running

Needs the `minires` extra (`pip install -e .[minires]`). From this directory:

```sh
python run.py --quick               # while developing: the coarse case, 5 truths, about a minute
python run.py                       # the official run: every method and case (over 10 minutes)
python run.py --method "ES-MDA"     # only the methods whose label contains this
```

A quick run (fewer truths than the case's own) prints its table and writes nothing. It is
printed next to the official scores on the same truths; the experiments are seeded, so any
difference is due to the code. Its numbers are tentative: with so few truths, the differences
between methods are often within the noise.

The official run writes `results/` and `RESULTS.md`. It is redone from time to time, not with
every change, so `RESULTS.md` may lag behind the methods and scores defined here.

## Adding a method or a score

- A method: an entry in `METHODS` of `methods.py`.
- A score: a key in the output of `scores.scores`, and a column in `COLUMNS` of `run.py`.
