## Documentation generation

Many python libraries use `sphinx` to generate docs via `readthedocs` (also hosts).
This setup is too powerful (and therefore complicated) for our purposes.
Instead, we use [pdoc](https://github.com/mitmproxy/pdoc), run via **GitHub Actions**,
as configured [here](./.github/workflows/deploy-docs.yml).
The resulting `html` is hosted with **Github Pages**.

`pdoc` grabs the docstrings of modules/classes/functions,
and renders them into pretty html.
The docstrings should be written using markdown syntax,
[with some extras (cross-referencing, math, etc)](https://pdoc.dev/docs/pdoc.html#what-is-pdoc)

### Run pdoc locally

To *live preview* your changes, do

```sh
pdoc -t docs/templates --math pipt popt misc ensemble simulator input_output docs/dev_guide.py
```

## Tests

Run the tests by the command

```sh
pytest
```

It will discover all [appropriately named tests](https://docs.pytest.org)
in the source (see the `tests` dir).

Use (for example) `pytest --doctest-modules docs/dev_guide.py` to
*also* run any example code **within** docstrings.

The CI also simply runs `pytest`.
