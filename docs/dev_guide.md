## Documentation

Many python libraries use `sphinx` to generate docs via `readthedocs` (also hosts).
This setup is too powerful (and therefore complicated) for our purposes.
Instead, we use [pdoc](https://github.com/mitmproxy/pdoc), run via **GitHub Actions**,
as configured [here](./.github/workflows/deploy-docs.yml).
The resulting `html` is hosted with **Github Pages**.

`pdoc` grabs the docstrings of modules/classes/functions,
and renders them into pretty html.
The docstrings should be written using markdown syntax.

In general, you should also try to [reference other objects](https://pdoc.dev/docs/pdoc.html#link-to-other-identifiers)
(if appropriate) by using backticks.
And if you want to do it really well, you should follow
the [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html#sections).

### Run pdoc locally

To *live preview* your changes, do

```sh
pdoc -t docs/templates --docformat=numpy --math pipt popt misc ensemble simulator input_output docs/dev_guide.py
```

## Tests

Th test suite is orchestrated using `pytest`. Both in **CI** and locally.
I.e. you can run the tests simply by the command

```sh
pytest
```

It will discover all [appropriately named tests](https://docs.pytest.org)
in the source (see the `tests` dir).

Use (for example) `pytest --doctest-modules some_file.py` to
*also* run any example code **within** docstrings.

We should also soon make use of a config file (for example `pyproject.toml`) for `pytest`.
