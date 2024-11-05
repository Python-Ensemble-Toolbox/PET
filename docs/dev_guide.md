# Developer guide

## Writing documentation

The documentation is built with `mkdocs`.

- It should be written in [the syntax of markdown](https://www.markdownguide.org/cheat-sheet/).
- The syntax is further augmented by [several pymdown plugins](https://squidfunk.github.io/mkdocs-material/reference/).
- **Docstrings** are processed as above, but should also
  declare parameters and return values in the [style of numpy](https://mkdocstrings.github.io/griffe/reference/docstrings/#numpydoc-style),
  and `>>>` markers must follow the "Examples" section.

!!! note
    You can preview the rendered html docs by running
    ```sh
    mkdocs serve
    ```

    - Temporarily disable `mkdocs-jupyter` in `mkdocs.yml` to speed up build reloads.
    - Set `validation: unrecognized_links: warn` to get warnings about linking issues.

A summary of how to add cross-reference links is given below.

### Linking to pages

You should use relative page links, including the `.md` extension.
For example, `[link label](sibling-page.md)`.

The following works, but does not get validated! `[link label](../sibling-page)`

!!! hint "Why not absolute links?"

    The downside of relative links is that if you move/rename source **or** destination,
    then they will need to be changed, whereas only the destination needs be watched
    when using absolute links.

    Previously, absolute links were not officially supported by MkDocs, meaning "not modified at all".
    Thus, if made like so `[label](/PET/references)`,
    i.e. without `.md` and including `/PET`,
    then they would **work** (locally with `mkdocs serve` and with GitHub hosting).
    Since [#3485](https://github.com/mkdocs/mkdocs/pull/3485) you can instead use `[label](/references)`
    i.e. omitting `PET` (or whatever domain sub-dir is applied in `site_url`)
    by setting `mkdocs.yml: validation: absolute_links: relative_to_docs`.
    A different workaround is the [`mkdocs-site-url` plugin](https://github.com/OctoPrint/mkdocs-site-urls).

    !!! tip "Either way"
        It will not be link that your editor can follow to the relevant markdown file
        (unless you create a symlink in your file system root?)
        nor will GitHub's internal markdown rendering manage to make sense of it,
        so my advise is not to use absolute links.

### Linking to headers/anchors

Thanks to the `autorefs` plugin,
links to **headings** (including page titles) don't even require specifying the page path!
Syntax: `[visible label][link]` i.e. double pairs of _brackets_. Shorthand: `[link][]`.
!!! info
    - Clearly, non-unique headings risk being confused with others in this way.
    - The link (anchor) must be lowercase!

This facilitates linking to

- **API (code reference)** items.
  For example, ``[`da_methods.ensemble`][]``,
  where the backticks are optional (makes the link _look_ like a code reference).
- **References**. For example ``[`bocquet2016`][]``,

### Docstring injection

Use the following syntax to inject the docstring of a code object.

```markdown
::: da_methods.ensemble
```

But we generally don't do so manually.
Instead it's taken care of by the reference generation via `docs/gen_ref_pages.py`.

### Including other files

The `pymdown` extension ["snippets"](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippets-notation)
enables the following syntax to include text from other files.

`--8<-- "/path/from/project/root/filename.ext"`

### Adding to the examples

Example scripts are very useful, and contributions are very desirable.  As well
as showcasing some feature, new examples should make sure to reproduce some
published literature results.  After making the example, consider converting
the script to the Jupyter notebook format (or vice versa) so that the example
can be run on Colab without users needing to install anything (see
`docs/examples/README.md`). This should be done using the `jupytext` plug-in (with
the `lightscript` format), so that the paired files can be kept in synch.

### Bibliography

In order to add new references,
insert their bibtex into `docs/bib/refs.bib`,
then run `docs/bib/bib2md.py`
which will format and add entries to `docs/references.md`
that can be cited with regular cross-reference syntax, e.g. `[bocquet2010a][]`.

### Hosting

The above command is run by a GitHub Actions workflow whenever
the `master` branch gets updated.
The `gh-pages` branch is no longer being used.
Instead [actions/deploy-pages](https://github.com/actions/deploy-pages)
creates an artefact that is deployed to Github Pages.


### Run pdoc locally

To *live preview* your changes, do

```sh
pdoc -t docs/templates --docformat=numpy --math pipt popt misc ensemble simulator input_output docs/dev_guide.py docs/tutorials.py
```

This should open a browser window with the rendered html.
You can also ctrl/cmd-click the printed localhost link, or simply copy-paste it into your browser.

If you want to reproduce errors that occur in **CI**, you'll want to include the option `-o docs-generated `.
Since this actually generates html *files*, it will processes **all** of the files by default
(without which you might not pick up on the error).

.. note:: PS: it seems that the upstream `pdoc` does not report where parsing errors occur
  (it simply quits with a traceback).
  We therefore use my (`patnr`) fork which

  - skips the markdown conversion for the erroneous docstring,
  - prints the specific docstring that causes issues.

## Tests

The test suite is orchestrated using `pytest`. Both in **CI** and locally.
I.e. you can run the tests simply by the command

```sh
pytest
```

It will discover all [appropriately named tests](https://docs.pytest.org)
in the source (see the `tests` dir).

Use (for example) `pytest --doctest-modules some_file.py` to
*also* run any example code **within** docstrings.

We should also soon make use of a config file (for example `pyproject.toml`) for `pytest`.
