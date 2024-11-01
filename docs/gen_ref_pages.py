"""Generate reference pages (md) from code (py).

Based on `https://mkdocstrings.github.io/recipes/`

Note that the generated markdown files have almost no content,
merely contain a reference to the corresponding `mkdocstring` identifier.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent

srcs = ["pipt", "popt", "simulator", "ensemble", "misc", "input_output", "tests"]
for s in srcs:
    pgk = root / s
    for path in sorted(pgk.rglob("*.py")):

        # Skip files that don't have docstrings (avoid build abortion)
        # More elaborate solution here: https://github.com/mkdocstrings/mkdocstrings/discussions/412
        txt = path.read_text()
        if txt and txt.splitlines()[0][0] not in ["'", '"']:
            continue

        parts = tuple(path.relative_to(pgk).with_suffix("").parts)
        path_md = Path("reference", s, path.relative_to(pgk).with_suffix(".md"))

        if parts[-1] == "__init__":
            parts = parts[:-1] or pgk.parts[-1:]
            if not parts:
                # we're in root pkg
                parts = pgk.parts[-1:]
            path_md = path_md.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        # PS: rm `mkdocs_gen_files` to get to inspect actual .md files
        # NB: will (over)write in docs/ folder.
        # path_md = Path("docs", path_md)
        # if not path_md.parent.exists():
        #     path_md.parent.mkdir(parents=True)
        with mkdocs_gen_files.open(path_md, "w") as fd:
            # Explicitly set the title to avoid mkdocs capitalizing
            # names and removing underscores (only applies to files)
            print(f"# {parts[-1]}", file=fd)

            identifier = ".".join(parts)
            print("::: " + identifier, file=fd)

        # mkdocs_gen_files.set_edit_path(path_md, ".." / path.relative_to(root))

# > So basically, you can use the literate-nav plugin just for its ability to
# > infer only sub-directories, without ever writing any actual "literate navs".
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
