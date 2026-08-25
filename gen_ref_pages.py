"""Generate reference pages (md) from code (py).

Based on `https://mkdocstrings.github.io/recipes/`

Note that the generated markdown files have almost no content,
merely contain a reference to the corresponding `mkdocstring` identifier.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent

src = root
for path in sorted(src.rglob("*.py")):

    # Skip "venv" and other similarly named directories
    if path.name.startswith("venv") or path.name.startswith(".venv") or path.name.startswith(".") or "site-packages" in path.parts:
        continue

    # Skip and print "cause"
    cause = None
    # Skip if read issue
    try:
        txt = path.read_text()
    except UnicodeDecodeError:
        cause = f"Warning: Skipping (due to read error) {path.relative_to(root)}"
    # Skip files that don't have docstrings (to avoid build abortion)
    # More elaborate solution: https://github.com/mkdocstrings/mkdocstrings/discussions/412
    if txt and (lines := txt.splitlines()):
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith(('"', "'")):
                cause = f"Warning: Skipping (due to missing docstring) {path.relative_to(root)}"
            break
    # namespace packages not unsupported.
    # Fix? https://github.com/mkdocstrings/mkdocstrings/discussions/563
    if path.name != "__init__.py":
        parent_has_init = (path.parent / "__init__.py").exists()
        if not parent_has_init:
            cause = f"Warning: Skipping namespace package file: {path.relative_to(root)}"
    if cause:
        print(cause)
        continue

    parts = tuple(path.relative_to(src).with_suffix("").parts)
    path_md = Path("reference", path.relative_to(src).with_suffix(".md"))

    if parts[-1] == "__init__":
        # Generate index.md
        parts = parts[:-1] # name of parent dir
        path_md = path_md.with_name("index.md")
    elif parts[0] == "docs":
        continue

    # PS: Uncomment (replace `mkdocs_gen_files.open`) to view actual .md files
    # path_md = Path("docs", path_md)
    # if not path_md.parent.exists():
    #     path_md.parent.mkdir(parents=True)
    # with open(path_md, "w") as fd:

    with mkdocs_gen_files.open(path_md, "w") as fd:
        # Explicitly set the title to avoid mkdocs capitalizing
        # names and removing underscores (only applies to files)
        print(f"# {parts[-1]}", file=fd)

        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(path_md, ".." / path.relative_to(root))

# > So basically, you can use the literate-nav plugin just for its ability to
# > infer only sub-directories, without ever writing any actual "literate navs".
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    # nav_file.writelines(nav.build_literate_nav())
    nav_file.writelines(
        "# Code reference\nUse links in sidebar to navigate the code docstrings.\n"
        + "".join(list(nav.build_literate_nav()))
    )
