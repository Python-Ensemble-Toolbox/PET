import subprocess

import pytest
import shutil

@pytest.fixture(scope="session")
def temp_examples_dir(request, tmp_path_factory):
    """Clone PET Examples repo to a temp dir. Return its path."""
    pth = tmp_path_factory.mktemp("temp_dir")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/patnr/PET-Examples.git", pth],
                   check=True)
    yield pth
    shutil.rmtree(str(pth))
