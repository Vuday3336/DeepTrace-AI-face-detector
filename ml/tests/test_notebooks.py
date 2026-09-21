"""The generated Kaggle notebooks must be valid Python.

Cell sources are built from strings in notebooks/build_notebooks.py, so a stray newline inside a
string literal produces a notebook that only fails hours later on Kaggle. Parse every cell here.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ML_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "00_generate_diffusion_faces",
    "01_data_prep_and_audit",
    "02_train_and_evaluate",
    "03_robustness_explain_export",
]


def strip_shell_lines(source: str) -> str:
    """Replace IPython `!command` lines (and their backslash continuations) with `pass`."""
    lines: list[str] = []
    skipping = False
    for line in source.splitlines():
        if skipping:
            skipping = line.rstrip().endswith("\\")
            continue
        match = re.match(r"^(\s*)!", line)
        if match:
            lines.append(match.group(1) + "pass")
            skipping = line.rstrip().endswith("\\")
        else:
            lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_parses(name: str) -> None:
    path = ML_DIR / "notebooks" / f"{name}.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells, f"{name} has no code cells"
    for index, cell in enumerate(code_cells):
        try:
            ast.parse(strip_shell_lines(cell.source))
        except SyntaxError as exc:
            pytest.fail(f"{name} code cell {index} has a syntax error: {exc}")


def test_notebooks_match_their_builder(tmp_path: Path) -> None:
    """Regenerating must not change the committed notebooks (builder and output stay in sync)."""
    before = {n: (ML_DIR / "notebooks" / f"{n}.ipynb").read_bytes() for n in NOTEBOOKS}
    subprocess.run([sys.executable, str(ML_DIR / "notebooks/build_notebooks.py")], check=True, capture_output=True)
    for name, content in before.items():
        assert (ML_DIR / "notebooks" / f"{name}.ipynb").read_bytes() == content, (
            f"{name}.ipynb is out of date — run notebooks/build_notebooks.py and commit the result"
        )
