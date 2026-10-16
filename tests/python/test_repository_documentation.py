"""Repository-level guards for public documentation links."""

from pathlib import Path
import re

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _public_markdown_files() -> list[Path]:
    """Return the public Markdown surfaces checked by CI."""

    return [
        _PROJECT_ROOT / "README.md",
        *sorted((_PROJECT_ROOT / "docs").glob("*.md")),
        _PROJECT_ROOT / "reports" / "README.md",
        _PROJECT_ROOT / "data" / "README.md",
        _PROJECT_ROOT / "cpp" / "README.md",
    ]


@pytest.mark.parametrize(
    "markdown_path",
    _public_markdown_files(),
    ids=lambda path: str(path.relative_to(_PROJECT_ROOT)),
)
def test_public_markdown_local_links_exist(
    markdown_path: Path,
) -> None:
    """Every repository-relative Markdown link resolves to a real path."""

    text = markdown_path.read_text(encoding="utf-8")
    missing_targets = []

    for raw_target in _MARKDOWN_LINK.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0]

        if (
            not target
            or target.startswith(("#", "http://", "https://", "mailto:"))
        ):
            continue

        relative_target = target.split("#", maxsplit=1)[0]
        resolved_target = (markdown_path.parent / relative_target).resolve()

        if not resolved_target.exists():
            missing_targets.append(target)

    assert not missing_targets, (
        f"{markdown_path.relative_to(_PROJECT_ROOT)} contains missing "
        f"local target(s): {missing_targets}"
    )
