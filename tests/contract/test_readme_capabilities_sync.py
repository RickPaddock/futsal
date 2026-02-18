from pathlib import Path

from src.utils.capabilities_report import (
    collect_capabilities,
    render_capabilities_markdown,
)


def test_readme_generated_capabilities_section_is_in_sync():
    repo_root = Path(__file__).resolve().parents[2]
    readme_path = repo_root / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    expected = render_capabilities_markdown(collect_capabilities(repo_root)).strip()
    assert expected in readme_text
