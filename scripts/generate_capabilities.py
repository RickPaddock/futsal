from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.capabilities_report import (
    collect_capabilities,
    render_capabilities_markdown,
    update_readme_capabilities,
)


def main() -> int:
    repo_root = REPO_ROOT
    readme_path = repo_root / "README.md"

    capabilities = collect_capabilities(repo_root)
    section = render_capabilities_markdown(capabilities)
    update_readme_capabilities(readme_path, section)

    print(f"Updated generated capabilities section in {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
