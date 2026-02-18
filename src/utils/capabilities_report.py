from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


START_MARKER = "<!-- IMPLEMENTED_CAPABILITIES_START -->"
END_MARKER = "<!-- IMPLEMENTED_CAPABILITIES_END -->"


def _extract_set_literal(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, set):
                        return sorted(str(v) for v in value)
    return []


def _extract_pass_choices(tree: ast.AST) -> list[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and arg0.value == "--pass":
                for kw in node.keywords:
                    if kw.arg == "choices":
                        value = ast.literal_eval(kw.value)
                        if isinstance(value, list):
                            return [int(v) for v in value]
    return []


def _extract_default_all_passes(tree: ast.AST) -> list[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "passes_to_run":
                    if isinstance(node.value, ast.List):
                        values = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                                values.append(elt.value)
                        if values:
                            return values
    return []


def collect_capabilities(repo_root: Path) -> dict[str, Any]:
    main_path = repo_root / "src" / "main.py"
    main_src = main_path.read_text(encoding="utf-8")
    tree = ast.parse(main_src)

    video_allowed = _extract_set_literal(tree, "ALLOWED_VIDEO_OUTPUT_PASSES")
    video_implemented = _extract_set_literal(tree, "IMPLEMENTED_VIDEO_OUTPUT_PASSES")
    pass_choices = _extract_pass_choices(tree)
    default_passes = _extract_default_all_passes(tree)

    skills_dir = repo_root / "src" / "skills"
    skills = sorted(
        p.stem for p in skills_dir.glob("*.py") if p.name not in {"__init__.py"}
    )

    validation_dir = repo_root / "src" / "validation"
    validation_modules = sorted(
        p.stem
        for p in validation_dir.glob("*_rules.py")
    )

    return {
        "pass_choices": pass_choices,
        "default_passes": default_passes,
        "video_allowed": video_allowed,
        "video_implemented": video_implemented,
        "skills": skills,
        "validation_modules": validation_modules,
    }


def render_capabilities_markdown(data: dict[str, Any]) -> str:
    pass_choices = ", ".join(str(v) for v in data["pass_choices"]) or "none"
    default_passes = ", ".join(str(v) for v in data["default_passes"]) or "none"
    video_allowed = ", ".join(data["video_allowed"]) or "none"
    video_implemented = ", ".join(data["video_implemented"]) or "none"

    skills_lines = "\n".join(f"- {name}" for name in data["skills"]) or "- none"
    validation_lines = "\n".join(f"- {name}" for name in data["validation_modules"]) or "- none"

    return (
        f"{START_MARKER}\n"
        "## Implemented Capabilities (Auto-Generated)\n\n"
        "This section is generated from code. Do not edit manually.\n\n"
        f"- CLI `--pass` choices: {pass_choices}\n"
        f"- Default passes run when `--pass` is omitted: {default_passes}\n"
        f"- `--video-output` allowed keys: {video_allowed}\n"
        f"- `--video-output` implemented keys: {video_implemented}\n\n"
        "### Implemented skill modules (`src/skills`)\n"
        f"{skills_lines}\n\n"
        "### Validation rule modules (`src/validation/*_rules.py`)\n"
        f"{validation_lines}\n"
        f"{END_MARKER}"
    )


def update_readme_capabilities(readme_path: Path, section: str) -> None:
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# Futsal Tracking System\n\n"

    if START_MARKER in content and END_MARKER in content:
        start = content.index(START_MARKER)
        end = content.index(END_MARKER) + len(END_MARKER)
        content = content[:start] + section + content[end:]
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + section + "\n"

    readme_path.write_text(content, encoding="utf-8")
