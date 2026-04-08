#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    unity_root = repo_root / "animal-ai-unity"
    out_file = repo_root / "unity_codebase_structure.txt"

    include_dirs = [
        unity_root / "Assets" / "Scripts",
        unity_root / "Assets" / "Prefabs",
        unity_root / "Assets" / "Scenes",
        unity_root / "Assets" / "ML-Agents",
        unity_root / "Assets" / "Resources" / "test_configs",
    ]

    include_files = [
        unity_root / "Packages" / "manifest.json",
        unity_root / "Packages" / "packages-lock.json",
        unity_root / "ProjectSettings" / "ProjectSettings.asset",
        unity_root / "ProjectSettings" / "EditorBuildSettings.asset",
        unity_root / "ProjectSettings" / "ProjectVersion.txt",
    ]

    exclude_dir_names = {
        "Library",
        "Logs",
        "Log",
        "Temp",
        "obj",
        "Build",
        "build",
        "Builds",
        "Cache",
        "Caches",
        "UserSettings",
        ".vs",
        ".vscode",
    }
    exclude_suffixes = {".meta", ".tmp", ".log", ".pidb", ".booproj", ".svd", ".user"}

    def should_skip(path: Path) -> bool:
        if any(part in exclude_dir_names for part in path.parts):
            return True
        if path.name.endswith(tuple(exclude_suffixes)):
            return True
        return False

    def list_tree(base: Path):
        lines = []

        def walk(current: Path, prefix: str = ""):
            entries = sorted(
                [p for p in current.iterdir() if not should_skip(p)],
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                branch = "└── " if is_last else "├── "
                name = entry.name + ("/" if entry.is_dir() else "")
                lines.append(f"{prefix}{branch}{name}")
                if entry.is_dir():
                    walk(entry, prefix + ("    " if is_last else "│   "))

        lines.append(f"{base.relative_to(unity_root)}/")
        if base.exists() and base.is_dir():
            walk(base)
        else:
            lines.append("└── (missing)")
        return lines

    output_lines = ["animal-ai-unity/"]
    output_lines.append("├── Relevant Unity assets (scripts, prefabs, scenes, ML-Agents, test configs)")

    for directory in include_dirs:
        section = list_tree(directory)
        for line in section:
            output_lines.append(f"│   {line}")

    output_lines.append("├── Adapter/project connection files")
    for index, file_path in enumerate(include_files):
        rel = file_path.relative_to(unity_root)
        marker = "└──" if index == len(include_files) - 1 else "├──"
        if file_path.exists() and not should_skip(file_path):
            output_lines.append(f"│   {marker} {rel}")
        else:
            output_lines.append(f"│   {marker} {rel} (missing)")

    output_lines.append("└── Excluded: Library/, Logs/, Temp/, build artifacts, caches, and *.meta files")

    out_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(out_file)
    print(f"Wrote {len(output_lines)} lines")


if __name__ == "__main__":
    main()
