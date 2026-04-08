#!/usr/bin/env python3
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_file = repo_root / "codebase_structure.txt"

    exclude_dir_names = {
        ".git",
        ".hg",
        ".svn",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        "__pycache__",
        "venv",
        ".venv",
        "node_modules",
        "animal-ai-unity",
        "unity",
        "Library",
        "Temp",
        "Logs",
        "Log",
        "Cache",
        "Caches",
    }

    exclude_file_suffixes = {".log", ".tmp", ".cache"}
    exclude_file_names = {".DS_Store", out_file.name}

    def should_exclude_dir(name: str) -> bool:
        n = name.strip()
        low = n.lower()
        exclude_dir_names_lower = {x.lower() for x in exclude_dir_names}
        if n in exclude_dir_names or low in exclude_dir_names_lower:
            return True
        if low.endswith("cache") or low.endswith("caches"):
            return True
        if low in {"logs", "log"}:
            return True
        return False

    def should_exclude_file(name: str) -> bool:
        if name in exclude_file_names:
            return True
        low = name.lower()
        if low.endswith(tuple(exclude_file_suffixes)):
            return True
        return False

    def tree_lines(path: Path, prefix: str = ""):
        try:
            entries = sorted(
                list(path.iterdir()),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return

        filtered = []
        for entry in entries:
            if entry.is_dir() and should_exclude_dir(entry.name):
                continue
            if entry.is_file() and should_exclude_file(entry.name):
                continue
            filtered.append(entry)

        for i, entry in enumerate(filtered):
            is_last = i == len(filtered) - 1
            branch = "└── " if is_last else "├── "
            name = entry.name + ("/" if entry.is_dir() else "")
            yield f"{prefix}{branch}{name}"
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                yield from tree_lines(entry, prefix + extension)

    lines = [f"{repo_root.name}/"]
    lines.extend(tree_lines(repo_root))
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_file)
    print(f"Wrote {len(lines)} lines")


if __name__ == "__main__":
    main()
