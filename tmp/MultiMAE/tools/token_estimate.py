#!/usr/bin/env python3

from pathlib import Path
import argparse

import tiktoken

# Optional: support .gitignore
try:
    import pathspec
except ImportError:
    pathspec = None


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    "node_modules",
    "tools",
}


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
    ".zsh",
    #".ipynb",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cu",
    ".cuh",
    ".java",
    ".go",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".xml",
}


def load_gitignore(root):
    if pathspec is None:
        return None

    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return None

    with gitignore.open() as f:
        return pathspec.PathSpec.from_lines(
            "gitwildmatch",
            f.readlines(),
        )


def should_skip(path, root, spec):
    rel = path.relative_to(root)

    for part in rel.parts:
        if part in DEFAULT_EXCLUDES:
            return True

    if spec and spec.match_file(str(rel)):
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="Tokenizer encoding (default: o200k_base)",
    )

    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)

    spec = load_gitignore(args.project)

    files = []

    total_tokens = 0
    total_lines = 0
    total_chars = 0

    for file in args.project.rglob("*"):
        if not file.is_file():
            continue

        if should_skip(file, args.project, spec):
            continue

        if file.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        tokens = len(enc.encode(text))
        lines = text.count("\n") + 1

        total_tokens += tokens
        total_lines += lines
        total_chars += len(text)

        files.append(
            (
                tokens,
                lines,
                len(text),
                file.relative_to(args.project),
            )
        )

    files.sort(reverse=True)

    print("=" * 70)
    print(f"Project : {args.project}")
    print(f"Files   : {len(files)}")
    print(f"Lines   : {total_lines:,}")
    print(f"Chars   : {total_chars:,}")
    print(f"Tokens  : {total_tokens:,}")
    print("=" * 70)

    print("\nTop 30 largest files:\n")

    for tokens, lines, chars, path in files[:30]:
        print(
            f"{tokens:8,d} tokens | "
            f"{lines:6,d} lines | "
            f"{chars:8,d} chars | "
            f"{path}"
        )


if __name__ == "__main__":
    main()
