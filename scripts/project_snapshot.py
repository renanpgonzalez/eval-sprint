#!/usr/bin/env python3
"""Write the whole repository into ONE markdown file.

Useful for mirroring the repo into a Claude Project doc (or any place that takes a
single document): a header (title, date, git commit, size), the file tree, then every
text file in a fenced code block with the language taken from its extension.

Left out by default, with a count per reason at the end of the file:

* ``.git``, ``runs/``, ``demo_out/``, ``__pycache__``, ``.venv``, ``*.egg-info`` and other build or cache folders;
* everything under ``orchestration/`` except the decision documents (``plan.md``, ``report.md``,
  ``contracts.md``, ``_contrato.md``, ``builders.md``, ``critique.md``): scratch scripts, logs, JSON results
  and fixtures of a multi-agent build are not the project (one scratch tree alone was 23 MB);
* any folder called ``work`` (scratch by convention);
* images and other binary files, symlinks, files above ``--max-kb`` (default 200 KB),
  and ``*.log`` files above ``--max-log-kb`` (default 50 KB).

    python scripts/project_snapshot.py                       # -> ./PROJECT_SNAPSHOT.md, prints the size
    python scripts/project_snapshot.py --no-tests            # also leave out tests/ (about half the size)
    python scripts/project_snapshot.py --include-orchestration   # keep orchestration/ scratch too (work/ stays out)
    python scripts/project_snapshot.py --out /tmp/snap.md --root /path/to/repo
    python scripts/project_snapshot.py --exclude 'examples/*' --exclude '*.jsonl'   # your own globs
    python scripts/project_snapshot.py --no-tests --exclude 'orchestration/*'       # the smallest useful snapshot

The copy that goes into the Claude Project is made with ``--no-tests`` (the tests stay in the git repository).
Sizes at v0.2.0: about 2.6 MB by default (the tests are about half), about 1.4 MB with ``--no-tests``, about 1.3 MB with
``--no-tests --exclude 'orchestration/*'``. A test keeps these figures within 30% of the real ones.

Standard library only.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    "runs",
    "demo_out",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}
EXCLUDED_DIR_SUFFIXES = (".egg-info",)
EXCLUDED_FILES = {".DS_Store"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".svg", ".tif", ".tiff", ".heic"}
BINARY_SUFFIXES = IMAGE_SUFFIXES - {".svg"} | {
    ".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".tar", ".whl", ".pyc", ".pyo", ".so", ".dll", ".dylib",
    ".exe", ".woff", ".woff2", ".ttf", ".otf", ".mp3", ".mp4", ".mov", ".sqlite", ".db", ".parquet", ".pkl", ".npy",
}
# Decision documents kept from orchestration/ (matched on the file name, at any depth).
ORCHESTRATION_DIR = "orchestration"
ORCHESTRATION_KEEP = {"plan.md", "report.md", "contracts.md", "_contrato.md", "builders.md", "critique.md"}
SCRATCH_DIR = "work"
LANGUAGES = {
    ".py": "python",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "jsonl",
    ".toml": "toml",
    ".sh": "bash",
    ".bash": "bash",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".csv": "csv",
    ".ini": "ini",
    ".cfg": "ini",
    ".sql": "sql",
    ".txt": "text",
    ".xml": "xml",
    ".svg": "xml",
}
NAMED_LANGUAGES = {"Makefile": "makefile", "Dockerfile": "dockerfile"}

# Reasons files are left out by rule (counted, not listed one by one).
REASON_ORCHESTRATION = "orchestration/ scratch (only plan, report, contracts, builders, critique documents are kept)"
REASON_WORK = f"scratch folders named {SCRATCH_DIR}/"
REASON_IMAGE = "images"
REASON_LOG = "logs above the size limit"
REASON_TESTS = "tests/ (--no-tests)"
REASON_USER = "--exclude patterns"


@dataclass
class Options:
    max_kb: int = 200
    max_log_kb: int = 50
    exclude: list[str] = field(default_factory=list)
    no_tests: bool = False
    include_orchestration: bool = False


@dataclass
class Snapshot:
    markdown: str
    included: list[tuple[Path, str]]
    skipped: list[tuple[Path, str]]
    excluded: Counter

    @property
    def size_bytes(self) -> int:
        return len(self.markdown.encode("utf-8"))


def language_for(path: Path) -> str:
    return NAMED_LANGUAGES.get(path.name) or LANGUAGES.get(path.suffix.lower(), "text")


def is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIRS or name.endswith(EXCLUDED_DIR_SUFFIXES)


def read_text_or_none(path: Path) -> str | None:
    """File content as text, or None when the file is binary / not UTF-8."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_user_excluded(rel: Path, patterns: list[str]) -> bool:
    posix = rel.as_posix()
    return any(fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(rel.name, pat) for pat in patterns)


def rule_exclusion(rel: Path, size: int | None, opts: Options) -> str | None:
    """Why the default rules leave `rel` out, or None. `size` may be None for a directory-level check."""
    parts = rel.parts
    if parts[0] == ORCHESTRATION_DIR and not opts.include_orchestration and rel.name not in ORCHESTRATION_KEEP:
        return REASON_ORCHESTRATION
    if SCRATCH_DIR in parts[:-1]:
        return REASON_WORK
    if opts.no_tests and parts[0] == "tests":
        return REASON_TESTS
    if rel.suffix.lower() in IMAGE_SUFFIXES:
        return REASON_IMAGE
    if size is not None and rel.suffix.lower() == ".log" and size > opts.max_log_kb * 1024:
        return REASON_LOG
    return None


def collect(
    root: Path, out_path: Path | None, max_bytes: int, exclude: list[str] | None = None, opts: Options | None = None
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], Counter]:
    """Walk `root` in a stable order.

    Returns (included [(rel_path, text)], skipped [(rel_path, reason)], excluded {reason: count}).
    `skipped` lists files that would normally count (too big, binary, symlink); `excluded` only counts the
    files the default rules leave out, so hundreds of scratch files do not bloat the list.
    """
    opts = opts or Options(max_kb=max_bytes // 1024, exclude=list(exclude or []))
    included: list[tuple[Path, str]] = []
    skipped: list[tuple[Path, str]] = []
    excluded: Counter = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not is_excluded_dir(d) and not (Path(dirpath) / d).is_symlink())
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            if name in EXCLUDED_FILES:
                continue
            if out_path is not None and path.resolve() == out_path:
                continue
            if is_user_excluded(rel, opts.exclude):
                excluded[REASON_USER] += 1
                continue
            try:
                size = path.lstat().st_size
            except OSError:
                continue
            reason = rule_exclusion(rel, size, opts)
            if reason:
                excluded[reason] += 1
                continue
            if path.is_symlink():
                skipped.append((rel, "symlink"))
                continue
            if path.suffix.lower() in BINARY_SUFFIXES:
                skipped.append((rel, "binary"))
                continue
            if size > max_bytes:
                skipped.append((rel, f"{size / 1024:.0f} KB, over the {max_bytes // 1024} KB limit"))
                continue
            text = read_text_or_none(path)
            if text is None:
                skipped.append((rel, "binary or not UTF-8"))
                continue
            included.append((rel, text))
    return included, skipped, excluded


def render_tree(paths: list[Path], root_name: str) -> str:
    """ASCII tree of the given relative paths (directories first, like `tree`)."""
    tree: dict = {}
    for p in paths:
        node = tree
        for part in p.parts[:-1]:
            node = node.setdefault(part, {})
        node[p.parts[-1]] = None

    lines = [f"{root_name}/"]

    def walk(node: dict, prefix: str) -> None:
        entries = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
        for i, (name, child) in enumerate(entries):
            last = i == len(entries) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if child is not None else ''}")
            if child is not None:
                walk(child, prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(lines)


def fence_for(text: str) -> str:
    """A backtick fence longer than any backtick run inside `text` (so nested ``` blocks survive)."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def git_info(root: Path) -> str | None:
    def git(*args: str) -> str | None:
        try:
            res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    sha = git("rev-parse", "--short=12", "HEAD")
    if not sha:
        return None
    subject = git("log", "-1", "--format=%s") or ""
    dirty = git("status", "--porcelain")
    info = f"`{sha}`" + (f" — {subject}" if subject else "")
    if dirty:
        info += " (uncommitted changes present)"
    return info


def describe_excluded(excluded: Counter) -> str:
    return "; ".join(f"{count} file(s): {reason}" for reason, count in sorted(excluded.items())) or "none"


def snapshot(root: Path, out_path: Path | None = None, opts: Options | None = None, now: datetime | None = None) -> Snapshot:
    """Build the snapshot document. The size in the header is the size of the finished document."""
    opts = opts or Options()
    included, skipped, excluded = collect(root, out_path, opts.max_kb * 1024, opts=opts)
    now = now or datetime.now(timezone.utc)
    title = root.name or str(root)

    def document(size_note: str) -> str:
        out = [
            f"# Project snapshot: {title}",
            "",
            f"- Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"- Git commit: {git_info(root) or 'not available (not a git repository, or git is not installed)'}",
            f"- Files included: {len(included)} (skipped: {len(skipped)}; left out by rule: {sum(excluded.values())})",
            f"- Size: {size_note}",
            "",
            "Auto-generated by `scripts/project_snapshot.py`. Do not edit by hand: regenerate instead.",
            "",
            "## File tree",
            "",
            "```text",
            render_tree([rel for rel, _ in included], title),
            "```",
            "",
            "## Files",
        ]
        for rel, text in included:
            fence = fence_for(text)
            out += ["", f"### `{rel.as_posix()}`", "", f"{fence}{language_for(rel)}", text.rstrip("\n"), fence]
        if excluded:
            out += ["", "## Left out by rule", ""]
            out += [f"- {count} file(s): {reason}" for reason, count in sorted(excluded.items())]
        if skipped:
            out += ["", "## Skipped files", ""]
            out += [f"- `{rel.as_posix()}`: {reason}" for rel, reason in skipped]
        return "\n".join(out) + "\n"

    # The size line is part of the document, so settle it in two passes (its own width barely changes).
    markdown = document("0 KB")
    for _ in range(3):
        note = f"{len(markdown.encode('utf-8')) / 1024:.0f} KB"
        rebuilt = document(note)
        if rebuilt == markdown:
            break
        markdown = rebuilt
    return Snapshot(markdown, included, skipped, excluded)


def build_snapshot(
    root: Path, out_path: Path | None = None, max_kb: int = 200, now: datetime | None = None, exclude: list[str] | None = None
) -> tuple[str, int, int]:
    """Returns (markdown, n_files_included, n_files_skipped). Kept for callers of the original signature."""
    snap = snapshot(root, out_path, Options(max_kb=max_kb, exclude=list(exclude or [])), now)
    return snap.markdown, len(snap.included), len(snap.skipped)


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Write the repository into one markdown file (tree + every text file).")
    ap.add_argument("--root", default=str(default_root), help="repository root (default: the parent of scripts/)")
    ap.add_argument("--out", default="PROJECT_SNAPSHOT.md", help="output file (default: PROJECT_SNAPSHOT.md in the current directory)")
    ap.add_argument("--max-kb", type=int, default=200, help="skip files larger than this many KB (default: 200)")
    ap.add_argument("--max-log-kb", type=int, default=50, help="leave out *.log files larger than this many KB (default: 50)")
    ap.add_argument(
        "--no-tests",
        action="store_true",
        help="leave out tests/ (about half the size). Use it for the copy that goes into the Claude Project: the tests stay in the git repository",
    )
    ap.add_argument(
        "--include-orchestration",
        action="store_true",
        help="keep every file under orchestration/ except folders named work/ (default: only plan, report, contracts, builders and critique documents)",
    )
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="also skip files matching this glob (repeatable), e.g. --exclude 'tests/fixtures/*'")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: --root {args.root} is not a directory", file=sys.stderr)
        return 2
    out_path = Path(args.out).resolve()
    opts = Options(
        max_kb=args.max_kb, max_log_kb=args.max_log_kb, exclude=args.exclude, no_tests=args.no_tests, include_orchestration=args.include_orchestration
    )
    snap = snapshot(root, out_path, opts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(snap.markdown, encoding="utf-8")
    print(f"wrote {out_path} ({snap.size_bytes / 1024:.0f} KB): {len(snap.included)} files, {len(snap.skipped)} skipped, {sum(snap.excluded.values())} left out by rule")
    if snap.excluded:
        print(f"left out by rule: {describe_excluded(snap.excluded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
