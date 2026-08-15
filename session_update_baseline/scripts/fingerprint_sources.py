#!/usr/bin/env python3
"""Compute a deterministic digest for an explicitly selected source tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


IGNORED_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    "build",
    "install",
    "log",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def iter_files(root: Path, selections: list[Path]):
    files: set[Path] = set()
    for selection in selections:
        path = (root / selection).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"selection escapes root: {selection}") from error
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() or path.is_symlink() else path.rglob("*")
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
                continue
            if candidate.suffix in IGNORED_SUFFIXES:
                continue
            if candidate.is_file() or candidate.is_symlink():
                files.add(candidate)
    yield from sorted(files, key=lambda item: item.relative_to(root).as_posix())


def fingerprint(root: Path, selections: list[Path]) -> str:
    digest = hashlib.sha256()
    count = 0
    for path in iter_files(root, selections):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            payload = ("SYMLINK:" + str(path.readlink())).encode("utf-8")
        else:
            payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        count += 1
    if count == 0:
        raise ValueError("source selection contains no files")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    print(fingerprint(root, args.paths))


if __name__ == "__main__":
    main()
