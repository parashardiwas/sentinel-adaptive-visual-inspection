"""Fail when tracked Git content contains common secrets or large model artifacts."""

import re
import subprocess
import sys
from pathlib import Path


SECRET_PATTERNS = {
    "Groq/API token": re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b"),
    "OpenAI/API token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".joblib", ".pem", ".p12", ".pfx", ".key"}
FORBIDDEN_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "secrets.toml"}
FORBIDDEN_DIRS = {".ssh", ".aws"}
MAX_FILE_BYTES = 100 * 1024 * 1024


def git_paths() -> tuple[list[Path], set[Path]]:
    output = subprocess.check_output(["git", "ls-files", "-z"], text=False)
    paths = {Path(item.decode("utf-8")) for item in output.split(b"\0") if item}
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], text=False)
    staged_paths = {Path(item.decode("utf-8")) for item in staged.split(b"\0") if item}
    paths.update(staged_paths)
    return sorted(paths), staged_paths


def main() -> int:
    problems = []
    paths, staged_paths = git_paths()
    for path in paths:
        normalized = path.as_posix().lower()
        if path.name.lower() in FORBIDDEN_NAMES or any(part.lower() in FORBIDDEN_DIRS for part in path.parts) or (path.name.startswith(".env.") and path.name != ".env.example"):
            problems.append(f"sensitive environment file is tracked: {path}")
        if path.name.lower() in FORBIDDEN_NAMES or "credentials" in path.name.lower() or "service-account" in path.name.lower() or any(normalized.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            problems.append(f"credential/model artifact type is tracked: {path}")
        if path in staged_paths:
            staged_size_result = subprocess.run(
                ["git", "cat-file", "-s", f":{path.as_posix()}"], capture_output=True, text=True, check=False
            )
            if staged_size_result.returncode != 0:
                continue
            staged_size = int(staged_size_result.stdout.strip())
            if staged_size > MAX_FILE_BYTES:
                problems.append(f"file exceeds 100 MiB: {path}")
                continue
            blob = subprocess.run(
                ["git", "show", f":{path.as_posix()}"], capture_output=True, check=False
            )
            if blob.returncode != 0:
                continue
            raw_content = blob.stdout
        elif path.is_file():
            raw_content = path.read_bytes()
            staged_size = path.stat().st_size
        else:
            continue
        if staged_size > MAX_FILE_BYTES:
            problems.append(f"file exceeds 100 MiB: {path}")
        try:
            content = raw_content.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                problems.append(f"possible {label} found in {path}")

    # Check existing reachable history too, so deleting a secret file from the
    # current tree does not make a previously committed credential look safe.
    history_paths = subprocess.check_output(
        ["git", "log", "--all", "--name-only", "--format="], text=True
    ).splitlines()
    for name in history_paths:
        path = Path(name)
        if path.name.lower() in FORBIDDEN_NAMES or any(part.lower() in FORBIDDEN_DIRS for part in path.parts) or (path.name.startswith(".env.") and path.name != ".env.example"):
            problems.append(f"sensitive environment file appears in Git history: {path}")
        if path.name.lower() in FORBIDDEN_NAMES or "credentials" in path.name.lower() or "service-account" in path.name.lower() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"credential/model artifact type appears in Git history: {path}")
    history = subprocess.check_output(["git", "log", "--all", "-p", "--format="], text=True, errors="replace")
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(history):
            problems.append(f"possible {label} found in Git history")

    if problems:
        print("Git safety check failed:")
        print("\n".join(f"- {problem}" for problem in problems))
        return 1
    print("Git safety check passed: no tracked credential files, common secret patterns, or oversized artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
