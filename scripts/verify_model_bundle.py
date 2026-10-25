"""Check that the local Sentinel model bundle matches model_manifest.json."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("MVTEC_PROJECT_DIR", Path(__file__).resolve().parents[1])),
        help="Bundle root containing models/ and assets/ (defaults to MVTEC_PROJECT_DIR or the repository).",
    )
    parser.add_argument("--with-rl", action="store_true", help="Also require temporal inference artifacts.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo_root / "model_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifact_distribution"]
    categories = manifest["dataset_categories"]
    required = []
    for item in artifacts["required_for_pipeline_b"]:
        if "<category>" in item:
            required.extend(Path(item.replace("<category>", category)) for category in categories)
        else:
            required.append(Path(item))
    if args.with_rl:
        required.extend(Path(item) for item in artifacts["required_for_rl"])

    missing = [path for path in required if not (args.root / path).is_file()]
    if missing:
        print(f"Model bundle incomplete at {args.root} ({len(missing)} missing file(s)):")
        for path in missing:
            print(f"  MISSING {path}")
        return 1

    checksum_file = repo_root / "MODEL_ASSETS.sha256"
    expected_checksums = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected_checksums[name.strip()] = digest
    mismatched = []
    for path in required:
        expected = expected_checksums.get(path.as_posix())
        if not expected:
            mismatched.append(f"{path} (no checksum entry)")
            continue
        digest = hashlib.sha256()
        with (args.root / path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            mismatched.append(f"{path} (SHA-256 mismatch)")
    if mismatched:
        print("Model bundle checksum verification failed:")
        for item in mismatched:
            print(f"  {item}")
        return 1

    print(f"Model bundle files and SHA-256 checks passed at {args.root}: {len(required)} checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
