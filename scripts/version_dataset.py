"""Create a content-addressed dataset manifest for review and release tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(source: Path, output: Path) -> None:
    frame = pd.read_csv(source, nrows=0)
    manifest = {
        "dataset_file": source.as_posix(),
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
        "columns": frame.columns.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset manifest written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Versioned CSV dataset")
    parser.add_argument("output", type=Path, help="Manifest JSON path")
    args = parser.parse_args()
    create_manifest(args.source, args.output)
