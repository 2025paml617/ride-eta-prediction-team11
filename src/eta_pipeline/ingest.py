"""Ingestion: stream the raw trip CSV out of the source archive in chunks.

The archive member is read through `zipfile.ZipFile.open`, so the 200 MB CSV is
never expanded onto disk. Everything is read as text and left untyped -- type
coercion is a validation concern (see `validate.coerce_types`), because a value
that fails to parse is a data-quality finding we want to report, not an
exception that kills the run.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .config import REQUIRED_COLUMNS, PipelineConfig

logger = logging.getLogger(__name__)


class SchemaError(RuntimeError):
    """The source file does not have the columns the pipeline needs."""


def _open_member(config: PipelineConfig):
    """Return a binary file object for the CSV, from a zip or a bare .csv."""
    path = Path(config.source_path)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")

    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        names = archive.namelist()
        if config.csv_member not in names:
            archive.close()
            raise SchemaError(
                f"{config.csv_member!r} not in {path.name}; members: {names}"
            )
        handle = archive.open(config.csv_member)
        # Keep the archive alive for as long as the member handle is in use.
        handle._owning_archive = archive  # noqa: SLF001
        return handle

    return path.open("rb")


def read_header(config: PipelineConfig) -> list[str]:
    """Read just the header row of the source CSV."""
    with _open_member(config) as handle:
        frame = pd.read_csv(handle, nrows=0)
    return list(frame.columns)


def validate_header(columns: list[str]) -> None:
    """Fail fast if required columns are absent; log any unexpected extras."""
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise SchemaError(
            "source is missing required column(s): "
            + ", ".join(missing)
            + f" -- found: {', '.join(columns)}"
        )

    extra = [c for c in columns if c not in REQUIRED_COLUMNS]
    if extra:
        logger.warning(
            "source has %d unexpected column(s), they will be dropped: %s",
            len(extra),
            ", ".join(extra),
        )


def iter_chunks(config: PipelineConfig) -> Iterator[pd.DataFrame]:
    """Yield the raw CSV as string-typed chunks of `config.chunksize` rows.

    Honours `config.sample_rows`, which caps the total number of rows read for
    fast iteration during development.
    """
    validate_header(read_header(config))

    remaining = config.sample_rows
    with _open_member(config) as handle:
        reader = pd.read_csv(
            handle,
            chunksize=config.chunksize,
            dtype=str,          # coercion happens in validate.py
            keep_default_na=True,
            nrows=config.sample_rows,
        )
        for index, chunk in enumerate(reader):
            chunk = chunk[[c for c in REQUIRED_COLUMNS]]
            logger.debug("read chunk %d: %d rows", index, len(chunk))
            yield chunk

            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
