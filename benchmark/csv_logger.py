"""
Lightweight CSV metric logger for benchmark training runs.

Writes metrics to a CSV file with headers, flushing on every write
so no data is lost if the training run crashes.
"""

from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    """
    Simple CSV logger that writes one row per call to log().

    Flushes after every write to avoid data loss on long runs.
    Supports context manager protocol for clean shutdown.
    """

    def __init__(self, path: str, fieldnames: list[str]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict) -> None:
        """Write a single row of metrics. Flushes immediately."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
