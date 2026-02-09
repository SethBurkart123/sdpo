"""Tests for CSV metric logger and evaluation callback."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from csv_logger import CSVLogger


class TestCSVLogger:
    """Tests for the lightweight CSV metric logger."""

    @pytest.fixture
    def tmp_csv(self, tmp_path):
        return tmp_path / "metrics.csv"

    def test_creates_file_with_headers(self, tmp_csv):
        logger = CSVLogger(str(tmp_csv), fieldnames=["step", "loss", "reward"])
        logger.close()
        assert tmp_csv.exists()
        with open(tmp_csv) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["step", "loss", "reward"]

    def test_log_row(self, tmp_csv):
        logger = CSVLogger(str(tmp_csv), fieldnames=["step", "loss"])
        logger.log({"step": 1, "loss": 0.5})
        logger.close()
        with open(tmp_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["step"] == "1"
        assert rows[0]["loss"] == "0.5"

    def test_multiple_rows(self, tmp_csv):
        logger = CSVLogger(str(tmp_csv), fieldnames=["step", "value"])
        for i in range(10):
            logger.log({"step": i, "value": i * 0.1})
        logger.close()
        with open(tmp_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10
        assert rows[0]["step"] == "0"
        assert rows[9]["step"] == "9"

    def test_flushes_on_each_write(self, tmp_csv):
        """Important for long training runs — don't lose data on crash."""
        logger = CSVLogger(str(tmp_csv), fieldnames=["step"])
        logger.log({"step": 1})
        # Don't close — read the file while logger is still open
        with open(tmp_csv) as f:
            content = f.read()
        assert "1" in content

    def test_context_manager(self, tmp_csv):
        with CSVLogger(str(tmp_csv), fieldnames=["step"]) as logger:
            logger.log({"step": 42})
        with open(tmp_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_missing_field_fills_empty(self, tmp_csv):
        logger = CSVLogger(str(tmp_csv), fieldnames=["step", "loss", "extra"])
        logger.log({"step": 1, "loss": 0.5})  # no "extra"
        logger.close()
        with open(tmp_csv) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["extra"] == ""

    def test_extra_field_ignored(self, tmp_csv):
        """Fields not in fieldnames should be silently ignored."""
        logger = CSVLogger(str(tmp_csv), fieldnames=["step"])
        logger.log({"step": 1, "unknown_field": "whatever"})
        logger.close()
        with open(tmp_csv) as f:
            reader = csv.DictReader(f)
            assert "unknown_field" not in reader.fieldnames

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "metrics.csv"
        logger = CSVLogger(str(nested), fieldnames=["step"])
        logger.log({"step": 1})
        logger.close()
        assert nested.exists()

    def test_float_precision(self, tmp_csv):
        """Floats should be written with enough precision."""
        logger = CSVLogger(str(tmp_csv), fieldnames=["val"])
        logger.log({"val": 0.123456789})
        logger.close()
        with open(tmp_csv) as f:
            rows = list(csv.DictReader(f))
        assert "0.1234" in rows[0]["val"]
