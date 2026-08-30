from __future__ import annotations

from pathlib import Path

import pytest

from adjudicator import Dataset, DatasetConfig, load_dataset

CAMEL_DATASET_PATH = Path("datasets/camel-1.6.csv")


@pytest.fixture
def camel_dataset() -> Dataset:
    """Load the Camel 1.6 dataset."""

    config = DatasetConfig(path=CAMEL_DATASET_PATH)
    return load_dataset(config)
