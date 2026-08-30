from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from attrs import define

__all__ = (
    "PROJECT_ROOT",
    "EXPECTED_ROWS",
    "EXPECTED_RAW_COLUMNS",
    "EXPECTED_POSITIVES",
    "FEATURE_COLUMNS",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "Dataset",
    "DatasetConfig",
    "DatasetValidationError",
    "load_dataset",
)

# Constants.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

EXPECTED_ROWS: Final[int] = 965
EXPECTED_RAW_COLUMNS: Final[int] = 24
EXPECTED_POSITIVES: Final[int] = 188

# cite: https://gromit.iiar.pwr.wroc.pl/p_inf/ckjm/metric.html
FEATURE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("wmc", "Weighted methods per class"),
    ("dit", "Depth of inheritance tree"),
    ("noc", "Number of children"),
    ("cbo", "Coupling between object classes"),
    ("rfc", "Response for a class"),
    ("lcom", "Lack of cohesion in methods"),
    ("ca", "Afferent couplings"),
    ("ce", "Efferent couplings"),
    ("npm", "Number of public methods"),
    ("lcom3", "Lack of cohesion in methods (version 3)"),
    ("loc", "Lines of code"),
    ("dam", "Data access metric"),
    ("moa", "Measure of aggregation"),
    ("mfa", "Measure of functional abstraction"),
    ("cam", "Cohesion among methods of class"),
    ("ic", "Inheritance coupling"),
    ("cbm", "Coupling between methods"),
    ("amc", "Average method complexity"),
    ("max_cc", "Maximum cyclomatic complexity"),
    ("avg_cc", "Average cyclomatic complexity"),
)

FEATURE_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in FEATURE_COLUMNS)
FEATURE_DESCRIPTIONS: Final[dict[str, str]] = dict(FEATURE_COLUMNS)


class DatasetValidationError(Exception):
    """Raised when the dataset fails validation."""


@define(frozen=True)
class DatasetConfig:
    """Configuration for a dataset."""

    path: Path


@define(frozen=True)
class Dataset:
    """Represents a valid loaded dataset."""

    features: pd.DataFrame
    target: pd.Series

    class_names: pd.Series
    feature_names: tuple[str, ...]


def _resolve_path(path: Path) -> Path:
    """Resolve a dataset path relative to the project root, if not absolute."""
    resolved = path if path.is_absolute() else PROJECT_ROOT / path

    if not resolved.exists():
        raise DatasetValidationError(f"Dataset file not found: {resolved}") from None

    return resolved


def load_dataset(config: DatasetConfig) -> Dataset:
    """Load and validate the dataset from the given configuration."""
    df = pd.read_csv(_resolve_path(config.path))

    if df.shape != (EXPECTED_ROWS, EXPECTED_RAW_COLUMNS):
        message = (
            f"Expected {EXPECTED_ROWS} rows and {EXPECTED_RAW_COLUMNS} columns, "
            f"but got {df.shape}"
        )

        raise DatasetValidationError(message) from None

    # Rename duplicate name columns.
    df = df.rename(columns={df.columns[0]: "project", df.columns[2]: "class_name"})

    # Compute the target before dropping anything.
    defective = df["bug"] >= 1
    if defective.sum() != EXPECTED_POSITIVES:
        message = (
            f"Expected {EXPECTED_POSITIVES} positive labels, "
            f"but got {defective.sum()}"
        )

        raise DatasetValidationError(message) from None

    class_names = df["class_name"]

    if missing := [name for name in FEATURE_NAMES if name not in df.columns]:
        described = ", ".join(
            f"{name!r} ({FEATURE_DESCRIPTIONS[name]})" for name in missing
        )

        raise DatasetValidationError(
            f"Missing expected feature column(s): {described}"
        ) from None

    features = df[list(FEATURE_NAMES)]

    for column in FEATURE_NAMES:
        series = features[column]
        description = FEATURE_DESCRIPTIONS[column]

        if not pd.api.types.is_numeric_dtype(series):
            raise DatasetValidationError(
                f"Column {column!r} ({description}) is not numeric, "
                f"got dtype {series.dtype}"
            ) from None

        nulls = series[series.isna()]
        if not nulls.empty:
            raise DatasetValidationError(
                f"Column {column!r} ({description}) has null values "
                f"at rows {list(nulls.index)}"
            ) from None

    return Dataset(
        features=features,
        target=defective,
        class_names=class_names,
        feature_names=FEATURE_NAMES,
    )
