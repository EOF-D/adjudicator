from __future__ import annotations

from pathlib import Path

import attrs
import pandas as pd
import pytest

from adjudicator import (
    EXPECTED_POSITIVES,
    EXPECTED_RAW_COLUMNS,
    EXPECTED_ROWS,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    Dataset,
    DatasetConfig,
    DatasetValidationError,
    load_dataset,
)

REAL_DATASET_PATH: Path = Path("datasets", "camel-1.6.csv")
REAL_HEADER = [
    "name",
    "version",
    "name",
    "wmc",
    "dit",
    "noc",
    "cbo",
    "rfc",
    "lcom",
    "ca",
    "ce",
    "npm",
    "lcom3",
    "loc",
    "dam",
    "moa",
    "mfa",
    "cam",
    "ic",
    "cbm",
    "amc",
    "max_cc",
    "avg_cc",
    "bug",
]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False)


def _row_values(overrides: dict[str, object] | None = None) -> list[object]:
    overrides = overrides or {}
    identity = [
        overrides.get("project", "foo"),
        overrides.get("version", "bar"),
        overrides.get("class_name", "baz"),
    ]

    defaults = {name: 1 for name in FEATURE_NAMES} | {"bug": 0}
    features = [overrides.get(col, defaults[col]) for col in REAL_HEADER[3:]]

    return identity + features


def _load_and_expect_error(config: DatasetConfig) -> str:
    with pytest.raises(DatasetValidationError) as err:
        load_dataset(config)

    return str(err)


class TestLoadDatasetHappyPath:
    """load_dataset against the real camel-1.6.csv dataset."""

    def test_loads_real_dataset(self, camel_dataset: Dataset) -> None:
        assert camel_dataset.features.shape == (EXPECTED_ROWS, len(FEATURE_NAMES))
        assert int(camel_dataset.target.sum()) == EXPECTED_POSITIVES
        assert len(camel_dataset.class_names) == EXPECTED_ROWS
        assert camel_dataset.feature_names == FEATURE_NAMES

    def test_feature_columns_match_names_in_order(self, camel_dataset: Dataset) -> None:
        assert camel_dataset.features.columns.tolist() == list(FEATURE_NAMES)

    def test_bug_project_version_class_name_absent_from_features(
        self, camel_dataset: Dataset
    ) -> None:
        for excluded in ("bug", "project", "version", "class_name"):
            assert excluded not in camel_dataset.features.columns

    def test_target_is_bool_dtype(self, camel_dataset: Dataset) -> None:
        assert camel_dataset.target.dtype == bool

    def test_all_feature_columns_numeric(self, camel_dataset: Dataset) -> None:
        for column in camel_dataset.features.columns:
            assert pd.api.types.is_numeric_dtype(camel_dataset.features[column])

    def test_class_names_are_string_valued(self, camel_dataset: Dataset) -> None:
        class_names = camel_dataset.class_names
        assert pd.api.types.is_object_dtype(class_names) or isinstance(
            class_names.dtype, pd.StringDtype
        )

        assert all(isinstance(value, str) for value in class_names)

    def test_relative_path_resolved_against_project_root(self) -> None:
        dataset = load_dataset(DatasetConfig(path=REAL_DATASET_PATH))
        assert dataset.features.shape[0] == EXPECTED_ROWS

    def test_absolute_path_also_works(self) -> None:
        abs_path = (Path(__file__).parent.parent / REAL_DATASET_PATH).resolve()
        dataset = load_dataset(DatasetConfig(path=abs_path))
        assert dataset.features.shape[0] == EXPECTED_ROWS


class TestMissingFile:
    """load_dataset raises when the dataset file does not exist."""

    def test_missing_file_raises_with_resolved_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "foo.csv"
        message = _load_and_expect_error(DatasetConfig(path=missing))

        assert "Dataset file not found" in message
        assert str(missing) in message

    def test_missing_relative_file_reports_resolved_absolute_path(self) -> None:
        relative = Path("datasets", "bar.csv")
        expected_resolved = Path(__file__).parent.parent / relative

        message = _load_and_expect_error(DatasetConfig(path=relative))
        assert str(expected_resolved) in message


class TestWrongShape:
    """load_dataset raises when the raw CSV shape is wrong."""

    def test_wrong_row_count_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "wrong_rows.csv"
        rows = [_row_values() for _ in range(5)]

        _write_csv(csv_path, REAL_HEADER, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert str((5, EXPECTED_RAW_COLUMNS)) in message

    def test_wrong_column_count_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "wrong_cols.csv"
        header = REAL_HEADER[:-1]
        rows = [_row_values()[:-1] for _ in range(EXPECTED_ROWS)]

        _write_csv(csv_path, header, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert str((EXPECTED_ROWS, EXPECTED_RAW_COLUMNS - 1)) in message


class TestWrongPositiveCount:
    """load_dataset raises when the positive-label count is wrong."""

    def test_wrong_positive_count_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "wrong_positives.csv"
        rows = [_row_values({"bug": 0}) for _ in range(EXPECTED_ROWS)]

        _write_csv(csv_path, REAL_HEADER, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert "positive labels" in message
        assert "but got 0" in message


class TestMissingFeatureColumn:
    """load_dataset raises when a named feature column is absent."""

    def test_missing_feature_column_raises_with_description(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "missing_feature.csv"

        header = list(REAL_HEADER)
        header[header.index("wmc")] = "wmc_renamed"

        rows = [
            _row_values({"bug": 1 if i < EXPECTED_POSITIVES else 0})
            for i in range(EXPECTED_ROWS)
        ]

        _write_csv(csv_path, header, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert "Missing expected feature column" in message
        assert "'wmc'" in message
        assert FEATURE_DESCRIPTIONS["wmc"] in message


class TestNonNumericFeatureColumn:
    """load_dataset raises when a feature column is not numeric."""

    def test_non_numeric_feature_column_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "non_numeric.csv"
        rows = [
            _row_values({"bug": 1 if i < EXPECTED_POSITIVES else 0, "wmc": "qux"})
            for i in range(EXPECTED_ROWS)
        ]

        _write_csv(csv_path, REAL_HEADER, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert "'wmc'" in message
        assert "is not numeric" in message
        assert FEATURE_DESCRIPTIONS["wmc"] in message


class TestNullFeatureValue:
    """load_dataset raises when a feature column has null values."""

    def test_null_feature_value_raises_with_row_index(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "null_feature.csv"
        rows: list[list[object]] = []

        for i in range(EXPECTED_ROWS):
            overrides: dict[str, object] = {"bug": 1 if i < EXPECTED_POSITIVES else 0}

            if i == 5:
                overrides["wmc"] = ""

            rows.append(_row_values(overrides))

        _write_csv(csv_path, REAL_HEADER, rows)

        message = _load_and_expect_error(DatasetConfig(path=csv_path))
        assert "'wmc'" in message
        assert "has null values at rows" in message
        assert "[5]" in message
        assert FEATURE_DESCRIPTIONS["wmc"] in message


class TestImmutability:
    """DatasetConfig and Dataset are frozen containers."""

    def test_dataset_config_is_frozen(self) -> None:
        config = DatasetConfig(path=REAL_DATASET_PATH)
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.path = Path(  # pyright: ignore[reportAttributeAccessIssue]
                "foo/bar.csv"
            )

    def test_dataset_is_frozen(self, camel_dataset: Dataset) -> None:
        dataset = camel_dataset
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            dataset.feature_names = ()  # pyright: ignore[reportAttributeAccessIssue]
