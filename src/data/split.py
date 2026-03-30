"""Dataset filtering and group-based split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

INDEX_COLUMNS: Final[list[str]] = [
    "filepath",
    "machine_type",
    "snr_db",
    "machine_id",
    "label",
]
SPLIT_COLUMNS: Final[list[str]] = INDEX_COLUMNS + ["group_id", "split"]
VALID_LABELS: Final[set[str]] = {"normal", "abnormal"}
VALID_SPLITS: Final[tuple[str, str, str]] = ("train", "val", "test")


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Validate split ratios are non-negative and sum to one."""
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "Split ratios must sum to 1.0, got "
            f"{ratio_sum:.6f} from train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )


def filter_index_by_machine_type(
    index_df: pd.DataFrame,
    machine_type: str | None = None,
) -> pd.DataFrame:
    """Filter a master index to a specific machine type or keep all rows."""
    required_columns = {"filepath", "machine_type", "snr_db", "machine_id", "label"}
    missing_columns = required_columns.difference(index_df.columns)
    if missing_columns:
        raise ValueError(
            "Index is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    if machine_type is None:
        filtered_df = index_df.copy()
    else:
        filtered_df = index_df.loc[index_df["machine_type"] == machine_type].copy()
        if filtered_df.empty:
            raise ValueError(
                f"No files found for machine_type='{machine_type}' in the provided index."
            )

    invalid_labels = sorted(set(filtered_df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(
            "Unexpected labels found after filtering: "
            f"{', '.join(invalid_labels)}."
        )

    return filtered_df.reset_index(drop=True)


def build_group_id(index_df: pd.DataFrame) -> pd.Series:
    """Construct group identifiers based on the machine types present in scope."""
    machine_types = index_df["machine_type"].drop_duplicates().tolist()
    single_machine_type = len(machine_types) == 1

    if single_machine_type:
        group_values = (
            index_df["machine_id"].astype(str) + "__" + index_df["snr_db"].astype(str)
        )
    else:
        group_values = (
            index_df["machine_type"].astype(str)
            + "__"
            + index_df["machine_id"].astype(str)
            + "__"
            + index_df["snr_db"].astype(str)
        )

    return group_values.rename("group_id")


def build_split_for_scope(
    index_df: pd.DataFrame,
    machine_type: str | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Filter an index to the selected scope and assign deterministic group-based splits."""
    scoped_index_df = filter_index_by_machine_type(
        index_df=index_df,
        machine_type=machine_type,
    )
    if machine_type is not None:
        return assign_group_splits(
            index_df=scoped_index_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    split_parts = []
    for machine_name, machine_df in scoped_index_df.groupby("machine_type", sort=True):
        machine_split_df = assign_group_splits(
            index_df=machine_df.reset_index(drop=True),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        machine_split_df = machine_split_df.loc[:, SPLIT_COLUMNS].copy()
        machine_split_df["group_id"] = (
            str(machine_name) + "__" + machine_split_df["group_id"].astype(str)
        )
        split_parts.append(machine_split_df)

    if not split_parts:
        raise ValueError("Cannot create splits from an empty index.")

    combined_split_df = pd.concat(split_parts, ignore_index=True)
    validate_no_group_leakage(combined_split_df)
    return (
        combined_split_df.loc[:, SPLIT_COLUMNS]
        .sort_values(["split", "machine_type", "group_id", "filepath"])
        .reset_index(drop=True)
    )


def assign_group_splits(
    index_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits by unique group ID."""
    if index_df.empty:
        raise ValueError("Cannot create splits from an empty index.")

    validate_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    working_df = index_df.loc[:, INDEX_COLUMNS].copy()
    invalid_labels = sorted(set(working_df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(
            "Unexpected labels found before splitting: "
            f"{', '.join(invalid_labels)}."
        )

    working_df["group_id"] = build_group_id(working_df)

    unique_groups = np.array(sorted(working_df["group_id"].unique()))
    if unique_groups.size == 0:
        raise ValueError("No groups found to split.")

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    n_groups = int(shuffled_groups.size)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))

    if n_train > n_groups:
        n_train = n_groups
    if n_train + n_val > n_groups:
        n_val = max(0, n_groups - n_train)

    train_groups = shuffled_groups[:n_train]
    val_groups = shuffled_groups[n_train : n_train + n_val]
    test_groups = shuffled_groups[n_train + n_val :]

    group_to_split = {group_id: "train" for group_id in train_groups}
    group_to_split.update({group_id: "val" for group_id in val_groups})
    group_to_split.update({group_id: "test" for group_id in test_groups})

    working_df["split"] = working_df["group_id"].map(group_to_split)
    if working_df["split"].isna().any():
        missing_groups = sorted(working_df.loc[working_df["split"].isna(), "group_id"].unique())
        raise ValueError(
            "Failed to assign split labels to all groups: "
            f"{', '.join(missing_groups)}."
        )

    validate_no_group_leakage(working_df)
    return working_df.loc[:, SPLIT_COLUMNS].sort_values(["split", "group_id", "filepath"]).reset_index(drop=True)


def validate_no_group_leakage(split_df: pd.DataFrame) -> None:
    """Ensure each group ID appears in exactly one split."""
    missing_columns = {"group_id", "split"}.difference(split_df.columns)
    if missing_columns:
        raise ValueError(
            "Split dataframe is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    invalid_splits = sorted(set(split_df["split"]) - set(VALID_SPLITS))
    if invalid_splits:
        raise ValueError(
            "Unexpected split labels found: "
            f"{', '.join(invalid_splits)}."
        )

    leakage = (
        split_df.groupby("group_id")["split"]
        .nunique()
        .loc[lambda series: series > 1]
    )
    if not leakage.empty:
        leaked_groups = ", ".join(leakage.index.astype(str).tolist())
        raise ValueError(f"Group leakage detected across splits: {leaked_groups}.")


def infer_scope_name(split_df: pd.DataFrame) -> str:
    """Infer output scope name from the machine types present in a split dataframe."""
    machine_types = sorted(split_df["machine_type"].drop_duplicates().tolist())
    if len(machine_types) == 1:
        return machine_types[0]
    return "all_machines"


def save_split_manifest(split_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist a split manifest CSV to disk."""
    missing_columns = [column for column in SPLIT_COLUMNS if column not in split_df.columns]
    if missing_columns:
        raise ValueError(
            "Split manifest is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    split_df.loc[:, SPLIT_COLUMNS].to_csv(output_file, index=False)
    return output_file


def load_split_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load a split manifest CSV from disk."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Split manifest does not exist: '{manifest_file}'.")

    split_df = pd.read_csv(manifest_file)
    missing_columns = [column for column in SPLIT_COLUMNS if column not in split_df.columns]
    if missing_columns:
        raise ValueError(
            "Loaded split manifest is missing required columns: "
            f"{', '.join(missing_columns)}."
        )

    validate_no_group_leakage(split_df)
    invalid_labels = sorted(set(split_df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(
            "Loaded split manifest contains invalid labels: "
            f"{', '.join(invalid_labels)}."
        )

    return split_df.loc[:, SPLIT_COLUMNS].copy()


def summarize_splits(split_df: pd.DataFrame) -> dict[str, object]:
    """Build summary statistics for console reporting."""
    total_files = int(len(split_df))

    split_counts = split_df["split"].value_counts()
    files_per_split = {
        split_name: int(split_counts.get(split_name, 0))
        for split_name in VALID_SPLITS
    }

    class_counts = (
        split_df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(VALID_SPLITS), fill_value=0)
        .reindex(columns=sorted(VALID_LABELS), fill_value=0)
    )
    class_counts_per_split = {
        split_name: {label: int(class_counts.loc[split_name, label]) for label in class_counts.columns}
        for split_name in class_counts.index
    }
    class_proportions_per_split = {}
    for split_name, class_counts_for_split in class_counts_per_split.items():
        split_total = sum(class_counts_for_split.values())
        if split_total == 0:
            class_proportions_per_split[split_name] = {
                label: 0.0 for label in class_counts_for_split
            }
            continue

        class_proportions_per_split[split_name] = {
            label: class_counts_for_split[label] / split_total
            for label in class_counts_for_split
        }

    group_counts = split_df.groupby("split")["group_id"].nunique()
    groups_per_split = {
        split_name: int(group_counts.get(split_name, 0))
        for split_name in VALID_SPLITS
    }

    return {
        "total_files": total_files,
        "files_per_split": files_per_split,
        "class_counts_per_split": class_counts_per_split,
        "class_proportions_per_split": class_proportions_per_split,
        "groups_per_split": groups_per_split,
    }
