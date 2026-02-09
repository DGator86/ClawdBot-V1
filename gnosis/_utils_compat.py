"""Common utilities shared across gnosis modules."""

from typing import List

import numpy as np
import pandas as pd


def drop_future_return_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we NEVER carry truth columns inside predictions until the single attach-truth step.
    This prevents pandas from creating future_return_x / future_return_y during merges.

    Args:
        df: DataFrame that may contain future_return columns

    Returns:
        DataFrame with future_return* columns removed
    """
    cols = [c for c in df.columns if c.startswith("future_return")]
    if cols:
        return df.drop(columns=cols, errors="ignore")
    return df


def safe_merge_no_truth(
    left: pd.DataFrame, right: pd.DataFrame, on: List[str], how: str = "left"
) -> pd.DataFrame:
    """
    Merge right into left but *force* right to not contribute any future_return* columns.

    Args:
        left: Left DataFrame
        right: Right DataFrame (future_return cols will be excluded)
        on: Columns to merge on
        how: Merge type (default "left")

    Returns:
        Merged DataFrame without future_return columns from right
    """
    right_clean = right[[c for c in right.columns if not c.startswith("future_return")]]
    return left.merge(right_clean, on=on, how=how)


def vectorized_abstain_mask(
    s_labels: pd.Series,
    s_pmax: pd.Series,
    confidence_floor: float = 0.65,
) -> np.ndarray:
    """
    Compute abstain mask using vectorized operations.

    Args:
        s_labels: Series of S_label values
        s_pmax: Series of S_pmax values
        confidence_floor: Threshold below which to abstain

    Returns:
        Boolean numpy array where True = should abstain
    """
    # Handle NaN values
    s_labels_clean = s_labels.fillna("S_UNCERTAIN")
    s_pmax_clean = s_pmax.fillna(0.0)

    # Abstain if S_UNCERTAIN or below confidence floor
    is_uncertain = s_labels_clean == "S_UNCERTAIN"
    below_floor = s_pmax_clean < confidence_floor

    return (is_uncertain | below_floor).values


def safe_json_value(val):
    """Convert float('inf') and float('-inf') to JSON-safe strings.

    Args:
        val: Value to convert

    Returns:
        JSON-safe value (inf -> "inf", -inf -> "-inf")
    """
    if isinstance(val, float):
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        if np.isnan(val):
            return None
    return val


def sanitize_for_json(obj):
    """Recursively sanitize a dict/list for JSON serialization.

    Args:
        obj: Dictionary, list, or value to sanitize

    Returns:
        JSON-safe version of the object
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    else:
        return safe_json_value(obj)
