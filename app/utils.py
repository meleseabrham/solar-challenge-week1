"""Utility functions for the solar potential Streamlit dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import plotly.express as px

# Paths & configuration -----------------------------------------------------------------

# Project layout: repository_root/solar-challenge-week1/data
# Handle both local development and Streamlit Cloud deployment
_current_file = Path(__file__).resolve()
# In Streamlit Cloud: app/utils.py -> go up 1 level to repo root
# In local dev: app/utils.py -> up 2 levels to project root, then into solar-challenge-week1
if (_current_file.parents[1] / "data").exists():
    # We're in the solar-challenge-week1 repository (Streamlit Cloud or direct repo)
    REPO_ROOT = _current_file.parents[1]
    DATA_DIR = REPO_ROOT / "data"
else:
    # Fallback: assume nested structure (local development)
    REPO_ROOT = _current_file.parents[2]
    DATA_DIR = REPO_ROOT / "solar-challenge-week1" / "data"
SOLAR_COLS = ["GHI", "DNI", "DHI"]

FILE_MAP = {
    "Benin": {"clean": "benin_clean.csv", "raw": "benin-malanville.csv"},
    "Sierra Leone": {"clean": "sierraleone_clean.csv", "raw": "sierraleone-bumbuna.csv"},
    "Togo": {"clean": "togo_clean.csv"},
}

AVAILABLE_COUNTRIES: List[str] = list(FILE_MAP.keys())

_REGION_COLUMNS = [
    "Region",
    "Subregion",
    "Site",
    "Station",
    "Location",
    "City",
]

# Data loading --------------------------------------------------------------------------


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def _coerce_solar_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in SOLAR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def _clean_dataframe(df: pd.DataFrame, apply_outlier_detection: bool = True) -> pd.DataFrame:
    """Clean dataframe with optional metric-specific outlier detection."""
    df = _coerce_timestamp(df)
    df = _coerce_solar_columns(df)
    df = df.dropna(subset=SOLAR_COLS, how="all")
    df = df.drop_duplicates()
    
    if apply_outlier_detection:
        # Detect outliers but don't remove them - flag for analysis
        outlier_flags, _ = detect_solar_metric_outliers(df)
        for col in outlier_flags.columns:
            if col in df.columns or col.replace("_outlier", "") in df.columns:
                df[col] = outlier_flags[col]
    
    return df


def load_country_dataset_from_upload(uploaded_file, country: str) -> pd.DataFrame:
    """Load dataset from an uploaded file object.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        country: Country name for labeling
    """
    import io
    # Reset file pointer in case it was read before
    uploaded_file.seek(0)
    df = pd.read_csv(io.BytesIO(uploaded_file.read()))
    df = _clean_dataframe(df.copy(), apply_outlier_detection=True)
    df["country"] = country
    return df


@lru_cache(maxsize=len(FILE_MAP))
def load_country_dataset(country: str, apply_outlier_detection: bool = True) -> pd.DataFrame:
    """Return the cleaned dataset for a single country, creating it if needed."""

    if country not in FILE_MAP:
        raise KeyError(f"Unknown country '{country}'. Options: {', '.join(FILE_MAP)}")

    config = FILE_MAP[country]
    clean_path = DATA_DIR / config["clean"]
    raw_path = DATA_DIR / config.get("raw", config["clean"])

    if clean_path.exists():
        df = pd.read_csv(clean_path)
    else:
        if not raw_path.exists():
            raise FileNotFoundError(f"No dataset available for {country} at {raw_path}")
        df = pd.read_csv(raw_path)
        df = _clean_dataframe(df, apply_outlier_detection=apply_outlier_detection)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(clean_path, index=False)

    df = _clean_dataframe(df.copy(), apply_outlier_detection=apply_outlier_detection)
    df["country"] = country
    return df


def load_combined_dataset(
    countries: Iterable[str] | None = None,
    uploaded_files: dict | None = None
) -> pd.DataFrame:
    """Return a concatenated dataframe for the provided countries.
    
    Args:
        countries: List of country names to load
        uploaded_files: Dictionary mapping country names to uploaded file objects
    """

    selected = list(countries) if countries is not None else AVAILABLE_COUNTRIES
    frames = []
    
    for country in selected:
        # Check if we have an uploaded file for this country
        if uploaded_files and country in uploaded_files:
            df = load_country_dataset_from_upload(uploaded_files[country], country)
        else:
            # Fall back to local files
            df = load_country_dataset(country, apply_outlier_detection=True)
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    combined["country"] = combined["country"].astype("category")
    return combined


# Aggregations & palette -----------------------------------------------------------------


def country_palette(countries: Iterable[str] | None = None) -> dict[str, str]:
    """Return a deterministic color mapping for the provided countries.

    Uses a high-saturation palette for better contrast.
    """

    palette = px.colors.qualitative.Bold
    countries = list(countries) if countries is not None else AVAILABLE_COUNTRIES
    return {country: palette[i % len(palette)] for i, country in enumerate(countries)}


def detect_region_column(df: pd.DataFrame) -> str:
    """Return the name of the best available region column, falling back to country."""

    for candidate in _REGION_COLUMNS:
        if candidate in df.columns:
            return candidate
    return "country"


def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/median/std summary for solar metrics grouped by country."""

    summary = (
        df.groupby("country", observed=True)[SOLAR_COLS]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    summary.columns = [" ".join(col).strip() for col in summary.columns]
    return summary.reset_index()


def top_regions(df: pd.DataFrame, metric: str, top_n: int = 5) -> pd.DataFrame:
    """Return the highest-average regions for the requested metric.

    Handles the case where the region column is itself 'country' to avoid
    duplicate column name collisions during reset_index().
    """

    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in dataframe columns")

    region_col = detect_region_column(df)
    group_cols = [region_col, "country"] if region_col != "country" else ["country"]

    rankings = (
        df.groupby(group_cols, observed=True)[metric]
        .mean()
        .reset_index()
        .sort_values(metric, ascending=False)
    )

    # Ensure the region column is present for display even when grouping by country only
    if region_col == "country" and region_col not in rankings.columns:
        rankings[region_col] = rankings["country"]

    rankings[metric] = rankings[metric].round(2)
    return rankings.head(top_n)
