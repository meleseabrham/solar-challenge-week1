"""Utility functions for the solar potential Streamlit dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import io
import zipfile

import requests

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

# For Cloud fallback: download from GitHub raw if files aren't present locally
GITHUB_OWNER = "meleseabrham"
GITHUB_REPO = "solar-challenge-week1"
GITHUB_BRANCH = "main"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data"

# Secondary fallback: downloadable archive from Dropbox shared folder
DROPBOX_DATA_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fo/j0dznkh3zt5fpv8pwuzjk/ANqnHNjthlQWS8g38GqaCrg"
    "?rlkey=owgskm621o3v40dic91c1iz87&st=ef6zavjl&dl=1"
)

def _ensure_remote_datasets_from_dropbox() -> None:
    """Download and extract datasets from the shared Dropbox archive if available.

    The shared link should point to the folder containing all CSVs. Dropbox delivers
    a zip archive when `dl=1` is appended.
    """

    marker = DATA_DIR / ".dropbox_sync_complete"
    if marker.exists():
        return

    try:
        response = requests.get(DROPBOX_DATA_ARCHIVE_URL, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download Dropbox archive: {exc}") from exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected_files = {cfg["clean"] for cfg in FILE_MAP.values()}
    expected_files.update({cfg["raw"] for cfg in FILE_MAP.values() if "raw" in cfg})

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            filename = Path(member).name
            if filename not in expected_files:
                continue
            target_path = DATA_DIR / filename
            with archive.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())

    try:
        marker.touch()
    except Exception:
        pass

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


# Metric-specific outlier detection thresholds
# GHI: Global Horizontal Irradiance - typically 0-1400 W/m², more variable
# DNI: Direct Normal Irradiance - typically 0-1000 W/m², sensitive to clear sky
# DHI: Diffuse Horizontal Irradiance - typically 0-300 W/m², more stable
OUTLIER_THRESHOLDS = {
    "GHI": {"z_score": 3.5, "iqr_multiplier": 2.0, "max_physical": 1500},
    "DNI": {"z_score": 3.0, "iqr_multiplier": 1.8, "max_physical": 1100},
    "DHI": {"z_score": 3.2, "iqr_multiplier": 2.2, "max_physical": 400},
}


def detect_outliers_ghi(series: pd.Series) -> pd.Series:
    """Detect outliers in GHI using metric-specific thresholds."""
    if series.empty or series.isna().all():
        return pd.Series(False, index=series.index)
    
    mask = pd.Series(False, index=series.index)
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    if numeric_series.dropna().empty or len(numeric_series.dropna()) < 10:
        return mask
    
    valid_idx = numeric_series.dropna().index
    valid_values = numeric_series.loc[valid_idx]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(valid_values, nan_policy="omit"))
    z_threshold = OUTLIER_THRESHOLDS["GHI"]["z_score"]
    z_outliers = valid_idx[z_scores > z_threshold]
    mask.loc[z_outliers] = True
    
    # IQR method
    Q1 = valid_values.quantile(0.25)
    Q3 = valid_values.quantile(0.75)
    IQR = Q3 - Q1
    multiplier = OUTLIER_THRESHOLDS["GHI"]["iqr_multiplier"]
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    iqr_outliers = series.index[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    mask.loc[iqr_outliers] = True
    
    # Physical maximum
    max_physical = OUTLIER_THRESHOLDS["GHI"]["max_physical"]
    physical_outliers = series.index[numeric_series > max_physical]
    mask.loc[physical_outliers] = True
    
    return mask


def detect_outliers_dni(series: pd.Series) -> pd.Series:
    """Detect outliers in DNI using metric-specific thresholds."""
    if series.empty or series.isna().all():
        return pd.Series(False, index=series.index)
    
    mask = pd.Series(False, index=series.index)
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    if numeric_series.dropna().empty or len(numeric_series.dropna()) < 10:
        return mask
    
    valid_idx = numeric_series.dropna().index
    valid_values = numeric_series.loc[valid_idx]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(valid_values, nan_policy="omit"))
    z_threshold = OUTLIER_THRESHOLDS["DNI"]["z_score"]
    z_outliers = valid_idx[z_scores > z_threshold]
    mask.loc[z_outliers] = True
    
    # IQR method (tighter for DNI)
    Q1 = valid_values.quantile(0.25)
    Q3 = valid_values.quantile(0.75)
    IQR = Q3 - Q1
    multiplier = OUTLIER_THRESHOLDS["DNI"]["iqr_multiplier"]
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    iqr_outliers = series.index[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    mask.loc[iqr_outliers] = True
    
    # Physical maximum
    max_physical = OUTLIER_THRESHOLDS["DNI"]["max_physical"]
    physical_outliers = series.index[numeric_series > max_physical]
    mask.loc[physical_outliers] = True
    
    return mask


def detect_outliers_dhi(series: pd.Series) -> pd.Series:
    """Detect outliers in DHI using metric-specific thresholds."""
    if series.empty or series.isna().all():
        return pd.Series(False, index=series.index)
    
    mask = pd.Series(False, index=series.index)
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    if numeric_series.dropna().empty or len(numeric_series.dropna()) < 10:
        return mask
    
    valid_idx = numeric_series.dropna().index
    valid_values = numeric_series.loc[valid_idx]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(valid_values, nan_policy="omit"))
    z_threshold = OUTLIER_THRESHOLDS["DHI"]["z_score"]
    z_outliers = valid_idx[z_scores > z_threshold]
    mask.loc[z_outliers] = True
    
    # IQR method
    Q1 = valid_values.quantile(0.25)
    Q3 = valid_values.quantile(0.75)
    IQR = Q3 - Q1
    multiplier = OUTLIER_THRESHOLDS["DHI"]["iqr_multiplier"]
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    iqr_outliers = series.index[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    mask.loc[iqr_outliers] = True
    
    # Physical maximum
    max_physical = OUTLIER_THRESHOLDS["DHI"]["max_physical"]
    physical_outliers = series.index[numeric_series > max_physical]
    mask.loc[physical_outliers] = True
    
    return mask


def detect_solar_metric_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Detect outliers for GHI, DNI, and DHI using metric-specific procedures.
    
    Returns:
        Tuple of (outlier_flags_dataframe, combined_outlier_mask)
    """
    outlier_flags = pd.DataFrame(index=df.index)
    combined_mask = pd.Series(False, index=df.index)
    
    if "GHI" in df.columns:
        ghi_outliers = detect_outliers_ghi(df["GHI"])
        outlier_flags["GHI_outlier"] = ghi_outliers
        combined_mask |= ghi_outliers
    
    if "DNI" in df.columns:
        dni_outliers = detect_outliers_dni(df["DNI"])
        outlier_flags["DNI_outlier"] = dni_outliers
        combined_mask |= dni_outliers
    
    if "DHI" in df.columns:
        dhi_outliers = detect_outliers_dhi(df["DHI"])
        outlier_flags["DHI_outlier"] = dhi_outliers
        combined_mask |= dhi_outliers
    
    return outlier_flags, combined_mask


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
        if raw_path.exists():
            df = pd.read_csv(raw_path)
        else:
            # Cloud fallback: try fetching from GitHub raw
            candidate_urls = [f"{GITHUB_RAW_BASE}/{config['clean']}"]
            if "raw" in config:
                candidate_urls.append(f"{GITHUB_RAW_BASE}/{config['raw']}")
            last_error: Exception | None = None
            df = None  # type: ignore[assignment]
            for url in candidate_urls:
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    df = pd.read_csv(io.StringIO(resp.text))
                    break
                except Exception as exc:
                    last_error = exc
            if df is None:
                # Try Dropbox archive fallback (single download containing all CSVs)
                try:
                    _ensure_remote_datasets_from_dropbox()
                except Exception as exc:
                    last_error = exc
                if clean_path.exists():
                    df = pd.read_csv(clean_path)
                elif raw_path.exists():
                    df = pd.read_csv(raw_path)
            if df is None:
                raise FileNotFoundError(
                    f"No dataset available for {country}. Looked at "
                    f"{raw_path} and {', '.join(candidate_urls)}. Last error: {last_error}"
                )
        df = _clean_dataframe(df, apply_outlier_detection=apply_outlier_detection)
        try:
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(clean_path, index=False)
        except Exception:
            # In read-only environments, just skip saving
            pass

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
