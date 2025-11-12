from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

# Robust imports to support both package and script execution
# Prioritize local utils within this repository to avoid conflicts with similarly named packages
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import (  # type: ignore
        AVAILABLE_COUNTRIES,
        SOLAR_COLS,
        country_palette,
        load_combined_dataset,
        summarise_metrics,
        top_regions,
    )
except Exception:
    from app.utils import (
        AVAILABLE_COUNTRIES,
        SOLAR_COLS,
        country_palette,
        load_combined_dataset,
        summarise_metrics,
        top_regions,
    )

st.set_page_config(
    page_title="Solar Potential Dashboard",
    page_icon="☀️",
    layout="wide",
)

# Minimal CSS polish for cleaner first paint
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: .2px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("☀️ Solar Potential Dashboard")
st.markdown(
    """
Explore solar irradiance characteristics across Benin, Sierra Leone, and Togo.
Use the controls in the sidebar to filter countries, constrain the analysis window,
choose metrics, and surface priority locations for downstream planning.
"""
)

# Sidebar configuration -----------------------------------------------------------------

st.sidebar.header("Data Upload")
st.sidebar.markdown("Upload CSV files for analysis:")

# File uploader for data files
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload benin_clean.csv, sierraleone_clean.csv, and/or togo_clean.csv"
)

# Store uploaded files in session state
if uploaded_files:
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = {}
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        # Map filenames to countries
        if "benin" in filename:
            st.session_state.uploaded_data["Benin"] = uploaded_file
        elif "sierraleone" in filename or "sierra" in filename:
            st.session_state.uploaded_data["Sierra Leone"] = uploaded_file
        elif "togo" in filename:
            st.session_state.uploaded_data["Togo"] = uploaded_file
    
    if st.session_state.uploaded_data:
        st.sidebar.success(f"✅ {len(st.session_state.uploaded_data)} file(s) uploaded")

st.sidebar.header("Controls")
selected_countries: Iterable[str] = st.sidebar.multiselect(
    "Countries",
    options=AVAILABLE_COUNTRIES,
    default=AVAILABLE_COUNTRIES,
)

metric = st.sidebar.selectbox(
    "Irradiance Metric",
    options=SOLAR_COLS,
    index=0,
    format_func=lambda value: f"{value} (W/m²)",
)

if not selected_countries:
    st.warning("Please select at least one country to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def get_data(countries: tuple[str, ...], uploaded_files: dict | None = None) -> pd.DataFrame:
    return load_combined_dataset(countries, uploaded_files=uploaded_files)

# Get uploaded files from session state
uploaded_data = st.session_state.get("uploaded_data", {})

# Defensive load with empty-state guidance
try:
    with st.spinner("Loading datasets..."):
        raw_df = get_data(tuple(selected_countries), uploaded_files=uploaded_data if uploaded_data else None)
except FileNotFoundError as exc:
    if not uploaded_data:
        st.error("⚠️ No data files available")
        st.info(
            """
            **Please upload CSV files using the file uploader in the sidebar.**
            
            Expected files:
            - `benin_clean.csv` or `benin-malanville.csv`
            - `sierraleone_clean.csv` or `sierraleone-bumbuna.csv`
            - `togo_clean.csv` or `togo-dapaong_qc.csv`
            
            Alternatively, if running locally, place these files in the `data/` directory.
            """
        )
    else:
        st.error(f"Error loading data: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Error processing data: {exc}")
    st.stop()

palette_map = country_palette(selected_countries)
category_order = list(selected_countries)

# Date filtering ------------------------------------------------------------------------

def filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
        return df

    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()
    default_range = (min_date, max_date)

    date_range = st.sidebar.date_input(
        "Date Range",
        value=default_range,
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )

    # Clamp any out-of-bounds selections automatically
    try:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range
            end_date = date_range
    except Exception:
        start_date, end_date = default_range

    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date

    start_dt = dt.datetime.combine(start_date, dt.time.min)
    end_dt = dt.datetime.combine(end_date, dt.time.max)

    mask = df["Timestamp"].between(start_dt, end_dt)
    return df.loc[mask]

filtered_df = filter_by_date(raw_df)

if filtered_df.empty:
    st.warning("No records available for the current filters. Adjust the date range or country selection.")
    st.stop()

# Resampling controls -------------------------------------------------------------------

FREQUENCIES = {
    "Hourly": "H",
    "Daily": "D",
    "Weekly": "W",
}

def resample_metric(df: pd.DataFrame, freq_label: str, metric_name: str) -> pd.DataFrame:
    if "Timestamp" not in df.columns or df["Timestamp"].isna().all():
        return pd.DataFrame()

    rule = FREQUENCIES[freq_label]
    resampled = (
        df.set_index("Timestamp")
        .groupby("country")[metric_name]
        .resample(rule)
        .mean()
        .reset_index()
        .dropna(subset=[metric_name])
    )
    return resampled

freq_choice = st.sidebar.selectbox("Aggregate to", options=list(FREQUENCIES.keys()), index=1)

# KPI ribbon ---------------------------------------------------------------------------

summary = summarise_metrics(filtered_df)
metric_columns = st.columns(len(selected_countries))

for col, country in zip(metric_columns, selected_countries):
    country_row = summary.loc[summary["country"] == country]
    if country_row.empty:
        continue
    mean_value = country_row.iloc[0][f"{metric} mean"]
    std_value = country_row.iloc[0][f"{metric} std"]
    col.metric(
        label=f"Avg {metric} – {country}",
        value=f"{mean_value:.1f} W/m²",
        delta=f"σ {std_value:.1f}",
    )

# Tabs for visuals ----------------------------------------------------------------------

tab_box, tab_trend, tab_table = st.tabs([
    "Distribution",
    "Trend",
    "Top Regions",
])

with tab_box:
    fig = px.box(
        filtered_df,
        x="country",
        y=metric,
        color="country",
        category_orders={"country": category_order},
        color_discrete_map=palette_map,
        points="suspectedoutliers",
    )
    fig.update_traces(marker=dict(opacity=0.95), line=dict(width=2))
    fig.update_layout(
        title=f"{metric} distribution",
        yaxis_title=f"{metric} (W/m²)",
        xaxis_title="Country",
        legend_title="Country",
    )
    st.plotly_chart(fig, width='stretch')

with tab_trend:
    resampled_df = resample_metric(filtered_df, freq_choice, metric)
    if resampled_df.empty:
        st.info("Trend view is unavailable because timestamps are missing or invalid.")
    else:
        fig = px.line(
            resampled_df,
            x="Timestamp",
            y=metric,
            color="country",
            category_orders={"country": category_order},
            color_discrete_map=palette_map,
        )
        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(
            title=f"{metric} trend ({freq_choice.lower()})",
            yaxis_title=f"{metric} (W/m²)",
            xaxis_title="Timestamp",
            legend_title="Country",
        )
        st.plotly_chart(fig, width='stretch')

with tab_table:
    top_n = st.slider("Top locations", min_value=3, max_value=10, value=5)
    top_table = top_regions(filtered_df, metric, top_n=top_n)
    st.dataframe(top_table, width='stretch')

# Summary table -------------------------------------------------------------------------

with st.expander("Summary statistics", expanded=True):
    st.dataframe(summary, width='stretch')

if uploaded_data:
    st.caption("Data source: uploaded CSV files")
else:
    st.caption("Data source: local CSV files stored in the project's data/ directory.")
