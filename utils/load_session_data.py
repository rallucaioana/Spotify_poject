import streamlit as st
import pandas as pd

from utils.data_loader import load_raw_spotify_data
from utils.preprocessing import build_clean_dataset
from utils.artist_audits import (
    build_artist_quality_report,
    audit_artist_name_to_id,
    audit_artist_id_to_name,
    build_reviewed_override_table,
)


@st.cache_data(show_spinner=False)
def build_album_selector_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "album_id",
        "album_name",
        "primary_artist_name",
        "album_popularity",
        "release_date",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    optional_cols = [
        c for c in ["album_type", "label", "total_tracks", "release_date_display"]
        if c in df.columns
    ]

    selector_df = (
        df.loc[
            df["album_id"].notna()
            & df["album_name"].notna()
            & df["primary_artist_name"].notna(),
            ["album_id", "album_name", "primary_artist_name", "album_popularity", "release_date"] + optional_cols,
        ]
        .copy()
        .drop_duplicates(subset=["album_id"])
        .sort_values(
            by=["album_popularity", "primary_artist_name", "album_name", "release_date"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )

    if "release_date_display" not in selector_df.columns:
        selector_df["release_date_display"] = "Unknown"

    selector_df["album_popularity_display"] = (
        pd.to_numeric(selector_df["album_popularity"], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
    )

    selector_df["dropdown_label"] = (
        selector_df["primary_artist_name"].astype(str).str.strip()
        + " - "
        + selector_df["album_name"].astype(str).str.strip()
        + " ("
        + selector_df["release_date_display"]
        + ")"
        + " | Popularity: "
        + selector_df["album_popularity_display"].astype(str)
    )

    return selector_df


@st.cache_data(show_spinner=False)
def load_app_data() -> dict:
    df_raw = load_raw_spotify_data()

    artist_quality_raw = build_artist_quality_report(df_raw)
    df_fixed = artist_quality_raw["df_fixed"]

    df_final = build_clean_dataset(df_fixed)

    # Run audits directly on the cleaned data, no need to re-run the full
    # resolution pipeline since df_final already has corrected artist IDs.
    album_selector_df = build_album_selector_df(df_final)

    return {
        "df_final": df_final,
        "album_selector_df": album_selector_df,
        "ambiguous_name_to_id": audit_artist_name_to_id(df_final),
        "inconsistent_id_to_name": audit_artist_id_to_name(df_final),
        "reviewed_artist_name_overrides": build_reviewed_override_table(),
        "auto_resolved_artist_ids": artist_quality_raw["auto_resolved_artist_ids"],
    }


def ensure_app_data_loaded() -> dict:
    if "app_data" not in st.session_state:
        st.session_state["app_data"] = load_app_data()
    return st.session_state["app_data"]