import streamlit as st
import pandas as pd

from utils.data_loader import load_raw_spotify_data
from utils.preprocessing import build_clean_dataset
from utils.album_features import get_album_feature_summary_split
from utils.album_visualisation import render_album_summary_page
from utils.spotify_api import get_album_cover_data
from utils.artist_audits import build_artist_quality_report


st.set_page_config(layout="wide")
st.title("Spotify Album Dashboard")


@st.cache_data
def load_dataset():
    df_raw = load_raw_spotify_data()

    artist_quality = build_artist_quality_report(df_raw)
    df_fixed = artist_quality["df_fixed"]

    df_final = build_clean_dataset(df_fixed)

    post_clean_quality = build_artist_quality_report(df_final)

    ambiguous_name_to_id = post_clean_quality["ambiguous_name_to_id"]
    inconsistent_id_to_name = post_clean_quality["inconsistent_id_to_name"]

    return df_final, ambiguous_name_to_id, inconsistent_id_to_name


@st.cache_data
def build_album_selector_df(df):
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

    optional_cols = [c for c in ["album_type", "label", "total_tracks", "release_date_display"] if c in df.columns]

    selector_df = (
        df.loc[
            df["album_id"].notna()
            & df["album_name"].notna()
            & df["primary_artist_name"].notna(),
            ["album_id", "album_name", "primary_artist_name", "album_popularity", "release_date"] + optional_cols
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

    selector_df["label"] = (
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



df_final, ambiguous_name_to_id, inconsistent_id_to_name = load_dataset()
album_selector_df = build_album_selector_df(df_final)

selected_label = st.selectbox(
    "Select album",
    options=album_selector_df["label"].tolist(),
    index=None,
    placeholder="Select an album",
)

if selected_label is not None:

    selected_row = album_selector_df.loc[
        album_selector_df["label"] == selected_label
    ].iloc[0]

    selected_album_id = selected_row["album_id"]

    with st.spinner("Loading album data..."):

        cover_data = get_album_cover_data(selected_album_id)

        track_table_df = (
            df_final.loc[
                df_final["album_id"].astype(str).str.strip() == str(selected_album_id).strip()
            ]
            .sort_values(["track_number", "track_name"], ascending=[True, True])
            .copy()
        )

        feature_summary_result = get_album_feature_summary_split(
            df=df_final,
            album_id=selected_album_id,
            track_df=track_table_df,
        )

        feature_summary_result = get_album_feature_summary_split(
            df=df_final,
            album_id=selected_album_id,
            track_df=track_table_df,
        )

    render_album_summary_page(
        feature_summary_result,
        cover_data
    )
    
    with st.expander("Artist data quality checks"):
        st.write("### Same normalised name with multiple artist IDs")
        if ambiguous_name_to_id.empty:
            st.success("No ambiguous primary artist names found.")
        else:
            st.warning(f"Found {len(ambiguous_name_to_id)} ambiguous cases.")
            st.dataframe(ambiguous_name_to_id, use_container_width=True)

        st.write("### Same artist ID with multiple name variants")
        if inconsistent_id_to_name.empty:
            st.success("No inconsistent primary artist naming found.")
        else:
            st.warning(f"Found {len(inconsistent_id_to_name)} inconsistent cases.")
            st.dataframe(inconsistent_id_to_name, use_container_width=True)

