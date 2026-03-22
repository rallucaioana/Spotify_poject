import streamlit as st
import pandas as pd

from utils.load_session_data import ensure_app_data_loaded
from utils.album_features import get_album_feature_summary_split
from utils.album_visualisation import render_album_summary_page
from utils.spotify_api import get_album_cover_data, render_rate_limit_warning

st.set_page_config(layout="wide")
st.title("Spotify Album Dashboard")

with st.spinner("Loading data from database..."):
    app_data = ensure_app_data_loaded()

render_rate_limit_warning()

df_final = app_data["df_final"]
album_selector_df = app_data["album_selector_df"]
ambiguous_name_to_id = app_data["ambiguous_name_to_id"]
inconsistent_id_to_name = app_data["inconsistent_id_to_name"]
reviewed_artist_name_overrides = app_data["reviewed_artist_name_overrides"]
auto_resolved_artist_ids = app_data["auto_resolved_artist_ids"]

# Pre-selection support
# Prefer session state (navigating from artist page),
# fall back to URL param (browser refresh).
url_album_id = st.query_params.get("album_id", None)
preselected_album_id = st.session_state.pop("preselected_album_id", None) or url_album_id

default_index = None
if preselected_album_id is not None:
    matching = album_selector_df.loc[
        album_selector_df["album_id"].astype(str).str.strip()
        == str(preselected_album_id).strip()
    ]
    if not matching.empty:
        label = matching.iloc[0]["dropdown_label"]
        try:
            default_index = album_selector_df["dropdown_label"].tolist().index(label)
        except ValueError:
            default_index = None

# Album selector
selected_label = st.selectbox(
    "Select album",
    options=album_selector_df["dropdown_label"].tolist(),
    index=default_index,
    placeholder="Select an album",
)

if selected_label is not None:
    selected_row = album_selector_df.loc[
        album_selector_df["dropdown_label"] == selected_label
    ].iloc[0]

    selected_album_id = selected_row["album_id"]

    with st.spinner("Loading album data..."):
        if st.session_state.get("spotify_rate_limited", False):
            cover_data = {"image_url": None, "spotify_url": None, "album_name": None}
        else:
            try:
                cover_data = get_album_cover_data(selected_album_id)
            except Exception:
                cover_data = {"image_url": None, "spotify_url": None, "album_name": None}

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

    render_album_summary_page(feature_summary_result, cover_data)