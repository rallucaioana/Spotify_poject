import streamlit as st

from utils.load_session_data import ensure_app_data_loaded
from utils.album_features import get_album_feature_summary_split
from utils.album_visualisation import render_album_summary_page
from utils.spotify_api import get_album_cover_data

st.set_page_config(layout="wide")
st.title("Spotify Album Dashboard")

if "app_data" not in st.session_state:
    st.warning("Load the dataset from the Home page first.")
    st.stop()

app_data = ensure_app_data_loaded()

df_final = app_data["df_final"]
album_selector_df = app_data["album_selector_df"]
ambiguous_name_to_id = app_data["ambiguous_name_to_id"]
inconsistent_id_to_name = app_data["inconsistent_id_to_name"]

selected_label = st.selectbox(
    "Select album",
    options=album_selector_df["dropdown_label"].tolist(),
    index=None,
    placeholder="Select an album",
)

if selected_label is not None:
    selected_row = album_selector_df.loc[
        album_selector_df["dropdown_label"] == selected_label
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

    render_album_summary_page(feature_summary_result, cover_data)

    with st.expander("Artist data quality checks"):
        st.write("### Same normalised name with multiple artist IDs")
        if ambiguous_name_to_id.empty:
            st.success("No ambiguous primary artist names found.")
        else:
            st.warning(f"Found {len(ambiguous_name_to_id)} ambiguous cases.")
            st.dataframe(ambiguous_name_to_id, width="stretch")

        st.write("### Same artist ID with multiple name variants")
        if inconsistent_id_to_name.empty:
            st.success("No inconsistent primary artist naming found.")
        else:
            st.warning(f"Found {len(inconsistent_id_to_name)} inconsistent cases.")
            st.dataframe(inconsistent_id_to_name, width="stretch")
            