import streamlit as st
import pandas as pd

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
reviewed_artist_name_overrides = app_data["reviewed_artist_name_overrides"]
auto_resolved_artist_ids = app_data["auto_resolved_artist_ids"]

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

        st.write("### Auto-resolved artist IDs")
        st.caption(
            "Cases where artist IDs were unified automatically each time the data loads. "
            "**Dominance**: one ID accounted for ≥ 60% of rows. "
            "**Tiebreak**: exactly 2 IDs with no dominant one — alphabetically-first ID chosen "
            "(two Spotify profiles for the same artist is the overwhelmingly likely explanation). "
            "To override a specific case, add an entry to "
            "`REVIEWED_ARTIST_NAME_OVERRIDES` in `utils/artist_audits.py`."
        )
        if auto_resolved_artist_ids.empty:
            st.success("No artist IDs were auto-resolved.")
        else:
            display_cols = {
                "artist_name_key": "Artist name key",
                "canonical_artist_id": "Canonical artist ID",
                "dominant_row_count": "Dominant ID rows",
                "total_row_count": "Total rows",
                "dominance_ratio": "Dominance ratio",
                "artist_id_count": "Original ID count",
                "remapped_row_count": "Rows remapped",
            }

            def _format_resolved(df_in):
                available = [c for c in display_cols if c in df_in.columns]
                out = df_in[available].rename(columns=display_cols).sort_values(
                    "Total rows", ascending=False
                )
                if "Dominance ratio" in out.columns:
                    out["Dominance ratio"] = out["Dominance ratio"].map("{:.1%}".format)
                return out

            has_method_col = "resolution_method" in auto_resolved_artist_ids.columns

            dominance_df = (
                auto_resolved_artist_ids[
                    auto_resolved_artist_ids["resolution_method"] == "dominance"
                ]
                if has_method_col
                else auto_resolved_artist_ids
            )
            tiebreak_df = (
                auto_resolved_artist_ids[
                    auto_resolved_artist_ids["resolution_method"] == "tiebreak"
                ]
                if has_method_col
                else pd.DataFrame()
            )

            if not dominance_df.empty:
                st.markdown(f"**Dominance** — {len(dominance_df)} group(s)")
                st.dataframe(_format_resolved(dominance_df), width="stretch", hide_index=True)

            if not tiebreak_df.empty:
                st.markdown(f"**Tiebreak** (2-ID, no dominant) — {len(tiebreak_df)} group(s)")
                st.dataframe(_format_resolved(tiebreak_df), width="stretch", hide_index=True)

        st.write("### Remaining: same normalised name with multiple artist IDs")
        st.caption(
            "These are groups with 3+ distinct artist IDs where no single ID was dominant. "
            "They are most likely genuinely different artists sharing a name (e.g. 'Traditional', "
            "'Sully', 'Breezy'). Review and add entries to `REVIEWED_ARTIST_NAME_OVERRIDES` "
            "only if you're certain two IDs refer to the same artist."
        )
        if ambiguous_name_to_id.empty:
            st.success("No ambiguous primary artist names found.")
        else:
            st.warning(f"Found {len(ambiguous_name_to_id)} unresolved ambiguous case(s).")
            st.dataframe(ambiguous_name_to_id, width="stretch")

        st.write("### Same artist ID with multiple name variants")
        if inconsistent_id_to_name.empty:
            st.success("No inconsistent primary artist naming found.")
        else:
            st.warning(f"Found {len(inconsistent_id_to_name)} inconsistent case(s).")
            st.dataframe(inconsistent_id_to_name, width="stretch")

        st.write("### Reviewed artist-name overrides")
        if reviewed_artist_name_overrides.empty:
            st.info("No reviewed artist-name overrides configured.")
        else:
            st.dataframe(reviewed_artist_name_overrides, width="stretch")