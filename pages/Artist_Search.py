import streamlit as st
import pandas as pd

from utils.load_session_data import ensure_app_data_loaded
from utils.artist_search import build_artist_selector_df, get_artist_overview, get_artist_releases
from utils.album_visualisation import render_collaborators
from utils.spotify_api import get_album_covers_batch, get_artist_profile_data, render_rate_limit_warning

st.set_page_config(layout="wide")
st.title("Artist Search")

with st.spinner("Loading data from database..."):
    app_data = ensure_app_data_loaded()
    
df_final = app_data["df_final"]

render_rate_limit_warning()

artist_selector_df = build_artist_selector_df(df_final)

# Read ID from URL on load
preselected_artist_id = st.query_params.get("artist_id", None)

default_artist_index = None
if preselected_artist_id:
    matching = artist_selector_df.loc[
        artist_selector_df["artist_ids"].astype(str).str.strip() == str(preselected_artist_id).strip()
    ]
    if not matching.empty:
        label = matching.iloc[0]["dropdown_label"]
        try:
            default_artist_index = artist_selector_df["dropdown_label"].tolist().index(label)
        except ValueError:
            default_artist_index = None

selected_label = st.sidebar.selectbox(
    "Search for an artist",
    options=artist_selector_df["dropdown_label"].tolist(),
    index=default_artist_index,
    placeholder="Search by name...",
)

if selected_label is None:
    st.stop()

selected_row = artist_selector_df.loc[
    artist_selector_df["dropdown_label"] == selected_label
].iloc[0]

artist_id = selected_row["artist_ids"]

# Write selected artist to URL only if changed
if st.query_params.get("artist_id") != artist_id:
    st.query_params["artist_id"] = artist_id

with st.spinner("Loading artist data..."):
    overview = get_artist_overview(df_final, artist_id)
    releases_df = get_artist_releases(df_final, artist_id)
    
    # checks for being rate-limited before calling the spotify API for the artist's profile picture
    if st.session_state.get("spotify_rate_limited", False):
        profile = {"image_url": None, "spotify_url": None, "name": None}
    else:
        try:
            profile = get_artist_profile_data(artist_id)
        except Exception:
            profile = {"image_url": None, "spotify_url": None, "name": None}


img_col, stats_col = st.columns([1, 3], vertical_alignment="top")

with img_col:
    if profile and profile.get("image_url"):
        st.image(profile["image_url"], width="stretch")
    else:
        st.markdown(
            "<div style='"
            "width:100%; aspect-ratio:1/1;"
            "background:#232323;"
            "border-radius:50%;"
            "display:flex; align-items:center; justify-content:center;"
            "font-size:4rem;"
            "'>👤</div>",
            unsafe_allow_html=True,
        )

    if profile and profile.get("spotify_url"):
        st.link_button(
            "Open artist on Spotify",
            profile["spotify_url"],
            type="primary",
            width="stretch",
            icon=":material/headphones:",
        )

with stats_col:
    st.markdown(
        f"<div style='font-size:3rem; font-weight:700; line-height:1.1; margin-bottom:1.5rem;'>"
        f"{overview['primary_artist_name']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        followers = overview["followers"]
        st.metric(
            "Followers",
            f"{int(followers):,}" if pd.notna(followers) else "N/A",
        )

    with c2:
        pop = overview["artist_popularity"]
        st.metric(
            "Popularity",
            f"{pop:.1f}" if pd.notna(pop) else "N/A",
        )

    with c3:
        first = overview["first_release_year"]
        last = overview["last_release_year"]
        if first and last and first != last:
            years_label = f"{first}–{last}"
        elif first:
            years_label = str(first)
        else:
            years_label = "Unknown"
        st.metric("Years active", years_label)

    releases_by_type = overview["releases_by_type"]
    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric("Albums", releases_by_type.get("album", 0))

    with c5:
        st.metric("Singles", releases_by_type.get("single", 0))

    with c6:
        st.metric("Tracks", overview["total_tracks"])

    genres = overview["genres"]
    if genres:
        render_collaborators(genres, title="Genres")


st.markdown("---")


# releases grid
st.markdown("### Releases")

if releases_df.empty:
    st.info("No releases found for this artist.")
    st.stop()
    
    
# releases filter options
available_types = sorted(
    releases_df["album_type"].dropna().astype(str).str.strip().str.capitalize().unique().tolist()
)

f1, f2 = st.columns(2)

with f1:
    selected_types = st.multiselect(
        "Release type",
        options=available_types,
        default=available_types,
    )

with f2:
    sort_order = st.selectbox(
        "Sort by",
        options=["Date (newest first)", "Date (oldest first)", "Popularity (high to low)", "Popularity (low to high)", "Name (A–Z)"],
    )

filtered_releases = releases_df[
    releases_df["album_type"].astype(str).str.strip().str.capitalize().isin(selected_types)
].copy()

if sort_order == "Date (newest first)":
    filtered_releases = filtered_releases.sort_values("release_date", ascending=False)
elif sort_order == "Date (oldest first)":
    filtered_releases = filtered_releases.sort_values("release_date", ascending=True)
elif sort_order == "Popularity (high to low)":
    filtered_releases = filtered_releases.sort_values("album_popularity", ascending=False)
elif sort_order == "Popularity (low to high)":
    filtered_releases = filtered_releases.sort_values("album_popularity", ascending=True)
elif sort_order == "Name (A–Z)":
    filtered_releases = filtered_releases.sort_values("album_name", ascending=True)

filtered_releases = filtered_releases.reset_index(drop=True)

if filtered_releases.empty:
    st.info("No releases match the selected filters.")
    st.stop()


# fetch all covers in batched API calls before rendering anything,
# so tiles all appear at once rather than popping in one by one.
# skip API calls if rate limited
if st.session_state.get("spotify_rate_limited", False):
    covers = {}
else:
    try:
        with st.spinner("Fetching album covers..."):
            album_ids = tuple(filtered_releases["album_id"].astype(str).tolist())
            covers = get_album_covers_batch(album_ids)
    except Exception:
        covers = {}


TILES_PER_ROW = 4

for row_start in range(0, len(filtered_releases), TILES_PER_ROW):
    row_slice = filtered_releases.iloc[row_start : row_start + TILES_PER_ROW]
    cols = st.columns(TILES_PER_ROW)

    for col, (_, release) in zip(cols, row_slice.iterrows()):
        album_id = release["album_id"]
        album_name = release.get("album_name", "Unknown")
        album_type = str(release.get("album_type", "")).strip().capitalize()
        release_year = (
            release["release_date"].year
            if pd.notna(release.get("release_date"))
            else None
        )
        popularity = release.get("album_popularity", None)

        cover = covers.get(str(album_id), {})

        with col:
            image_url = cover.get("image_url") if cover else None
            image_html = (
                f"<div style='width:100%; height:250px; background:transparent; border-radius:4px; "
                f"display:flex; align-items:center; justify-content:center;'>"
                f"<img src='{image_url}' style='max-width:100%; max-height:250px; object-fit:contain; border-radius:4px;'/>"
                f"</div>"
                if image_url else
                f"<div style='"
                f"width:100%; height:250px;"
                f"background:#232323;"
                f"border-radius:4px;"
                f"display:flex; align-items:center; justify-content:center;"
                f"padding:1rem;"
                f"text-align:center;"
                f"font-size:0.95rem;"
                f"font-weight:700;"
                f"color:#B3B3B3;"
                f"'>{album_name}</div>"
            )
            st.markdown(image_html, unsafe_allow_html=True)

            st.markdown(
                f"<div style='font-weight:700; font-size:0.95rem; "
                f"margin-top:0.5rem; line-height:1.3; "
                f"height:2.6rem; overflow:hidden; display:-webkit-box; "
                f"-webkit-line-clamp:2; -webkit-box-orient:vertical;'>"
                f"{album_name}"
                f"</div>",
                unsafe_allow_html=True,
            )

            track_count = release.get("track_count", None)
            track_str = f"{int(track_count)} {'track' if int(track_count) == 1 else 'tracks'}" if pd.notna(track_count) and track_count is not None else None
            meta_parts = [p for p in [album_type, str(release_year) if release_year else None, track_str] if p]
            st.caption(" · ".join(meta_parts))

            if pd.notna(popularity):
                st.progress(
                    int(popularity) / 100,
                    text=f"Popularity: {int(popularity)}",
                )

            st.markdown(
                f"<a href='/Album_Dashboard?album_id={album_id}' target='_self'>"
                f"<button style='"
                f"background:transparent;"
                f"border:1px solid rgba(255,255,255,0.2);"
                f"border-radius:0.5rem;"
                f"color:white;"
                f"padding:0.4rem 0.8rem;"
                f"cursor:pointer;"
                f"font-size:0.875rem;"
                f"'>"
                f"Open in Album Dashboard"
                f"</button></a>",
                unsafe_allow_html=True,
            )

    st.markdown("")