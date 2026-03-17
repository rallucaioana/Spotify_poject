import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.load_session_data import ensure_app_data_loaded
from utils.helpers import add_era_column, get_theme_colors

PRIMARY_COLOR, PRIMARY_COLOR_FILL = get_theme_colors(fill_alpha=0.45)

st.set_page_config(page_title="Spotify Dashboard", layout="wide")

st.title("Spotify Dashboard")
st.write("Welcome. Explore your Spotify database from the pages in the sidebar.")

data_loaded = "app_data" in st.session_state

if not data_loaded:
    with st.spinner("Loading Spotify data..."):
        ensure_app_data_loaded()
    st.session_state["just_loaded"] = True
    st.rerun()

just_loaded = st.session_state.get("just_loaded", False)

if just_loaded:
    st.success("Data loaded successfully.")
    st.session_state["just_loaded"] = False

app_data = st.session_state["app_data"]
df_final = app_data["df_final"].copy()
album_selector_df = app_data["album_selector_df"].copy()

df_summary = add_era_column(df_final)

total_tracks = len(df_summary)
total_albums = album_selector_df["album_id"].nunique() if "album_id" in album_selector_df.columns else 0
total_artists = df_summary["artist_ids"].nunique() if "artist_ids" in df_summary.columns else 0

release_years = df_summary["release_year"].dropna()
min_year = int(release_years.min()) if not release_years.empty else None
max_year = int(release_years.max()) if not release_years.empty else None
year_range = f"{min_year}–{max_year}" if min_year is not None and max_year is not None else "Unknown"

avg_album_popularity = (
    pd.to_numeric(df_summary["album_popularity"], errors="coerce").mean()
    if "album_popularity" in df_summary.columns else None
)

avg_track_popularity = (
    pd.to_numeric(df_summary["track_popularity"], errors="coerce").mean()
    if "track_popularity" in df_summary.columns else None
)

explicit_tracks = int(df_summary["explicit"].sum()) if "explicit" in df_summary.columns else 0

top_era = (
    df_summary["era"].dropna().mode().iloc[0]
    if "era" in df_summary.columns and not df_summary["era"].dropna().empty
    else "Unknown"
)

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total tracks", f"{total_tracks:,}")
col2.metric("Albums", f"{total_albums:,}")
col3.metric("Artists", f"{total_artists:,}")
col4.metric("Release span", year_range)

col5, col6, col7, col8 = st.columns(4)
col5.metric(
    "Avg album popularity",
    f"{avg_album_popularity:.1f}" if avg_album_popularity is not None else "N/A",
)
col6.metric(
    "Avg track popularity",
    f"{avg_track_popularity:.1f}" if avg_track_popularity is not None else "N/A",
)
col7.metric("Most common era", top_era)

col8.metric("Explicit tracks", f"{explicit_tracks:,}")

album_era_df = (
    df_summary.loc[df_summary["album_id"].notna(), ["album_id", "era"]]
    .drop_duplicates(subset=["album_id"])
    .dropna(subset=["era"])
    .copy()
)

if not album_era_df.empty:
    era_counts = (
        album_era_df.groupby("era")
        .size()
        .reset_index(name="albums")
    )
    era_counts["era_sort"] = era_counts["era"].str.replace("s", "", regex=False).astype(int)
    era_counts = era_counts.sort_values("era_sort")

    fig = go.Figure()

    # barplot
    fig.add_trace(
        go.Bar(
            x=era_counts["era"],
            y=era_counts["albums"],
            marker=dict(
                color=PRIMARY_COLOR_FILL,
                line=dict(color=PRIMARY_COLOR, width=2),
            ),
            hovertemplate="Era: %{x}<br>Albums: %{y}<extra></extra>",
            name="Albums",
        )
    )

    # trendline
    fig.add_trace(
        go.Scatter(
            x=era_counts["era"],
            y=era_counts["albums"],
            mode="lines+markers",
            line=dict(
                color=PRIMARY_COLOR,
                width=3,
                shape="spline",
            ),
            marker=dict(
                size=6,
                color=PRIMARY_COLOR,
            ),
            hoverinfo="skip",
            name="Trend",
        )
    )

    fig.update_layout(
        title="Albums by era",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        xaxis=dict(
            title="Era",
            tickfont=dict(color="#B3B3B3"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Albums",
            tickfont=dict(color="#B3B3B3"),
            gridcolor="#333333",
            zeroline=False,
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
    )

    st.plotly_chart(fig, width="stretch")


if {"primary_artist_name", "artist_ids", "album_id", "album_type", "artist_popularity"}.issubset(df_summary.columns):
    artist_overview_df = (
        df_summary.loc[
            df_summary["album_id"].notna()
            & df_summary["primary_artist_name"].notna()
            & df_summary["artist_ids"].notna()
        ,
            ["primary_artist_name", "artist_ids", "album_id", "album_type", "artist_popularity"]
        ]
        .copy()
    )

    # artist_overview_df["album_type"] = (
    #     artist_overview_df["album_type"]
    #     .astype(str)
    #     .str.strip()
    #     .str.lower()
    # )

    # # keep only albums, not singles
    # artist_overview_df = artist_overview_df.loc[
    #     artist_overview_df["album_type"] == "album"
    # ].copy()

    # collapse repeated track-level rows so each album counts once per artist
    # sadly re-releases are still counted
    artist_overview_df = artist_overview_df.drop_duplicates(
        subset=["artist_ids", "album_id"]
    )

    top_artists_df = (
        artist_overview_df.groupby(["artist_ids", "primary_artist_name"], dropna=False)
        .agg(
            spotify_releases=("album_id", "nunique"),
            artist_popularity=("artist_popularity", "max"),
        )
        .reset_index()
        .sort_values(
            ["spotify_releases", "artist_popularity", "primary_artist_name"],
            ascending=[False, False, True],
        )
        .head(10)
        .reset_index(drop=True)
    )

    top_artists_df = top_artists_df.rename(
        columns={
            "primary_artist_name": "Artist",
            "spotify_releases": "Spotify releases",
            "artist_popularity": "Artist popularity",
        }
    )

    top_artists_df["Artist popularity"] = (
        pd.to_numeric(top_artists_df["Artist popularity"], errors="coerce")
        .round(0)
        .astype("Int64")
    )

    st.subheader("Top artists by Spotify release count")

    st.dataframe(
        top_artists_df[["Artist", "Spotify releases", "Artist popularity"]],
        width="stretch",
        hide_index=True,
    )
    
