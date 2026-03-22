import streamlit as st
import pandas as pd
import numpy as np
import html
import plotly.graph_objects as go

from utils.helpers import get_theme_colors

PRIMARY_COLOR, PRIMARY_COLOR_FILL = get_theme_colors(fill_alpha=0.35)

def render_collaborators(artists, title="Collaborating Artists"):
    if not artists:
        return

    st.markdown(f"**{title}**")

    tags_html = "".join(
        (
            "<span style='"
            "display:inline-flex;"
            "align-items:center;"
            "padding:0.38rem 0.85rem;"
            "border-radius:999px;"
            "background:#232323;"
            "border:1px solid #333333;"
            "color:#FFFFFF;"
            "font-size:0.9rem;"
            "font-weight:600;"
            "white-space:nowrap;"
            "'>"
            f"{html.escape(str(artist))}"
            "</span>"
        )
        for artist in artists
    )


    container_html = (
        "<div style='"
        "display:flex;"
        "flex-wrap:wrap;"
        "gap:0.6rem;"
        "margin-top:0.35rem;"
        "padding-bottom:0.5rem;"
        "'>"
        f"{tags_html}"
        "</div>"
    )

    st.markdown(container_html, unsafe_allow_html=True)



def render_album_header(summary_result, cover_data):
    metadata = summary_result["metadata_df"].iloc[0]

    cover_col, meta_col = st.columns([1, 2.6], vertical_alignment="top")


    with cover_col:
        if cover_data and cover_data.get("image_url"):
            st.image(cover_data["image_url"], width="stretch")
        else:
            st.markdown(
                "<div style='"
                "width:100%; aspect-ratio:1/1;"
                "background:#232323;"
                "border-radius:4px;"
                "display:flex; align-items:center; justify-content:center;"
                "font-size:5rem;"
                "'>💿</div>",
                unsafe_allow_html=True,
            )

        if cover_data and cover_data.get("spotify_url"):
            st.link_button("Open album on Spotify", cover_data["spotify_url"], type="primary",
                        use_container_width=True, icon=":material/headphones:")


    with meta_col:
        st.markdown(
            f"""
            <div style="margin-bottom:0rem;">
                <span style="font-style:italic; opacity:0.8;">
                    {str(metadata['album_type']).capitalize()}
                </span>
            </div>
            <div style="font-size:3rem; font-weight:700; line-height:1.1; margin-bottom: 1rem, margin-top: 0rem">
                {metadata['album_name']}
            </div>
            """,
            unsafe_allow_html=True
        )


        top_left, top_right = st.columns(2)

        with top_left:
            st.markdown("**Primary artist**")
            st.markdown(
                f"<div style='font-size:2.2rem; font-weight:400; line-height:1.15; margin-bottom:1rem;'>"
                f"{metadata['primary_artist_name']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with top_right:
            st.markdown("**Release date**")
            st.markdown(
                f"<div style='font-size:1.5rem; font-weight:400; line-height:1.2;'>"
                f"{metadata['release_date_display']}"
                f"</div>",
                unsafe_allow_html=True,
            )
            
        # differing display logic depending on collaborating artists being present or not
        collaborators = metadata.get("collaborating_artists", None)
        if collaborators:

            meta_a, meta_b = st.columns(2)

            with meta_a:
                st.markdown("**Label**")
                st.markdown(
                    f"<div style='font-size:1.5rem; font-weight:400; line-height:1.2; word-break:break-word;'>"
                    f"{metadata['label']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with meta_b:
                genres = metadata.get("primary_artist_genres", None)
                if genres:
                    render_collaborators(genres, title="Artist Genres")

            
            render_collaborators(collaborators)
                
        else:
            meta_a, meta_b = st.columns(2)
            
            with meta_a:
                st.markdown("**Label**")
                st.markdown(
                    f"<div style='font-size:1.5rem; font-weight:400; line-height:1.2; word-break:break-word;'>"
                    f"{metadata['label']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with meta_b:
                genres = metadata.get("primary_artist_genres", None)
                if genres:
                    render_collaborators(genres, title="Artist Genres")
            



def render_album_feature_summary(feature_summary_result):
    metadata_df = feature_summary_result["metadata_df"]
    popularity_df = feature_summary_result["popularity_df"]
    audio_features_df = feature_summary_result["audio_features_df"]
    track_df = feature_summary_result["track_df"]

    metadata = metadata_df.iloc[0]
    popularity = popularity_df.iloc[0]
    audio = audio_features_df.iloc[0]

    st.markdown("### Album details")
    
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Release date", metadata["release_date_display"])

    with c2:
        total_minutes = metadata["total_duration_min"]

        hours = int(total_minutes // 60)
        minutes = int(round(total_minutes % 60))

        if hours > 0:
            duration_display = f"{hours}h {minutes}min"
        else:
            duration_display = f"{minutes}min"

        st.metric("Duration", duration_display)
        
    with c3:
        st.metric("Tracks", int(metadata["track_count"]))
        
    with c4:
        st.metric("Explicit tracks", int(metadata["explicit_track_count"]))
        

    st.markdown("### Popularity")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Album popularity", f"{popularity['album_popularity']:.1f}")

    with c2:
        st.metric("Avg track popularity", f"{popularity['avg_track_popularity']:.1f}")

    with c3:
        st.metric("Primary artist popularity", f"{popularity['primary_artist_popularity']:.1f}")

    with c4:
        if pd.notna(popularity["primary_artist_followers"]):
            st.metric("Primary artist followers", f"{popularity['primary_artist_followers']:,.0f}")
        else:
            st.metric("Primary artist followers", "N/A")

    st.markdown("---")

    st.markdown("### Audio features")

    audio_chart_df = pd.DataFrame({
        "feature": [
            "Danceability",
            "Energy",
            "Speechiness",
            "Acousticness",
            "Instrumentalness",
            "Liveness",
            "Valence",
        ],
        "value": [
            audio["avg_danceability"],
            audio["avg_energy"],
            audio["avg_speechiness"],
            audio["avg_acousticness"],
            audio["avg_instrumentalness"],
            audio["avg_liveness"],
            audio["avg_valence"],
        ],
    })

    audio_chart_df["value_pct"] = (audio_chart_df["value"] * 100).round(1)

    radar_fig = go.Figure()

    # Close the polygon by repeating the first value
    r_values = audio_chart_df["value_pct"].tolist()
    theta_values = audio_chart_df["feature"].tolist()

    r_closed = r_values + [r_values[0]]
    theta_closed = theta_values + [theta_values[0]]

    radar_fig.add_trace(
        go.Scatterpolar(
            r=r_closed,
            theta=theta_closed,
            fill="toself",
            fillcolor=PRIMARY_COLOR_FILL,
            # border colour
            line=dict(color=PRIMARY_COLOR, width=2),
            name="Album audio profile",
            hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
        )
    )

    radar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#333333",
                linecolor="#333333",
                tickfont=dict(color="#B3B3B3"),
            ),
            angularaxis=dict(
                gridcolor="#333333",
                linecolor="#333333",
                tickfont=dict(color="#FFFFFF"),
            ),
        ),
        showlegend=False,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )



    st.plotly_chart(radar_fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg tempo", f"{audio['avg_tempo']:.1f} BPM")
    with c2:
        st.metric("Avg loudness", f"{audio['avg_loudness']:.1f} dB")
    with c3:
        avg_duration_ms = audio["avg_duration_ms"]

        if pd.notna(avg_duration_ms):
            total_seconds = int(round(avg_duration_ms / 1000))
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_display = f"{minutes}min {seconds}sec"
        else:
            duration_display = "N/A"

        st.metric("Avg duration", duration_display)


    st.markdown("---")


    st.markdown("### Track-level view")

    track_display = track_df.copy()

    if "track_number" in track_display.columns:
        track_display = track_display.sort_values(["track_number", "track_name"]).copy()
        track_display["#"] = track_display["track_number"]
    else:
        track_display = track_display.reset_index(drop=True)
        track_display["#"] = track_display.index + 1

    track_display["Track"] = (
        track_display["track_name"].fillna("Unknown")
        if "track_name" in track_display.columns
        else "Unknown"
    )

    track_display["Artists"] = (
        track_display["artist_names"].fillna("")
        if "artist_names" in track_display.columns
        else ""
    )

    if "track_popularity" in track_display.columns:
        track_display["Popularity"] = pd.to_numeric(
            track_display["track_popularity"], errors="coerce"
        ).round(0)

    if "tempo" in track_display.columns:
        track_display["Tempo"] = pd.to_numeric(
            track_display["tempo"], errors="coerce"
        ).round(1)

    if "time_signature" in track_display.columns:
        track_display["Time sig."] = track_display["time_signature"]

    if "explicit" in track_display.columns:
        track_display["Explicit"] = np.where(track_display["explicit"], "Yes", "No")

    if "duration_ms" in track_display.columns:
        total_seconds = pd.to_numeric(track_display["duration_ms"], errors="coerce") / 1000
        total_seconds = total_seconds.round().astype("Int64")
        minutes = (total_seconds // 60).astype("Int64")
        seconds = (total_seconds % 60).astype("Int64")
        track_display["Duration"] = (
            minutes.astype(str) + ":" + seconds.astype(str).str.zfill(2)
        )

    feature_pct_map = {
        "energy": "Energy",
        "danceability": "Danceability",
        "valence": "Valence",
        "speechiness": "Speechiness",
        "acousticness": "Acousticness",
        "instrumentalness": "Instrumentalness",
        "liveness": "Liveness",
    }

    for source_col, target_col in feature_pct_map.items():
        if source_col in track_display.columns:
            track_display[target_col] = (
                pd.to_numeric(track_display[source_col], errors="coerce") * 100
            ).round(1)

    final_cols = [
        "#",
        "Track",
        "Artists",
        "Duration",
        "Popularity",
        "Explicit",
        "Tempo",
        "Time sig.",
        "Energy",
        "Danceability",
        "Valence",
        "Speechiness",
        "Acousticness",
        "Instrumentalness",
        "Liveness",
    ]
    final_cols = [col for col in final_cols if col in track_display.columns]

    st.dataframe(
        track_display[final_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Track": st.column_config.TextColumn(width="medium"),
            "Artists": st.column_config.TextColumn(width="large"),
            "Duration": st.column_config.TextColumn(width="small"),
            "Popularity": st.column_config.ProgressColumn(
                "Popularity",
                min_value=0,
                max_value=100,
                format="%d",
                width="medium",
            ),
            "Explicit": st.column_config.TextColumn(width="small"),
            "Tempo": st.column_config.NumberColumn(format="%.1f BPM", width="small"),
            "Time sig.": st.column_config.NumberColumn(width="small"),
            "Energy": st.column_config.ProgressColumn(
                "Energy",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Danceability": st.column_config.ProgressColumn(
                "Danceability",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Valence": st.column_config.ProgressColumn(
                "Valence",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Speechiness": st.column_config.ProgressColumn(
                "Speechiness",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Acousticness": st.column_config.ProgressColumn(
                "Acousticness",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Instrumentalness": st.column_config.ProgressColumn(
                "Instrumentalness",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
            "Liveness": st.column_config.ProgressColumn(
                "Liveness",
                min_value=0,
                max_value=100,
                format="%.1f%%",
                width="medium",
            ),
        },
    )
    
#The explicit vs non-explicit section on the album dashboard
#shows a pie chart, a table of explicit tracks and a bar chart comparing average popularity

def render_explicit_section(track_df):
    st.markdown("### Explicit vs Non-Explicit tracks")
    
    from utils.album_features import get_explicit_counts, get_explicit_popularity
    
    explicit_df = get_explicit_counts(track_df)
    pie_fig = go.Figure(go.Pie(
    labels=explicit_df["type"],
    values=explicit_df["count"],
    marker=dict(colors=[
        PRIMARY_COLOR if t == "Explicit" else "#535353" 
        for t in explicit_df["type"]
    ]),
))
    st.plotly_chart(pie_fig, width="stretch")

    explicit_tracks = track_df[track_df["explicit"] == True][["track_name", "artist_names"]].copy()
    explicit_tracks.columns = ["Track", "Artists"]

    if not explicit_tracks.empty:
        st.markdown("### Explicit Tracks")
        st.dataframe(explicit_tracks, hide_index=True, width="stretch")
    else:
        st.markdown("### Explicit Tracks")
        st.info("No explicit tracks on this album.")

    st.markdown("### Average popularity: explicit vs non-explicit")
    popularity_df = get_explicit_popularity(track_df)
    bar_fig = go.Figure(go.Bar(
        x=popularity_df["type"],
        y=popularity_df["avg_popularity"],
        width=0.3,
        marker=dict(color=[                                                
            PRIMARY_COLOR_FILL if t == "Explicit" else "#535353"
            for t in popularity_df["type"]],
            line=dict(color=PRIMARY_COLOR, width=1)),
    ))

    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#B3B3B3"), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#B3B3B3"), gridcolor="#333333", zeroline=False),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
    )

    st.plotly_chart(bar_fig, width="stretch")

# function to render the full page
def render_album_summary_page(summary_result, cover_data):
    render_album_header(summary_result, cover_data)
    st.markdown("---")
    render_album_feature_summary(summary_result)
    render_explicit_section(summary_result["track_df"])

