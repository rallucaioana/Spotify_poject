import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import add_era_column, get_theme_colors

PRIMARY_COLOR, PRIMARY_COLOR_FILL = get_theme_colors(fill_alpha=0.45)

st.set_page_config(layout="wide")
st.title("Music Through the Years")

if "app_data" not in st.session_state:
    st.warning("Load the dataset from the Home page first.")
    st.stop()

df_final = st.session_state["app_data"]["df_final"].copy()

@st.cache_data(show_spinner=False)
def build_album_level_era_summary(df: pd.DataFrame):
    df = add_era_column(df)

    feature_map = {
        "danceability": "Danceability",
        "energy": "Energy",
        "valence": "Valence",
        "acousticness": "Acousticness",
        "speechiness": "Speechiness",
        "instrumentalness": "Instrumentalness",
        "liveness": "Liveness",
        "tempo": "Tempo",
        "album_popularity": "Album popularity",
        "track_popularity": "Track popularity",
    }

    available_features = [col for col in feature_map if col in df.columns]

    required_cols = ["album_id", "album_name", "release_date", "era"] + available_features
    album_df = (
        df.loc[df["album_id"].notna(), required_cols]
        .copy()
        .groupby(["album_id", "album_name", "release_date", "era"], dropna=False)[available_features]
        .mean()
        .reset_index()
    )

    album_df = album_df.loc[album_df["era"].notna()].copy()

    era_summary = (
        album_df.groupby("era", dropna=False)[available_features]
        .mean()
        .reset_index()
    )

    era_summary["era_sort"] = (
        era_summary["era"].str.replace("s", "", regex=False).astype(int)
    )
    era_summary = era_summary.sort_values("era_sort").reset_index(drop=True)

    return era_summary, available_features


def render_era_bar_chart(df: pd.DataFrame, feature_col: str, \
                         feature_label: str, value_format: str = ".2f",):

    chart_df = (
        df[["era", "era_sort", feature_col]]
        .dropna()
        .sort_values("era_sort")
        .copy()
    )

    fig = go.Figure()

    # barplot
    fig.add_trace(
        go.Bar(
            x=chart_df["era"],
            y=chart_df[feature_col],
            marker=dict(
                color=PRIMARY_COLOR_FILL,
                line=dict(color=PRIMARY_COLOR, width=2),
            ),
            hovertemplate=(
                "Era: %{x}<br>"
                + f"{feature_label}: "
                + "%{y:" + value_format + "}<extra></extra>"
            ),
            name="Average",
        )
    )

    # trendline
    fig.add_trace(
        go.Scatter(
            x=chart_df["era"],
            y=chart_df[feature_col],
            mode="lines+markers",
            line=dict(
                color=PRIMARY_COLOR,
                width=3,
                shape="spline"
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
        title=feature_label,
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
            title="Average",
            tickfont=dict(color="#B3B3B3"),
            gridcolor="#333333",
            zeroline=False,
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
    )

    st.plotly_chart(fig, width="stretch")
    

era_summary_df, available_features = build_album_level_era_summary(df_final)

st.markdown(
    "Average album-level feature values by release era. "
    "Albums are grouped into decades using their release year."
)

if era_summary_df.empty:
    st.info("No era data available.")
    st.stop()

st.dataframe(era_summary_df.drop(columns=["era_sort"]), width="stretch")

feature_labels = {
    "danceability": "Danceability",
    "energy": "Energy",
    "valence": "Valence",
    "acousticness": "Acousticness",
    "speechiness": "Speechiness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "tempo": "Tempo",
    "album_popularity": "Album popularity",
    "track_popularity": "Average track popularity",
}

default_feature = "danceability"
default_index = available_features.index(default_feature) if default_feature in available_features else 0

selected_feature = st.selectbox(
    "Select feature",
    options=available_features,
    index=default_index,
    format_func=lambda x: feature_labels.get(x, x),
)

value_format = (
    ".1f"
    if selected_feature in ["tempo", "album_popularity", "track_popularity"]
    else ".2f"
)

render_era_bar_chart(
    era_summary_df,
    feature_col=selected_feature,
    feature_label=feature_labels.get(selected_feature, selected_feature),
    value_format=value_format,
)
