import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.helpers import prepare_year_column, filter_by_year_range
from utils.time_trends import build_album_level_era_summary, build_album_level_year_summary, render_era_bar_chart, render_year_line_chart

st.set_page_config(layout="wide")
st.title("Music Through the Years")

if "app_data" not in st.session_state:
    st.warning("Load the dataset from the Home page first.")
    st.stop()

df_final = st.session_state["app_data"]["df_final"].copy()

df_final = prepare_year_column(df_final)

min_year = int(df_final["release_year"].min())
max_year = int(df_final["release_year"].max())

#Creates a slider in the sidebar to filter the data by a year range and stored in session state 
default_year_range = st.session_state.get("year_range", (min_year, max_year))

year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=default_year_range,
    key="year_range_music",
)

st.session_state["year_range"] = year_range

df_final = filter_by_year_range(df_final, year_range)

#Creates summary tables by era and by year 
era_summary_df, available_features = build_album_level_era_summary(df_final)
year_summary_df, _ = build_album_level_year_summary(df_final)

st.markdown(
    "Average album-level feature values by release era. "
    "Albums are grouped into decades using their release year."
)

if era_summary_df.empty:
    st.info("No era data available.")
    st.stop()

# Lets the users pick their view, by era or by year 
view_mode = st.radio(
    "View mode",
    options=["Per decade", "Per year"],
    horizontal=True,
)

# Shows the chosen table 
if view_mode == "Per decade":
    st.dataframe(era_summary_df.drop(columns=["era_sort"]), width="stretch")
else:
    st.dataframe(year_summary_df, width="stretch")
    
# Feature selection
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

# a drop down for selecting which feature to show
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

#Shows the graph corresponding to the chosen view mode 
if view_mode == "Per decade":
    render_era_bar_chart(
        era_summary_df,
        feature_col=selected_feature,
        feature_label=feature_labels.get(selected_feature, selected_feature),
        value_format=value_format,
    )
else:
    render_year_line_chart(
        year_summary_df,
        feature_col=selected_feature,
        feature_label=feature_labels.get(selected_feature, selected_feature),
        value_format=value_format,
    )
