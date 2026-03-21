import streamlit as st
import plotly.graph_objects as go

from utils.load_session_data import ensure_app_data_loaded
from utils.artist_audits import get_top_artists_by_feature

st.set_page_config(layout="wide")
st.title("Top Artists by Feature")

if "app_data" not in st.session_state:
    st.warning("Load the dataset from the Home page first.")
    st.stop()

app_data = ensure_app_data_loaded()
df_final = app_data["df_final"]

feature_options = [
    "danceability", "energy", "valence", "acousticness",
    "speechiness", "instrumentalness", "liveness", "tempo"
]

selected_feature = st.selectbox("Select feature", feature_options)
artist_col = "primary_artist_name"

artist_counts = df_final.groupby(artist_col).size()
valid_artists = artist_counts[artist_counts >=3]

n_artists = len(valid_artists)

top_n = st.slider("Number of artists",
	min_value=1,
   	max_value=100,  
    	value=10)

if selected_feature:
    top_artists_df = get_top_artists_by_feature(
        df_final,
        selected_feature,
        top_n=top_n
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_artists_df["avg_feature"],
            y=top_artists_df["primary_artist_name"],
            orientation="h"
        )
    )

    fig.update_layout(
        title=f"Top Artists by {selected_feature}",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, width="stretch")
