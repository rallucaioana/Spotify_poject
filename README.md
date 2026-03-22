# Spotify Data Analysis Project

## Description
This project analyses a Spotify dataset containing information on tracks, albums, artists and audio features collected in 2023. The end product is an interactive dashboard built with Streamlit that allows for statistical analyses of the data.

## Installation

### Requirements
Make sure you have Python installed. Then install the required libraries by running:
pip install streamlit pandas numpy plotly scipy scikit-learn matplotlib seaborn requests

### Running the Dashboard
To run the dashboard locally, navigate to the project folder and run:
streamlit run Overview.py

## Data
The dataset is stored in a SQLite database and contains the following tables:
- artist_data: artist statistics including popularity and followers
- albums_data: album and track information
- tracks_data: track popularity and explicit content
- features_data: audio features per track (danceability, energy, tempo, etc.)

Uses Spotify Web API for fetching album covers and artist profile pictures. 

## Dashboard Features
- Home page: general statistics and overview of the dataset
- Album Dashboard: detailed analysis per album including audio features, popularity, and explicit track breakdown
- Music Through The Years: trends in music features over time
- Search by Artist: search for a specific artist and explore their statistics

## Authors
- Teun J. Heerze 
- Raluca I. Dinu 
- Christianne C. Schröder 
- Samira Delawar
