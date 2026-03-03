import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

con = sqlite3.connect("/Users/ralucadinu/Desktop/data engineering/Spotify_poject/data/spotify_database.db")

q= ("""SELECT 
                      t.id, 
                      t.track_popularity,
                      f.danceability,
                      f.energy,
                      f.loudness,
                      f.tempo,
                      a.duration_ms
                  FROM tracks_data t
                  JOIN features_data f ON t.id = f.id
                  JOIN albums_data a ON t.id = a.track_id""")

df_tracks = pd.read_sql(q, con)

df_tracks.head()

#1 detect outliers
features = [
    "track_popularity",
    "danceability",
    "energy",
    "loudness",
    "tempo"
]

for f in features:
    Q1 = df_tracks[f].quantile(0.25)
    Q3 = df_tracks[f].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_tracks[(df_tracks[f] < lower) | (df_tracks[f] > upper)]
    print(len(outliers))

for f in features:
    plt.figure()
    plt.boxplot(df_tracks[f])
    plt.title(f"Boxplot of {f}")
    plt.show()

#2 delete invalid records

#remove if id is NaN
df_tracks = df_tracks.dropna(subset=["id"])

#remove if duration is below 0
df_tracks = df_tracks[df_tracks["duration_ms"] > 0]

#remove if tempo is below 0
df_tracks = df_tracks[df_tracks["tempo"] > 0]

#5 remove duplicates
query_artists = """
SELECT id, name
FROM artist_data
"""
df_artists = pd.read_sql(query_artists, con)

#no full row duplicates
df_artists.duplicated().sum()

df_artists["name_clean"] = df_artists["name"].str.lower().str.strip()

duplicate_names = df_artists.duplicated(subset=["name_clean"], keep=False)

id_counts = df_artists.groupby("name_clean")["id"].nunique()

multiple_id_names = id_counts[id_counts > 1]

print(multiple_id_names)

#9
query = """
SELECT f.id AS track_id,
       f.danceability,
       a.artist_id
FROM features_data f
JOIN albums_data a ON f.id = a.track_id
"""

df = pd.read_sql(query, con)

genre_query = """
SELECT id AS artist_id,
       genre_0, genre_1, genre_2, genre_3, genre_4
FROM artist_data
"""

df_genre = pd.read_sql(genre_query, con)

df = df.merge(df_genre, on="artist_id", how="left")

df = df.dropna(subset=["danceability"])

df["danceability_level"] = pd.qcut(
    df["danceability"],5,labels=["very low", "low", "medium", "high", "very high")

low_high = df[df["danceability_level"].isin(["very low", "very high"])]

highest = (
    low_high
    .sort_values("danceability", ascending=False)
    [["danceability", "genre_0"]]
    .iloc[0]
)

highest

lowest = (
    low_high
    .sort_values("danceability", ascending=True)
    [["danceability", "genre_0"]]
    .iloc[0]
)

lowest