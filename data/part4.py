import sqlite3
import pandas as pd
import numpy as np


connect = sqlite3.connect("spotify_database.db")
cursor = connect.cursor()

cursor.execute("""
SELECT danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo, duration_ms
FROM features_data
""")

rows = cursor.fetchall()

df = pd.DataFrame(rows, columns=[
    "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "liveness", "valence", "tempo", "duration_ms"])

print(df.describe())

print(len(df[df["loudness"] > 0]))
print(len(df[df["tempo"] == 0]))
print(len(df[
    (df["duration_ms"] < 30000) |
    (df["duration_ms"] > 1800000)
]))


#OUTLIER DETECTION features
Q1 = df["loudness"].quantile(0.25)
Q3 = df["loudness"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_duration = df[
    (df["loudness"] < lower_bound) |
    (df["loudness"] > upper_bound)
]

print("\nNumber of loudness outliers:", len(outliers_duration))


Q1 = df["tempo"].quantile(0.25)
Q3 = df["tempo"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_duration = df[
    (df["tempo"] < lower_bound) |
    (df["tempo"] > upper_bound)
]

print("Number of tempo outliers:", len(outliers_duration))


Q1 = df["duration_ms"].quantile(0.25)
Q3 = df["duration_ms"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_duration = df[
    (df["duration_ms"] < lower_bound) |
    (df["duration_ms"] > upper_bound)
]

print("Number of duration outliers:", len(outliers_duration))

#POPULARITY OUTLIERS
cursor.execute("SELECT track_popularity FROM tracks_data WHERE track_popularity IS NOT NULL")

popularity = pd.DataFrame(cursor.fetchall(), columns=["popularity"])
#print(popularity.describe())

Q1 = popularity["popularity"].quantile(0.25)
Q3 = popularity["popularity"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

pop_outliers = popularity[(popularity["popularity"] < lower) | 
                   (popularity["popularity"] > upper)]

print("\nNumber of popularity outliers:", len(pop_outliers))

cursor.execute(f"""
SELECT
  t.id AS track_id,
  t.track_popularity,
  t.explicit,
  a.track_name,
  a.artist_id,
  a.album_id,
  a.album_name
FROM tracks_data t
JOIN albums_data a
  ON a.track_id = t.id
WHERE t.track_popularity > {upper}
ORDER BY t.track_popularity DESC
""")

df = pd.DataFrame(cursor.fetchall(), columns=[
    "track_id", "popularity", "explicit", "track_name",
    "artist_id", "album_id", "album_name"
])

df_unique = (
    df.sort_values("popularity", ascending=False)
      .drop_duplicates(subset=["track_id"])
)

print(df_unique.head(10))
print("Rows before:", len(df), "after", len(df_unique))
print(df_unique["artist_id"].value_counts().head(20))


df = pd.read_sql_query("""
SELECT
    a.track_id,
    a.track_name,
    a.artist_id,
    f.danceability,
    f.energy,
    f.loudness,
    f.tempo,
    f.duration_ms,
    t.track_popularity,
    ar.name
FROM albums_data a
JOIN features_data f ON a.track_id = f.id
JOIN tracks_data t ON a.track_id = t.id
JOIN artist_data ar ON a.artist_id = ar.id
WHERE
    f.loudness IS NOT NULL AND
    f.tempo IS NOT NULL AND
    f.duration_ms IS NOT NULL AND
    t.track_popularity IS NOT NULL
""", connect)

df = df.drop_duplicates(subset=["track_id"])

clean_df = df[
    (df["duration_ms"] > 0) &
    (df["tempo"] > 0) &
    (df["track_popularity"] >= 0) &
    (df["track_popularity"] <= 100)
].dropna(subset=[
    "track_id",
    "artist_id",
    "duration_ms",
    "tempo",
    "track_popularity"
]).copy()

def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

loud_lo, loud_hi = iqr_bounds(df["loudness"])
tempo_lo, tempo_hi = iqr_bounds(df["tempo"])
dur_lo, dur_hi = iqr_bounds(df["duration_ms"])

filtered_df = df[
    (df["loudness"].between(loud_lo, loud_hi)) &
    (df["tempo"].between(tempo_lo, tempo_hi)) &
    (df["duration_ms"].between(dur_lo, dur_hi)) &
    (df["track_popularity"] >= 0) &
    (df["track_popularity"] <= 100)
    ].copy()

print(filtered_df.head(10))

print("\nfiltered describe:")
print(filtered_df[["loudness","tempo","duration_ms","track_popularity"]].describe())