import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

con = sqlite3.connect("data/spotify_database.db")
cur = con.cursor()

#1
#pick album and investigate features
album = "Heartbreak On A Full Moon"

cur.execute(" SELECT album_id, COUNT(*) as n_tracks FROM albums_data WHERE album_name = ? GROUP BY album_id ORDER BY n_tracks DESC;", (album,))

album_versions = cur.fetchall()

print("\nAvailable album versions:")
for r in album_versions:
    print(r)

album_id = "3zak0kNLcOY5vFcB3Ipskp"
cur.execute("SELECT track_number, track_name FROM albums_data WHERE album_id = ? ORDER BY track_number;", (album_id,))
rows = cur.fetchall()

print(f"\nTracks on album: {album}\n")

for row in rows:
    print(f"{row[0]}. {row[1]}")

#consistency of features over all tracks 
cur.execute("""SELECT albums_data.track_name, features_data.danceability, features_data.loudness
FROM albums_data JOIN features_data ON albums_data.track_id = features_data.id WHERE albums_data.album_id = ?""", (album_id,))
rows = cur.fetchall()

print(f"\nFeatures for album: {album}")
for row in rows:
    print(row)

dance = [row[1] for row in rows]
loud = [row[2] for row in rows]

print("\nDanceability:")
print("Mean:", np.mean(dance))
print("Std:", np.std(dance))

print("\nLoudness:")
print("Mean:", np.mean(loud))
print("Std:", np.std(loud))

plt.plot(dance)
plt.title(f"Danceability – {album}")
plt.xlabel("Track number")
plt.ylabel("Danceability")
plt.show()

plt.plot(loud)
plt.title(f"Loudness – {album}")
plt.xlabel("Track number")
plt.ylabel("Loudness")
plt.show()    

#2 
feature = "energy"

cur.execute(
    f"""
    SELECT
        a.track_id AS track_id,
        a.track_name,
        a.artist_0,
        a.artist_1,
        a.artist_2,
        a.artist_3,
        a.artist_4,
        a.artist_5,
        a.artist_6,
        a.artist_7,
        a.artist_8,
        a.artist_9,
        a.artist_10,
        a.artist_11,
        f.{feature} AS feature_value
    FROM albums_data a
    JOIN features_data f
        ON a.track_id = f.id
    WHERE f.{feature} IS NOT NULL
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]
df_feature = pd.DataFrame(rows, columns=cols)

cutoff = df_feature["feature_value"].quantile(0.90)
df_selected = df_feature[
    df_feature["feature_value"] >= cutoff
]

# merge artist columns into one artists column per track
artist_cols = [f"artist_{i}" for i in range(12)]

df_selected["artists"] = df_selected[artist_cols].apply(
    lambda r: [x for x in r.tolist() if pd.notna(x)],
    axis=1
)

# count which artists appear most in the top 10%
top_artists = (
    df_selected["artists"]
    .explode()
    .value_counts()
)
print(top_artists)

#3
#relationship between artist popularity and album popularity
cur.execute(
    """
    SELECT
        a.artist_id,
        AVG(a.album_popularity) AS mean_album_popularity,
        ad.artist_popularity
    FROM albums_data a
    JOIN artist_data ad
        ON a.artist_id = ad.id
    WHERE a.album_popularity IS NOT NULL
    GROUP BY a.artist_id;
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]

df_popularity = pd.DataFrame(rows, columns=cols)

corr = df_popularity["mean_album_popularity"].corr(df_popularity["artist_popularity"])
print("Correlation:", corr)

sns.regplot(
    x=df_popularity["artist_popularity"],
    y=df_popularity["mean_album_popularity"],
    marker="x",
    line_kws=dict(color="r"),
    ci=95
)

plt.xlabel("Artist Popularity")
plt.ylabel("Mean Album Popularity")
plt.title("Artist vs Mean Album Popularity")
plt.show()

#4
#add eras column
cur.execute(
    """
    SELECT * 
    FROM albums_data
    WHERE release_date IS NOT NULL;
    """
)
rows = cur.fetchall()
cols = [d[0] for d in cur.description]

df_albums = pd.DataFrame(rows, columns=cols)
df_albums["release_date"] = pd.to_datetime(df_albums["release_date"])

df_albums["era"] = (df_albums["release_date"].dt.year // 10) * 10
df_albums["era"] = df_albums["era"].astype("Int64").astype(str) + "s"

# checking results
print(sorted(df_albums["era"].dropna().unique()))
print(df_albums[["release_date", "era"]].sample(10))

#5
# Are explicit tracks more popular 
cur.execute (" SELECT track_popularity, explicit FROM tracks_data WHERE track_popularity is NOT NULL AND explicit is NOT NULL")
rows = cur.fetchall()

explicit = np.array([r[0] for r in rows if r[1] == 'true'], dtype=float)
nonexplicit = np.array([r[0] for r in rows if r[1] == 'false'], dtype=float)

print("\nExplicit vs popularity analysis:")
print("\nNumber explicit:", len(explicit))
print("Number non-explicit:", len(nonexplicit))

print("\nMean popularity explicit:", round(np.mean(explicit), 3))
print("Mean popularity non-explicit:", round(np.mean(nonexplicit), 3))

plt.boxplot([nonexplicit, explicit], tick_labels=["Non-explicit", "Explicit"])
plt.ylabel("Track popularity")
plt.title("Explicit vs Non-explicit Popularity")
plt.show()

#6
#most amount of explicit tracks 
query = """
SELECT ar.name AS artist,
       t.explicit
FROM tracks_data t
JOIN albums_data al ON t.id = al.track_id
JOIN artist_data ar ON al.artist_id = ar.id
"""

df_artist = pd.read_sql_query(query, con)

df_artist["explicit_binary"] = df_artist["explicit"].map({
    "true": 1,
    "false": 0
})

artist_ratio = (
    df_artist
    .groupby("artist")["explicit_binary"]
    .agg(["mean", "count"])
)

artist_ratio = artist_ratio[artist_ratio["count"] > 20]
artist_ratio = artist_ratio.sort_values("mean", ascending=False)

print(artist_ratio.head(10))

#7
#collaborations
cur.execute(
    """
    SELECT
        t.id,
        t.track_popularity,
        a.artist_0,
        a.artist_1,
        a.artist_2,
        a.artist_3,
        a.artist_4,
        a.artist_5,
        a.artist_6,
        a.artist_7,
        a.artist_8,
        a.artist_9,
        a.artist_10,
        a.artist_11
    FROM tracks_data t
    JOIN albums_data a
        ON t.id = a.track_id
    WHERE t.track_popularity IS NOT NULL;
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]

df_collab = pd.DataFrame(rows, columns=cols)

artist_cols = [f"artist_{i}" for i in range(12)]

# convert empty / whitespace-only strings to NaN
df_collab[artist_cols] = (
    df_collab[artist_cols]
    .replace(r"^\s*$", np.nan, regex=True)
)

# count number of non-null artists per track
df_collab["artist_count"] = df_collab[artist_cols].notna().sum(axis=1)

# collaboration = more than 1 artist
df_collab["collaboration"] = (df_collab["artist_count"] > 1).astype(int)

# dropping potential duplicate tracks
df_collab = df_collab.drop_duplicates(subset=["id"])


print(df_collab.groupby("collaboration")["track_popularity"]
      .agg(["mean", "median", "count"]))

solo = df_collab[df_collab["collaboration"] == 0]["track_popularity"]
collab = df_collab[df_collab["collaboration"] == 1]["track_popularity"]
u_stat, p_value = mannwhitneyu(collab, solo, alternative="two-sided")
print("\nP-value of statistical difference between distrubutions:", p_value)

sns.boxplot(
    x="collaboration",
    y="track_popularity",
    data=df_collab
)
plt.show()

con.close()
