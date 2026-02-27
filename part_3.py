import pandas as pd
import numpy as np
import statsmodels.api as sm
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu


con = sqlite3.connect("data/spotify_database.db")
cur = con.cursor()

# tests
# cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
# tables = cur.fetchall()

# print("Tables:")
# for table in tables:
#     print(table[0])

# cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")

# cur.execute("PRAGMA table_info(albums_data);")
# col_names = [c[1] for c in cur.fetchall()]

# print(col_names)


# === 1. ===
album_name = "The Colour And The Shape"
cur.execute(
    """
    SELECT a.*, f.*
    FROM albums_data a
    JOIN features_data f
      ON a.track_id = f.id
    WHERE a.album_name = ?
    ORDER BY a.track_number
    """,
    (album_name,)
)
data = cur.fetchall()
col_names = [desc[0] for desc in cur.description]
print(col_names)

df_album = pd.DataFrame(data, columns=col_names)
print(df_album)

# Summary stats
stats = df_album[["danceability", "loudness"]].agg(["mean", "std", "min", "max"])
stats.loc["range"] = stats.loc["max"] - stats.loc["min"]

# Relative variability (smaller = more consistent)
cv = (df_album[["danceability", "loudness"]].std() / df_album[["danceability", "loudness"]].mean().abs()) * 100
print(stats)
print("\nCoefficient of variation (%):")
print(cv)

print(df_album[["track_number", "danceability", "loudness"]].to_string(index=False))


# === 2. ===
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



# === 3. ===
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



# === 4. ===
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



# === 5. ===
cur.execute(
    """
    SELECT
        track_popularity,
        explicit
    FROM tracks_data
    WHERE track_popularity IS NOT NULL;
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]

df_explicit = pd.DataFrame(rows, columns=cols)

# converting string true/false to boolean
df_explicit["explicit"] = df_explicit["explicit"].map({
    "false": 0,
    "true": 1
})

explicit_mean = df_explicit[df_explicit["explicit"] == 1]["track_popularity"].mean()
clean_mean = df_explicit[df_explicit["explicit"] == 0]["track_popularity"].mean()

explicit_median = df_explicit[df_explicit["explicit"] == 1]["track_popularity"].median()
clean_median = df_explicit[df_explicit["explicit"] == 0]["track_popularity"].median()

print("Explicit: \nMean:", explicit_mean, "\nMedian:", explicit_median, \
    "\n\nNon-explicit: \nMean:", clean_mean, "\nMedian:", clean_median)

sns.boxplot(
    x="explicit",
    y="track_popularity",
    data=df_explicit
)
plt.show()

explicit = df_explicit[df_explicit["explicit"] == 1]["track_popularity"]
clean = df_explicit[df_explicit["explicit"] == 0]["track_popularity"]

u_stat, p_value = mannwhitneyu(explicit, clean, alternative="two-sided")

print("\nP-value of statistical difference between distrubutions:", p_value)



# === 6. ===
cur.execute(
    """
    SELECT
        t.id,
        t.explicit,
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
        ON t.id = a.track_id;
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]

df_explicit_pop = pd.DataFrame(rows, columns=cols)

# converting string true/false to numeric
df_explicit_pop["explicit"] = df_explicit_pop["explicit"].map({
    "false": 0, 
    "true": 1
})

artist_cols = [f"artist_{i}" for i in range(12)]

df_explicit_pop["artists"] = df_explicit_pop[artist_cols].apply(
    lambda r: [x for x in r.tolist() if pd.notna(x)],
    axis=1
)

df_pairs = (
    df_explicit_pop[["id", "explicit", "artists"]]
      .explode("artists")
      .dropna(subset=["artists"])
      .rename(columns={"artists": "artist_id"})
      .drop_duplicates(subset=["id", "artist_id"])
)

artist_stats = (
    df_pairs.groupby("artist_id")["explicit"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "explicit_proportion", "count": "track_count"})
      .sort_values("explicit_proportion", ascending=False)
)

# filtering out artists who have less than 75 tracks published
artist_stats = artist_stats[artist_stats["track_count"] >= 75]

print(artist_stats.head(10))


# === 7. ===
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

