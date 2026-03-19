import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect("spotify_database.db")
cursor= conn.cursor()

#Choosing album to investigate 
query = """
SELECT DISTINCT album_name
FROM albums_data
WHERE album_name LIKE '%Dawn%'
"""
print(pd.read_sql_query(query, conn))

#Analyzing 
album_name = "Dawn Fm"

query = f"""
SELECT al.track_name,
       f.danceability,
       f.loudness,
       f.energy,
       f.valence
FROM albums_data al
JOIN features_data f ON al.track_id = f.id
WHERE al.album_name = '{album_name}'
"""

df_album = pd.read_sql_query(query, conn)
df_album = df_album.drop_duplicates(subset="track_name")

print(df_album)
print(df_album.describe())

#Plot danceability
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df_album["danceability"], marker="o")
plt.title("Danceability per Track - Dawn Fm")
plt.xlabel("Track Index")
plt.ylabel("Danceability")
#plt.show()

#plot loudness
plt.figure()
plt.plot(df_album["loudness"], marker="o")
plt.title("Loudness per Track - Dawn Fm")
plt.xlabel("Track Index")
plt.ylabel("Loudness (dB)")
#plt.show()

 
plt.figure()
df_album[["danceability","energy","valence"]].plot()
plt.title("Audio Features - Dawn Fm")
#plt.show()

#boxplot 
plt.figure()
df_album[["danceability","loudness","energy","valence"]].plot(kind="box")
plt.title("Feature Distribution - Dawn Fm")
#plt.show()

query = """
SELECT ar.name AS artist,
       f.loudness
FROM features_data f
JOIN tracks_data t ON f.id = t.id
JOIN albums_data al ON t.id = al.track_id
JOIN artist_data ar ON al.artist_id = ar.id
"""
df = pd.read_sql_query(query, conn)


threshold = df["loudness"].quantile(0.9)
top_10 = df[df["loudness"] >= threshold]

top_10_filtered = top_10[top_10["artist"] != "Various Artists"]

artist_counts = top_10_filtered["artist"].value_counts()

print(artist_counts.head(10))


#correlatation energy and louder music
query = """
SELECT f.loudness, f.energy
FROM features_data f
"""
df_corr = pd.read_sql_query(query, conn)

print(df_corr.corr())


#Which artists have the highest proportion of explicit tracks?
query = """
SELECT explicit,
       AVG(track_popularity) AS avg_popularity,
       COUNT(*) AS n_tracks
FROM tracks_data
GROUP BY explicit
"""

df_exp = pd.read_sql_query(query, conn)
print(df_exp) #explicit content does not negatively affect popularity 

#most amount of explicit tracks 
query = """
SELECT ar.name AS artist,
       t.explicit
FROM tracks_data t
JOIN albums_data al ON t.id = al.track_id
JOIN artist_data ar ON al.artist_id = ar.id
"""

df_artist = pd.read_sql_query(query, conn)

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