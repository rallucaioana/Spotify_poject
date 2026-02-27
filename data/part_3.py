import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


connect = sqlite3.connect("spotify_database.db")
cursor = connect.cursor()

# GEEFT ALBUM ID EN AANTAL TRACKS VAN ALBUMS MET ZELFDE NAAM (versions)
album = "Heartbreak On A Full Moon"

cursor.execute(" SELECT album_id, COUNT(*) as n_tracks FROM albums_data WHERE album_name = ? GROUP BY album_id ORDER BY n_tracks DESC;", (album,))

album_versions = cursor.fetchall()

print("\nAvailable album versions:")
for r in album_versions:
    print(r)

album_id = "3zak0kNLcOY5vFcB3Ipskp"
cursor.execute("SELECT track_number, track_name FROM albums_data WHERE album_id = ? ORDER BY track_number;", (album_id,))
rows = cursor.fetchall()

print(f"\nTracks on album: {album}\n")

for row in rows:
    print(f"{row[0]}. {row[1]}")


cursor.execute("""SELECT albums_data.track_name, features_data.danceability, features_data.loudness
FROM albums_data JOIN features_data ON albums_data.track_id = features_data.id WHERE albums_data.album_id = ?""", (album_id,))
rows = cursor.fetchall()

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


# FEAUTURE TOP 10
feature = "danceability"

cursor.execute(f" SELECT {feature} FROM features_data WHERE {feature} IS NOT NULL ORDER BY {feature}")
cutoff = np.percentile(cursor.fetchall(), 90)

cursor.execute(f"""
SELECT albums_data.track_name,
albums_data.artist_id,
artist_data.name,
features_data.{feature}
FROM albums_data 
JOIN features_data  ON albums_data.track_id = features_data.id
JOIN artist_data ON artist_data.id = albums_data.artist_id
WHERE features_data.{feature} >= ?
ORDER BY features_data.{feature} DESC
""", (cutoff,))

top_10p = cursor.fetchall()

artist_counts = Counter([r[2] for r in top_10p])

print(f"\nArtists appearing most in top 10% on {feature}:")
for name, count in artist_counts.most_common(20):
    print(name, count)

# ARE EXPLICIT TRACKS MORE POPULAR
cursor.execute (" SELECT track_popularity, explicit FROM tracks_data WHERE track_popularity is NOT NULL AND explicit is NOT NULL")
rows = cursor.fetchall()

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
