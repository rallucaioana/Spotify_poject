import sqlite3
import pandas as pd

con = sqlite3.connect("data/spotify_database.db")

artist_cols = pd.read_sql_query("PRAGMA table_info(artist_data);", con)
album_cols = pd.read_sql_query("PRAGMA table_info(albums_data);", con)

print("artist_data columns:")
print(artist_cols[["name", "type"]])

print("\nalbums_data columns:")
print(album_cols[["name", "type"]])