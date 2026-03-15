import sqlite3
import pandas as pd
import re
from utils.helpers import normalise_apostrophe_caps


def clean_text_columns(df):
    text_cols = [
        "album_name",
        "track_name",
        "primary_artist_name",
        "artist_names",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalise_apostrophe_caps)

    return df


def load_raw_spotify_data(db_path="data/spotify_database.db"):
    con = sqlite3.connect(db_path)

    # Filtering df first to speed up joins (due to potentially less rows being present)
    query = """
        WITH filtered_albums AS (
            SELECT *
            FROM albums_data
            WHERE album_id IS NOT NULL
            AND TRIM(album_id) <> ""
            AND track_id IS NOT NULL
            AND TRIM(track_id) <> ""
        )

        SELECT
            al.album_id,
            al.album_name,
            al.release_date,
            al.album_type,
            al.label,
            al.album_popularity,
            al.total_tracks,
            al.track_id AS id,
            al.track_name,
            al.track_number,
            al.duration_ms,
            al.artist_0 AS primary_artist_name,
            TRIM(
                COALESCE(NULLIF(al.artist_0, ""), "") ||
                CASE WHEN NULLIF(al.artist_1, "") IS NOT NULL THEN " | " || al.artist_1 ELSE "" END ||
                CASE WHEN NULLIF(al.artist_2, "") IS NOT NULL THEN " | " || al.artist_2 ELSE "" END ||
                CASE WHEN NULLIF(al.artist_3, "") IS NOT NULL THEN " | " || al.artist_3 ELSE "" END ||
                CASE WHEN NULLIF(al.artist_4, "") IS NOT NULL THEN " | " || al.artist_4 ELSE "" END ||
                CASE WHEN NULLIF(al.artist_5, "") IS NOT NULL THEN " | " || al.artist_5 ELSE "" END ||
                CASE WHEN NULLIF(al.artist_6, "") IS NOT NULL THEN " | " || al.artist_6 ELSE "" END
            ) AS artist_names,
            t.track_popularity,
            t.explicit,
            f.danceability,
            f.energy,
            f.key,
            f.loudness,
            f.mode,
            f.speechiness,
            f.acousticness,
            f.instrumentalness,
            f.liveness,
            f.valence,
            f.tempo,
            f.time_signature,
            ad.artist_popularity,
            ad.followers,
            ad.id AS artist_ids,
            ad.artist_genres,
            ad.genre_0,
            ad.genre_1,
            ad.genre_2,
            ad.genre_3,
            ad.genre_4,
            ad.genre_5,
            ad.genre_6

        FROM filtered_albums al

        LEFT JOIN tracks_data t
            ON al.track_id = t.id

        LEFT JOIN features_data f
            ON al.track_id = f.id

        LEFT JOIN artist_data ad
            ON al.artist_id = ad.id
    """

    df = pd.read_sql_query(query, con)
    con.close()
    
    df = clean_text_columns(df)

    return df
