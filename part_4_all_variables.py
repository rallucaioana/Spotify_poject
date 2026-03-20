import pandas as pd
import numpy as np
import sqlite3
import math
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"


con = sqlite3.connect("data/spotify_database.db")
cur = con.cursor()

# build one row per track while keeping a broad set of variables.
cur.execute(
    """
    WITH artist_per_track AS (
        SELECT
            al.track_id,
            AVG(ad.artist_popularity) AS artist_popularity,
            AVG(ad.followers) AS followers,
            GROUP_CONCAT(DISTINCT ad.id) AS artist_ids
        FROM albums_data al
        LEFT JOIN artist_data ad
            ON al.artist_id = ad.id
        GROUP BY al.track_id
    ),
    album_per_track AS (
        SELECT
            track_id,
            MAX(album_popularity) AS album_popularity,
            MIN(release_date) AS release_date,
            MAX(album_name) AS album_name,
            MAX(artist_0) AS primary_artist_name,
            MAX(
                TRIM(
                    COALESCE(NULLIF(artist_0, ''), '') ||
                    CASE WHEN NULLIF(artist_1, '') IS NOT NULL THEN ' | ' || artist_1 ELSE '' END ||
                    CASE WHEN NULLIF(artist_2, '') IS NOT NULL THEN ' | ' || artist_2 ELSE '' END ||
                    CASE WHEN NULLIF(artist_3, '') IS NOT NULL THEN ' | ' || artist_3 ELSE '' END ||
                    CASE WHEN NULLIF(artist_4, '') IS NOT NULL THEN ' | ' || artist_4 ELSE '' END ||
                    CASE WHEN NULLIF(artist_5, '') IS NOT NULL THEN ' | ' || artist_5 ELSE '' END ||
                    CASE WHEN NULLIF(artist_6, '') IS NOT NULL THEN ' | ' || artist_6 ELSE '' END
                )
            ) AS artist_names
        FROM albums_data
        GROUP BY track_id
    )
    SELECT
        t.*,
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
        f.duration_ms,
        f.time_signature,
        ap.artist_popularity,
        ap.followers,
        ap.artist_ids,
        albt.album_popularity,
        albt.release_date,
        albt.album_name,
        albt.primary_artist_name,
        albt.artist_names
    FROM tracks_data t
    LEFT JOIN features_data f
        ON t.id = f.id
    LEFT JOIN artist_per_track ap
        ON t.id = ap.track_id
    LEFT JOIN album_per_track albt
        ON t.id = albt.track_id
    WHERE t.id IS NOT NULL
        AND TRIM(t.id) <> ''
    """
)

rows = cur.fetchall()
cols = [d[0] for d in cur.description]
df_songs = pd.DataFrame(rows, columns=cols)

print("len after loading data:", len(df_songs))

# === 2. ===
def mask_invalid_data(df):
    m = pd.Series(False, index=df.index)

    def exists(col):
        return col in df.columns

    bounded_01 = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ]
    for col in bounded_01:
        if exists(col):
            m |= df[col].isna()
            m |= (df[col] < 0) | (df[col] > 1)

    for col in ["track_popularity", "artist_popularity", "album_popularity"]:
        if exists(col):
            m |= df[col].isna()
            m |= (df[col] < 0) | (df[col] > 100)

    if exists("followers"):
        m |= df["followers"].isna()
        m |= (df["followers"] < 0)

    if exists("tempo"):
        m |= df["tempo"].isna()
        m |= (df["tempo"] <= 0) | (df["tempo"] > 400)

    if exists("duration_ms"):
        m |= df["duration_ms"].isna()
        m |= (df["duration_ms"] <= 0)

    if exists("loudness"):
        m |= df["loudness"].isna()
        m |= (df["loudness"] < -80) | (df["loudness"] > 5)

    if exists("explicit"):
        explicit_norm = (
            df["explicit"].astype(str).str.strip().str.lower()
            .replace({"0": "false", "1": "true"})
        )
        m |= ~explicit_norm.isin(["true", "false"])

    return m


bad_mask = mask_invalid_data(df_songs)
df_clean = df_songs[~bad_mask].copy()
print("len cleaned df:", len(df_clean))
print("rows flagged as bad data:", int(bad_mask.sum()))
print("rows kept:", len(df_clean))


# === 1. ===
def detect_outliers_isolation_forest(df, columns, contamination="auto", random_state=42):
    X = df[columns].copy()

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled)

    pred = model.predict(X_scaled)
    score = model.decision_function(X_scaled)
    row_mask = pred == -1

    results = {
        "method": "IsolationForest",
        "contamination": contamination,
        "n_features_used": len(columns),
        "features_used": columns,
        "num_outliers": int(row_mask.sum()),
    }
    return results, row_mask, score


df_clean["followers_log"] = np.log1p(df_clean["followers"])

# variables per specification:
# tracks/artists whose features or popularity deviate significantly.
feature_cols = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

popularity_cols = [
    "track_popularity",
    "artist_popularity",
    "album_popularity",
    "followers_log"
]

candidate_features = feature_cols + popularity_cols
model_features = [c for c in candidate_features if c in df_clean.columns]

if not model_features:
    raise ValueError("No valid columns found for outlier detection.")

results, row_mask, anomaly_score = detect_outliers_isolation_forest(
    df_clean,
    columns=model_features,
    contamination=0.01,
    random_state=42,
)

df_clean["is_outlier"] = row_mask
df_clean["anomaly_score"] = anomaly_score

print("Outlier method:", results["method"])
print("Features used:", ", ".join(results["features_used"]))
print("Contamination:", results["contamination"])
print("Outliers flagged:", results["num_outliers"])
print("Rows before:", len(df_clean))
print("Rows removed:", int(row_mask.sum()))
print("Rows after:", int((df_clean["is_outlier"] == False).sum()))

# outlier handling:
# - flag: keep all rows, only mark outliers
# - remove: exclude outlier rows
# - winsorize: keep rows, cap extreme popularity tails
outlier_action = "flag"

if outlier_action == "remove":
    df_final = df_clean[~df_clean["is_outlier"]].copy()
elif outlier_action == "winsorize":
    df_final = df_clean.copy()
    winsor_cols = [
        c
        for c in ["track_popularity", "artist_popularity", "album_popularity", "followers_log"]
        if c in df_final.columns
    ]
    for col in winsor_cols:
        low = df_final[col].quantile(0.01)
        high = df_final[col].quantile(0.99)
        df_final[col] = df_final[col].clip(lower=low, upper=high)
else:
    df_final = df_clean.copy()

df_final = df_final.drop(columns=["followers_log"])
print("Outlier action:", outlier_action)
print("Final rows:", len(df_final))


# === 4. ===
def get_album_feature_summary(df, album_name, artist_name=None):

    if "album_name" not in df.columns:
        raise ValueError("df must contain an 'album_name' column.")

    album_mask = (
        df["album_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == album_name.strip().lower()
    )
    df_album = df[album_mask].copy()

    if artist_name is not None:
        artist_col = None
        for candidate in ["artist_names", "primary_artist_name", "artist_name", "artists"]:
            if candidate in df_album.columns:
                artist_col = candidate
                break

        if artist_col is None:
            raise ValueError("No artist name column found in df for artist filtering.")

        df_album = df_album[
            df_album[artist_col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(artist_name.strip().lower(), regex=False)
        ].copy()

    if df_album.empty:
        raise ValueError("No rows found for the requested album.")

    if "explicit" in df_album.columns:
        explicit_norm = (
            df_album["explicit"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"1": "true", "0": "false"})
        )
        explicit_track_count = int((explicit_norm == "true").sum())
        explicit_ratio = float((explicit_norm == "true").mean())
    else:
        explicit_track_count = np.nan
        explicit_ratio = np.nan

    summary = {
        "album_name": (
            df_album["album_name"].dropna().mode().iloc[0]
            if not df_album["album_name"].dropna().empty
            else album_name
        ),
        "primary_artist_name": (
            df_album["primary_artist_name"].dropna().mode().iloc[0]
            if "primary_artist_name" in df_album.columns and not df_album["primary_artist_name"].dropna().empty
            else np.nan
        ),
        "artist_names": (
            df_album["artist_names"].dropna().mode().iloc[0]
            if "artist_names" in df_album.columns and not df_album["artist_names"].dropna().empty
            else np.nan
        ),
        "release_date": (
            df_album["release_date"].dropna().min()
            if "release_date" in df_album.columns and not df_album["release_date"].dropna().empty
            else np.nan
        ),
        "track_count": (
            int(df_album["id"].nunique())
            if "id" in df_album.columns
            else int(len(df_album))
        ),
        "album_popularity": (
            float(df_album["album_popularity"].max())
            if "album_popularity" in df_album.columns and not df_album["album_popularity"].dropna().empty
            else np.nan
        ),
        "avg_track_popularity": (
            float(df_album["track_popularity"].mean())
            if "track_popularity" in df_album.columns
            else np.nan
        ),
        "median_track_popularity": (
            float(df_album["track_popularity"].median())
            if "track_popularity" in df_album.columns
            else np.nan
        ),
        "avg_artist_popularity": (
            float(df_album["artist_popularity"].mean())
            if "artist_popularity" in df_album.columns
            else np.nan
        ),
        "avg_followers": (
            float(df_album["followers"].mean())
            if "followers" in df_album.columns
            else np.nan
        ),
        "explicit_track_count": explicit_track_count,
        "explicit_ratio": explicit_ratio,
        "avg_danceability": (
            float(df_album["danceability"].mean())
            if "danceability" in df_album.columns
            else np.nan
        ),
        "avg_energy": (
            float(df_album["energy"].mean())
            if "energy" in df_album.columns
            else np.nan
        ),
        "avg_loudness": (
            float(df_album["loudness"].mean())
            if "loudness" in df_album.columns
            else np.nan
        ),
        "avg_speechiness": (
            float(df_album["speechiness"].mean())
            if "speechiness" in df_album.columns
            else np.nan
        ),
        "avg_acousticness": (
            float(df_album["acousticness"].mean())
            if "acousticness" in df_album.columns
            else np.nan
        ),
        "avg_instrumentalness": (
            float(df_album["instrumentalness"].mean())
            if "instrumentalness" in df_album.columns
            else np.nan
        ),
        "avg_liveness": (
            float(df_album["liveness"].mean())
            if "liveness" in df_album.columns
            else np.nan
        ),
        "avg_valence": (
            float(df_album["valence"].mean())
            if "valence" in df_album.columns
            else np.nan
        ),
        "avg_tempo": (
            float(df_album["tempo"].mean())
            if "tempo" in df_album.columns
            else np.nan
        ),
        "median_tempo": (
            float(df_album["tempo"].median())
            if "tempo" in df_album.columns
            else np.nan
        ),
        "min_tempo": (
            float(df_album["tempo"].min())
            if "tempo" in df_album.columns
            else np.nan
        ),
        "max_tempo": (
            float(df_album["tempo"].max())
            if "tempo" in df_album.columns
            else np.nan
        ),
        "avg_duration_ms": (
            float(df_album["duration_ms"].mean())
            if "duration_ms" in df_album.columns
            else np.nan
        ),
        "total_duration_ms": (
            float(df_album["duration_ms"].sum())
            if "duration_ms" in df_album.columns
            else np.nan
        ),
        "total_duration_min": (
            float(df_album["duration_ms"].sum() / 60000.0)
            if "duration_ms" in df_album.columns
            else np.nan
        ),
        "avg_time_signature": (
            float(df_album["time_signature"].mean())
            if "time_signature" in df_album.columns
            else np.nan
        ),
        "danceability_std": (
            float(df_album["danceability"].std())
            if "danceability" in df_album.columns and len(df_album) > 1
            else 0.0
        ),
        "energy_std": (
            float(df_album["energy"].std())
            if "energy" in df_album.columns and len(df_album) > 1
            else 0.0
        ),
        "valence_std": (
            float(df_album["valence"].std())
            if "valence" in df_album.columns and len(df_album) > 1
            else 0.0
        ),
        "tempo_std": (
            float(df_album["tempo"].std())
            if "tempo" in df_album.columns and len(df_album) > 1
            else 0.0
        ),
        "outlier_track_count": (
            int(df_album["is_outlier"].sum())
            if "is_outlier" in df_album.columns
            else 0
        ),
        "outlier_ratio": (
            float(df_album["is_outlier"].mean())
            if "is_outlier" in df_album.columns
            else 0.0
        ),
        "avg_anomaly_score": (
            float(df_album["anomaly_score"].mean())
            if "anomaly_score" in df_album.columns
            else np.nan
        ),
    }

    summary_df = pd.DataFrame([summary])

    return {
        "summary_df": summary_df,
        "track_df": df_album
    }


def get_album_feature_summary_split(df, album_name, artist_name=None):
    result = get_album_feature_summary(df, album_name, artist_name)
    s = result["summary_df"].iloc[0]

    metadata_df = pd.DataFrame([{
        "album_name": s["album_name"],
        "primary_artist_name": s["primary_artist_name"],
        "artist_names": s["artist_names"],
        "release_date": s["release_date"],
        "track_count": s["track_count"],
        "total_duration_min": s["total_duration_min"],
        "explicit_track_count": s["explicit_track_count"],
        "explicit_ratio": s["explicit_ratio"],
        "outlier_track_count": s["outlier_track_count"],
        "outlier_ratio": s["outlier_ratio"],
    }])

    popularity_df = pd.DataFrame([{
        "album_popularity": s["album_popularity"],
        "avg_track_popularity": s["avg_track_popularity"],
        "median_track_popularity": s["median_track_popularity"],
        "avg_artist_popularity": s["avg_artist_popularity"],
        "avg_followers": s["avg_followers"],
        "avg_anomaly_score": s["avg_anomaly_score"],
    }])

    audio_features_df = pd.DataFrame([{
        "avg_danceability": s["avg_danceability"],
        "avg_energy": s["avg_energy"],
        "avg_loudness": s["avg_loudness"],
        "avg_speechiness": s["avg_speechiness"],
        "avg_acousticness": s["avg_acousticness"],
        "avg_instrumentalness": s["avg_instrumentalness"],
        "avg_liveness": s["avg_liveness"],
        "avg_valence": s["avg_valence"],
        "avg_tempo": s["avg_tempo"],
        "avg_duration_ms": s["avg_duration_ms"],
        "avg_time_signature": s["avg_time_signature"],
    }])

    variability_df = pd.DataFrame([{
        "danceability_std": s["danceability_std"],
        "energy_std": s["energy_std"],
        "valence_std": s["valence_std"],
        "tempo_std": s["tempo_std"],
        "min_tempo": s["min_tempo"],
        "max_tempo": s["max_tempo"],
        "median_tempo": s["median_tempo"],
    }])

    return {
        "summary_df": result["summary_df"],
        "metadata_df": metadata_df,
        "popularity_df": popularity_df,
        "audio_features_df": audio_features_df,
        "variability_df": variability_df,
        "track_df": result["track_df"],
    }
    

feature_summary_result = get_album_feature_summary_split(
    df=df_final,
    album_name="the colour and the shape"
)

summary_df = feature_summary_result["summary_df"]
metadata_df = feature_summary_result["metadata_df"]
popularity_df = feature_summary_result["popularity_df"]
audio_features_df = feature_summary_result["audio_features_df"]
variability_df = feature_summary_result["variability_df"]
track_df = feature_summary_result["track_df"]

print(summary_df)
print(track_df)

print(summary_df.columns)
print(track_df.columns)

# === 5. ===


# === 6. ===


con.close()
