import pandas as pd
import numpy as np

# gets all collaborators from a release
def extract_album_collaborators(df_album, primary_artist_name):
    if "artist_names" not in df_album.columns:
        return []

    all_artists = set()

    for value in df_album["artist_names"].dropna():
        artists = [a.strip() for a in str(value).split("|") if a.strip()]
        all_artists.update(artists)

    primary_artist_name = str(primary_artist_name).strip()
    collaborators = sorted(a for a in all_artists if a != primary_artist_name)

    return collaborators


# gets all genres for the primary artist of a release
def extract_primary_artist_genres(df_album):
    genre_cols = [f"genre_{i}" for i in range(7) if f"genre_{i}" in df_album.columns]

    invalid_tokens = {"", "[]", "nan", "none", "null"}

    genres = set()

    for col in genre_cols:
        for value in df_album[col].dropna():
            value = str(value).strip()
            if value.lower() not in invalid_tokens:
                genres.add(value.title())

    if not genres and "artist_genres" in df_album.columns:
        for value in df_album["artist_genres"].dropna():
            for g in str(value).split("|"):
                g = g.strip()
                if g.lower() not in invalid_tokens:
                    genres.add(g.title())

    return sorted(genres)


# creates the feature summary of an album
def get_album_feature_summary(df, album_id):
    if "album_id" not in df.columns:
        raise ValueError("df must contain an \"album_id\" column.")

    df_album = df[df["album_id"].astype(str).str.strip() == str(album_id).strip()].copy()

    if df_album.empty:
        raise ValueError("No rows found for the requested album_id.")

    if "explicit" in df_album.columns:
        explicit_track_count = int(df_album["explicit"].sum())
        explicit_ratio = float(df_album["explicit"].mean())
    else:
        explicit_track_count = np.nan
        explicit_ratio = np.nan
        
    primary_artist_name = (
        df_album["primary_artist_name"].dropna().mode().iloc[0]
        if "primary_artist_name" in df_album.columns and not df_album["primary_artist_name"].dropna().empty
        else np.nan
    )

    collaborating_artists = extract_album_collaborators(df_album, primary_artist_name)
    
    primary_artist_genres = extract_primary_artist_genres(df_album)

    is_artist_name_ambiguous = (
        bool(df_album["is_artist_name_ambiguous"].fillna(False).any())
        if "is_artist_name_ambiguous" in df_album.columns
        else False
    )

    summary = {
        "album_id": album_id,
        "album_name": (
            df_album["album_name"].dropna().mode().iloc[0]
            if "album_name" in df_album.columns and not df_album["album_name"].dropna().empty
            else np.nan
        ),
        "primary_artist_name": primary_artist_name,
        "artist_names": (
            df_album["artist_names"].dropna().mode().iloc[0]
            if "artist_names" in df_album.columns and not df_album["artist_names"].dropna().empty
            else np.nan
        ),
        "collaborating_artists": collaborating_artists,
        "collaborating_artists_display": ", ".join(collaborating_artists) if collaborating_artists else "",
        "primary_artist_genres": primary_artist_genres,
        "primary_artist_genres_display": ", ".join(primary_artist_genres) if primary_artist_genres else "",
        "release_date": (
            df_album["release_date"].dropna().min()
            if "release_date" in df_album.columns and not df_album["release_date"].dropna().empty
            else pd.NaT
        ),
        "release_date_display": (
            df_album["release_date_display"].dropna().iloc[0]
            if "release_date_display" in df_album.columns and not df_album["release_date_display"].dropna().empty
            else "Unknown"
        ),
        "album_type": (
            df_album["album_type"].dropna().mode().iloc[0]
            if "album_type" in df_album.columns and not df_album["album_type"].dropna().empty
            else np.nan
        ),
        "label": (
            df_album["label"].dropna().mode().iloc[0]
            if "label" in df_album.columns and not df_album["label"].dropna().empty
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
        "primary_artist_popularity": (
            float(df_album["artist_popularity"].dropna().mode().iloc[0])
            if "artist_popularity" in df_album.columns and not df_album["artist_popularity"].dropna().empty
            else np.nan
        ),
        "primary_artist_followers": (
            float(df_album["followers"].dropna().mode().iloc[0])
            if "followers" in df_album.columns and not df_album["followers"].dropna().empty
            else np.nan
        ),
        "avg_danceability": float(df_album["danceability"].mean()) if "danceability" in df_album.columns else np.nan,
        "avg_energy": float(df_album["energy"].mean()) if "energy" in df_album.columns else np.nan,
        "avg_loudness": float(df_album["loudness"].mean()) if "loudness" in df_album.columns else np.nan,
        "avg_speechiness": float(df_album["speechiness"].mean()) if "speechiness" in df_album.columns else np.nan,
        "avg_acousticness": float(df_album["acousticness"].mean()) if "acousticness" in df_album.columns else np.nan,
        "avg_instrumentalness": float(df_album["instrumentalness"].mean()) if "instrumentalness" in df_album.columns else np.nan,
        "avg_liveness": float(df_album["liveness"].mean()) if "liveness" in df_album.columns else np.nan,
        "avg_valence": float(df_album["valence"].mean()) if "valence" in df_album.columns else np.nan,
        "avg_tempo": float(df_album["tempo"].mean()) if "tempo" in df_album.columns else np.nan,
        "median_tempo": float(df_album["tempo"].median()) if "tempo" in df_album.columns else np.nan,
        "min_tempo": float(df_album["tempo"].min()) if "tempo" in df_album.columns else np.nan,
        "max_tempo": float(df_album["tempo"].max()) if "tempo" in df_album.columns else np.nan,
        "avg_duration_ms": float(df_album["duration_ms"].mean()) if "duration_ms" in df_album.columns else np.nan,
        "total_duration_ms": float(df_album["duration_ms"].sum()) if "duration_ms" in df_album.columns else np.nan,
        "total_duration_min": float(df_album["duration_ms"].sum() / 60000.0) if "duration_ms" in df_album.columns else np.nan,
        "avg_time_signature": float(df_album["time_signature"].mean()) if "time_signature" in df_album.columns else np.nan,
        "danceability_std": float(df_album["danceability"].std()) if "danceability" in df_album.columns and len(df_album) > 1 else 0.0,
        "energy_std": float(df_album["energy"].std()) if "energy" in df_album.columns and len(df_album) > 1 else 0.0,
        "valence_std": float(df_album["valence"].std()) if "valence" in df_album.columns and len(df_album) > 1 else 0.0,
        "tempo_std": float(df_album["tempo"].std()) if "tempo" in df_album.columns and len(df_album) > 1 else 0.0,
        "outlier_track_count": int(df_album["is_outlier"].sum()) if "is_outlier" in df_album.columns else 0,
        "outlier_ratio": float(df_album["is_outlier"].mean()) if "is_outlier" in df_album.columns else 0.0,
        "is_artist_name_ambiguous": is_artist_name_ambiguous,
        "explicit_track_count": explicit_track_count,
        "explicit_ratio": explicit_ratio
    }

    summary_df = pd.DataFrame([summary])

    return {
        "summary_df": summary_df,
        "track_df": df_album
    }


# splits the created feature summary in separate dfs
def get_album_feature_summary_split(df, album_id, track_df=None):
    result = get_album_feature_summary(df, album_id)
    s = result["summary_df"].iloc[0]

    metadata_df = pd.DataFrame([{
        "album_name": s["album_name"],
        "primary_artist_name": s["primary_artist_name"],
        "artist_names": s["artist_names"],
        "collaborating_artists": s["collaborating_artists"],
        "collaborating_artists_display": s["collaborating_artists_display"],
        "primary_artist_genres": s["primary_artist_genres"],
        "primary_artist_genres_display": s["primary_artist_genres_display"],
        "release_date": s["release_date"],
        "release_date_display": s["release_date_display"],
        "album_type": s["album_type"],
        "label": s["label"],
        "track_count": s["track_count"],
        "explicit_track_count": s["explicit_track_count"],
        "explicit_ratio": s["explicit_ratio"],
        "total_duration_ms": s["total_duration_ms"],
        "total_duration_min": s["total_duration_min"],
        "is_artist_name_ambiguous": s["is_artist_name_ambiguous"],
    }])

    popularity_df = pd.DataFrame([{
        "album_popularity": s["album_popularity"],
        "avg_track_popularity": s["avg_track_popularity"],
        "median_track_popularity": s["median_track_popularity"],
        "primary_artist_popularity": s["primary_artist_popularity"],
        "primary_artist_followers": s["primary_artist_followers"]
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

    if track_df is None:
        track_df = result["track_df"]

    return {
        "summary_df": result["summary_df"],
        "metadata_df": metadata_df,
        "popularity_df": popularity_df,
        "audio_features_df": audio_features_df,
        "variability_df": variability_df,
        "track_df": track_df,
    }

# gets the count of explicit songs on a release
def get_explicit_counts(track_df):
    explicit_df = track_df["explicit"].map({True: "Explicit", False: "Non-Explicit"}).value_counts().reset_index()
    explicit_df.columns = ["type", "count"]
    return explicit_df

# gets the popularity of the explicit songs on a release
def get_explicit_popularity(track_df):
    popularity_df = track_df.groupby(
        track_df["explicit"].map({True: "Explicit", False: "Non-Explicit"})
    )["track_popularity"].mean().reset_index()
    popularity_df.columns = ["type", "avg_popularity"]
    popularity_df["avg_popularity"] = popularity_df["avg_popularity"].round(1)
    return popularity_df
