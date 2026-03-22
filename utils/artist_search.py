import pandas as pd
import numpy as np

from utils.album_features import extract_primary_artist_genres
from utils.helpers import normalise_apostrophe_caps


#  builds a deduplicated artist list for the search dropdown, sorted by artist popularity.
def build_artist_selector_df(df: pd.DataFrame):
    
    required = ["artist_ids", "primary_artist_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    cols = required + [c for c in ["artist_popularity", "followers"] if c in df.columns]

    artist_df = (
        df.loc[df["artist_ids"].notna() & df["primary_artist_name"].notna(), cols]
        .drop_duplicates(subset=["artist_ids"])
        .copy()
    )

    sort_cols = (
        ["artist_popularity", "primary_artist_name"]
        if "artist_popularity" in artist_df.columns
        else ["primary_artist_name"]
    )
    sort_order = [False, True] if "artist_popularity" in artist_df.columns else [True]

    artist_df = (
        artist_df
        .sort_values(sort_cols, ascending=sort_order)
        .reset_index(drop=True)
    )

    if "artist_popularity" in artist_df.columns:
        popularity_str = (
            pd.to_numeric(artist_df["artist_popularity"], errors="coerce")
            .fillna(0)
            .round(0)
            .astype(int)
            .astype(str)
        )
    else:
        popularity_str = pd.Series(["N/A"] * len(artist_df), index=artist_df.index)

    artist_df["dropdown_label"] = (
        artist_df["primary_artist_name"].astype(str).str.strip()
        + " | Popularity: "
        + popularity_str
    )

    return artist_df


# calculates an artist's overview summary statistics
def get_artist_overview(df: pd.DataFrame, artist_id: str):
    df_artist = df[df["artist_ids"].astype(str).str.strip() == str(artist_id).strip()].copy()

    if df_artist.empty:
        raise ValueError(f"No data found for artist_id: {artist_id}")

    primary_artist_name = (
        df_artist["primary_artist_name"].dropna().mode().iloc[0]
        if "primary_artist_name" in df_artist.columns and not df_artist["primary_artist_name"].dropna().empty
        else "Unknown"
    )

    followers = (
        float(df_artist["followers"].dropna().mode().iloc[0])
        if "followers" in df_artist.columns and not df_artist["followers"].dropna().empty
        else np.nan
    )

    artist_popularity = (
        float(df_artist["artist_popularity"].dropna().mode().iloc[0])
        if "artist_popularity" in df_artist.columns and not df_artist["artist_popularity"].dropna().empty
        else np.nan
    )

    genres = extract_primary_artist_genres(df_artist)

    releases_df = (
        df_artist.loc[df_artist["album_id"].notna()]
        .drop_duplicates(subset=["album_id"])
        .copy()
    )

    total_releases = len(releases_df)

    total_tracks = (
        int(df_artist["id"].nunique())
        if "id" in df_artist.columns
        else len(df_artist)
    )

    release_years = (
        pd.to_datetime(df_artist["release_date"], errors="coerce")
        .dt.year
        .dropna()
    )
    first_release_year = int(release_years.min()) if not release_years.empty else None
    last_release_year = int(release_years.max()) if not release_years.empty else None

    releases_by_type: dict = {}
    if "album_type" in releases_df.columns:
        releases_by_type = (
            releases_df["album_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .to_dict()
        )

    return {
        "primary_artist_name": primary_artist_name,
        "followers": followers,
        "artist_popularity": artist_popularity,
        "genres": genres,
        "total_releases": total_releases,
        "total_tracks": total_tracks,
        "first_release_year": first_release_year,
        "last_release_year": last_release_year,
        "releases_by_type": releases_by_type,
    }


# gets the artist's releases and limited metadata about the releases
def get_artist_releases(df: pd.DataFrame, artist_id: str):

    df_artist = df[df["artist_ids"].astype(str).str.strip() == str(artist_id).strip()].copy()

    if df_artist.empty:
        return pd.DataFrame()

    base_cols = ["album_id", "album_name", "release_date", "album_type", "album_popularity"]
    available_cols = [c for c in base_cols if c in df_artist.columns]

    releases = (
        df_artist.loc[df_artist["album_id"].notna(), available_cols]
        .drop_duplicates(subset=["album_id"])
        .copy()
    )

    if "album_name" in releases.columns:
        releases["album_name"] = releases["album_name"].apply(normalise_apostrophe_caps)

    releases["release_date"] = pd.to_datetime(releases["release_date"], errors="coerce")
    releases["release_year"] = releases["release_date"].dt.year

    # second deduplication pass: same name + type + year with different album_ids are
    # regional duplicates. Keeps the copy with the highest popularity.
    if "album_popularity" in releases.columns:
        releases = (
            releases
            .sort_values("album_popularity", ascending=False)
            .drop_duplicates(subset=["album_name", "album_type", "release_year"], keep="first")
        )
    else:
        releases = releases.drop_duplicates(subset=["album_name", "album_type", "release_year"], keep="first")

    releases = releases.sort_values("release_date", ascending=False).reset_index(drop=True)

    if "id" in df_artist.columns:
        track_counts = (
            df_artist.groupby("album_id")["id"]
            .nunique()
            .reset_index(name="track_count")
        )
        releases = releases.merge(track_counts, on="album_id", how="left")

    return releases