import pandas as pd
import numpy as np

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
        m |= df["followers"] < 0

    if exists("tempo"):
        m |= df["tempo"].isna()
        m |= (df["tempo"] <= 0) | (df["tempo"] > 400)

    if exists("duration_ms"):
        m |= df["duration_ms"].isna()
        m |= df["duration_ms"] <= 0

    if exists("loudness"):
        m |= df["loudness"].isna()
        m |= (df["loudness"] < -80) | (df["loudness"] > 5)

    if exists("explicit"):
        explicit_norm = (
            df["explicit"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"0": "false", "1": "true"})
        )
        m |= ~explicit_norm.isin(["true", "false"])

    return m


def detect_outliers(df, columns, iqr_multiplier=1.5, min_flagged_features=1):
    if not columns:
        raise ValueError("No columns supplied for outlier detection.")

    X = df[columns].copy()

    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1

    # only use columns with actual spread
    valid_iqr = iqr > 0
    usable_cols = X.columns[valid_iqr]

    if len(usable_cols) == 0:
        result = df.copy()
        result["is_outlier"] = False
        return result

    lower_bounds = q1[usable_cols] - iqr_multiplier * iqr[usable_cols]
    upper_bounds = q3[usable_cols] + iqr_multiplier * iqr[usable_cols]

    outlier_mask = (
        X[usable_cols].lt(lower_bounds, axis=1)
        | X[usable_cols].gt(upper_bounds, axis=1)
    )

    flagged_count = outlier_mask.sum(axis=1)

    result = df.copy()
    result["is_outlier"] = flagged_count >= min_flagged_features

    return result


def add_release_date_columns(df):
    result = df.copy()

    if "release_date" in result.columns:
        result["release_date"] = pd.to_datetime(result["release_date"], errors="coerce")
        result["release_date_display"] = result["release_date"].dt.strftime("%d/%m/%Y")
        result["release_date_display"] = result["release_date_display"].fillna("Unknown")

    return result


def build_clean_dataset(df, outlier_action="flag"):
    df = add_release_date_columns(df)

    bad_mask = mask_invalid_data(df)
    df_clean = df.loc[~bad_mask].copy()
    
    if "explicit" in df_clean.columns:
        df_clean["explicit"] = (
            df_clean["explicit"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "1": True,
                "0": False,
                "true": True,
                "false": False,
            })
            .astype("boolean")
        )

    if "followers" in df_clean.columns:
        df_clean["followers_log"] = np.log1p(df_clean["followers"])

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
        "duration_ms",
    ]

    popularity_cols = [
        "track_popularity",
        "artist_popularity",
        "album_popularity",
        "followers_log",
    ]

    candidate_features = feature_cols + popularity_cols
    iqr_features = [c for c in candidate_features if c in df_clean.columns]

    if not iqr_features:
        raise ValueError("No valid columns found for outlier detection.")

    df_clean = detect_outliers(df_clean, iqr_features, iqr_multiplier=1.5, min_flagged_features=2)

    if outlier_action == "remove":
        df_final = df_clean.loc[~df_clean["is_outlier"]].copy()
    elif outlier_action == "winsorize":
        df_final = df_clean.copy()
        winsor_cols = [
            c for c in [
                "track_popularity",
                "artist_popularity",
                "album_popularity",
                "followers_log",
            ]
            if c in df_final.columns
        ]
        for col in winsor_cols:
            low = df_final[col].quantile(0.01)
            high = df_final[col].quantile(0.99)
            df_final[col] = df_final[col].clip(lower=low, upper=high)
    elif outlier_action == "flag":
        df_final = df_clean.copy()
    else:
        raise ValueError(
            "outlier_action must be one of: \"flag\", \"remove\", \"winsorize\"."
        )

    return df_final

