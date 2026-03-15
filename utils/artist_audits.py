import re
import pandas as pd

from utils.helpers import normalise_apostrophe_caps


def normalise_artist_key(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()
    if not value:
        return pd.NA

    value = normalise_apostrophe_caps(value)
    value = re.sub(r"\s+", " ", value)
    value = value.casefold()

    return value if value else pd.NA


def _prepare_primary_artist_df(df):
    required_cols = ["primary_artist_name", "artist_ids"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    result = (
        df.loc[
            df["primary_artist_name"].notna() & df["artist_ids"].notna(),
            ["primary_artist_name", "artist_ids"],
        ]
        .copy()
    )

    result["primary_artist_name"] = (
        result["primary_artist_name"].astype(str).str.strip().apply(normalise_apostrophe_caps)
    )
    result["artist_ids"] = result["artist_ids"].astype(str).str.strip()
    result["artist_name_key"] = result["primary_artist_name"].apply(normalise_artist_key)

    return result


def audit_artist_name_to_id(df):
    """
    Flags cases where the same normalised primary artist name maps to multiple artist IDs.
    These are ambiguous and should be reviewed manually.
    """
    audit_df = _prepare_primary_artist_df(df)

    result = (
        audit_df.groupby("artist_name_key", dropna=False)
        .agg(
            raw_name_variants=("primary_artist_name", lambda s: sorted(set(map(str, s)))),
            distinct_artist_ids=("artist_ids", lambda s: sorted(set(map(str, s)))),
            artist_id_count=("artist_ids", "nunique"),
            row_count=("artist_ids", "size"),
        )
        .reset_index()
        .sort_values(["artist_id_count", "row_count", "artist_name_key"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    ambiguous = result.loc[result["artist_id_count"] > 1].copy()
    return ambiguous


def audit_artist_id_to_name(df):
    """
    Flags cases where the same artist ID appears with multiple primary artist name variants.
    These are usually safe to standardise automatically.
    """
    audit_df = _prepare_primary_artist_df(df)

    result = (
        audit_df.groupby("artist_ids", dropna=False)
        .agg(
            raw_name_variants=("primary_artist_name", lambda s: sorted(set(map(str, s)))),
            normalised_name_keys=("artist_name_key", lambda s: sorted(set(x for x in s.dropna()))),
            name_variant_count=("primary_artist_name", "nunique"),
            normalised_key_count=("artist_name_key", "nunique"),
            row_count=("primary_artist_name", "size"),
        )
        .reset_index()
        .sort_values(["name_variant_count", "row_count", "artist_ids"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    inconsistent = result.loc[result["name_variant_count"] > 1].copy()
    return inconsistent


def build_artist_id_canonical_map(df):
    """
    Builds one canonical primary artist name per artist_id.

    Rule:
    - use the most frequent cleaned variant
    - break ties alphabetically
    """
    audit_df = _prepare_primary_artist_df(df)

    canonical = (
        audit_df.groupby(["artist_ids", "primary_artist_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["artist_ids", "n", "primary_artist_name"], ascending=[True, False, True])
        .drop_duplicates(subset=["artist_ids"], keep="first")
        .rename(columns={"primary_artist_name": "canonical_primary_artist_name"})
        .loc[:, ["artist_ids", "canonical_primary_artist_name"]]
        .reset_index(drop=True)
    )

    return canonical


def apply_canonical_primary_artist_names(df):
    """
    Standardises primary_artist_name using the canonical value per artist_id.

    This only fixes same-ID / many-name issues.
    It does NOT merge many-ID / same-name cases.
    """
    required_cols = ["primary_artist_name", "artist_ids"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    result = df.copy()
    result["artist_ids"] = result["artist_ids"].astype(str).str.strip()

    canonical_map = build_artist_id_canonical_map(result)

    result = result.merge(canonical_map, on="artist_ids", how="left")

    result["primary_artist_name"] = result["canonical_primary_artist_name"].combine_first(
        result["primary_artist_name"]
    )

    result = result.drop(columns=["canonical_primary_artist_name"])

    if "artist_names" in result.columns:
        result["artist_names"] = result["artist_names"].apply(normalise_apostrophe_caps)

    return result


def build_artist_quality_report(df):
    """
    Convenience wrapper returning:
    - cleaned df with safe name standardisation applied
    - ambiguous same-name/multi-ID cases
    - inconsistent same-ID/multi-name cases
    """
    df_fixed = apply_canonical_primary_artist_names(df)
    ambiguous_name_to_id = audit_artist_name_to_id(df_fixed)
    inconsistent_id_to_name = audit_artist_id_to_name(df_fixed)

    return {
        "df_fixed": df_fixed,
        "ambiguous_name_to_id": ambiguous_name_to_id,
        "inconsistent_id_to_name": inconsistent_id_to_name,
    }
