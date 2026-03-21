import re
import pandas as pd

from utils.helpers import normalise_apostrophe_caps


REVIEWED_ARTIST_NAME_OVERRIDES = [
    # Use this to manually resolve ambiguous name->ID cases that auto-resolution
    # missed or got wrong. Each entry pins an artist_name_key to a preferred_artist_id.
    #
    # Example:
    # {
    #     "artist_name_key": "prince",
    #     "preferred_artist_id": "5a2EaR3hamoenG9rDuVn8j",
    #     "resolution_status": "reviewed_keep_preferred_id",
    #     "review_notes": "Multiple 'Prince' artists in DB; prefer the Purple Rain artist.",
    # },
]

# Auto-resolution parameters
# A name->multiID group is auto-resolved when:
#   - total rows in the group  >= MIN_ROWS_FOR_RESOLUTION
#   - the single dominant ID accounts for >= DOMINANCE_THRESHOLD of those rows
#
# Raise DOMINANCE_THRESHOLD toward 1.0 to be more conservative (fewer auto-resolves).
# Lower it toward 0.5 to be more aggressive (resolve closer 50/50 splits).
DOMINANCE_THRESHOLD: float = 0.60
MIN_ROWS_FOR_RESOLUTION: int = 5

# Groups that didn't meet the dominance threshold but have at most TIEBREAK_MAX_IDS
# distinct IDs are resolved by a deterministic tiebreak (alphabetically first artist_id).
# This is safe for 2-ID groups — two profiles for the same artist on Spotify is the
# overwhelmingly likely explanation. Set to 0 or 1 to disable tiebreak resolution.
TIEBREAK_MAX_IDS: int = 2


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
        result["primary_artist_name"]
        .astype(str)
        .str.strip()
        .apply(normalise_apostrophe_caps)
    )
    result["artist_ids"] = result["artist_ids"].astype(str).str.strip()
    result["artist_name_key"] = result["primary_artist_name"].apply(normalise_artist_key)

    return result


# Audit helpers
def audit_artist_name_to_id(df):
    """Same normalised artist name mapped to multiple artist IDs."""
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
        .sort_values(
            ["artist_id_count", "row_count", "artist_name_key"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    return result.loc[result["artist_id_count"] > 1].copy()


def audit_artist_id_to_name(df):
    """Same artist ID appearing with multiple name variants."""
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
        .sort_values(
            ["name_variant_count", "row_count", "artist_ids"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    return result.loc[result["name_variant_count"] > 1].copy()


# Name canonicalization (same ID, multiple names)
def build_artist_id_canonical_map(df):
    """One canonical primary artist name per artist ID (most-frequent, alphabetical tie-break)."""
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
    """Standardises display names for same-ID / multi-name cases."""
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


# Dominance-based auto-resolution (same name, multiple IDs)
def build_auto_resolution_map(
    df,
    dominance_threshold: float = DOMINANCE_THRESHOLD,
    min_total_rows: int = MIN_ROWS_FOR_RESOLUTION):
    """
    Identifies name->multiID groups where one artist_id dominates enough to be
    treated as the canonical ID automatically.

    Eligibility criteria (both must hold):
      - total rows for the name_key >= min_total_rows
      - the single most-common artist_id accounts for >= dominance_threshold
        of the group's rows

    Returns a DataFrame with one row per resolvable name_key:
      artist_name_key | canonical_artist_id | dominant_row_count |
      total_row_count | dominance_ratio     | artist_id_count
    """
    audit_df = _prepare_primary_artist_df(df)

    counts = (
        audit_df.groupby(["artist_name_key", "artist_ids"], dropna=False)
        .size()
        .reset_index(name="id_row_count")
    )

    totals = (
        counts.groupby("artist_name_key")
        .agg(
            total_row_count=("id_row_count", "sum"),
            artist_id_count=("artist_ids", "nunique"),
        )
        .reset_index()
    )

    counts = counts.merge(totals, on="artist_name_key")
    counts["dominance_ratio"] = counts["id_row_count"] / counts["total_row_count"]

    # Only groups with more than one distinct ID are candidates
    multi = counts[counts["artist_id_count"] > 1].copy()

    # Pick the most-used ID per group (tie-break: alphabetical artist_id)
    best = (
        multi.sort_values(
            ["artist_name_key", "id_row_count", "artist_ids"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["artist_name_key"], keep="first")
    )

    eligible = best[
        (best["dominance_ratio"] >= dominance_threshold)
        & (best["total_row_count"] >= min_total_rows)
    ].copy()

    return (
        eligible
        .rename(columns={"artist_ids": "canonical_artist_id", "id_row_count": "dominant_row_count"})
        [[
            "artist_name_key",
            "canonical_artist_id",
            "dominant_row_count",
            "total_row_count",
            "dominance_ratio",
            "artist_id_count",
        ]]
        .reset_index(drop=True)
    )


def build_tiebreak_resolution_map(
    df,
    dominance_threshold: float = DOMINANCE_THRESHOLD,
    min_total_rows: int = MIN_ROWS_FOR_RESOLUTION,
    tiebreak_max_ids: int = TIEBREAK_MAX_IDS):
    """
    Finds name->multiID groups that were NOT resolved by dominance (either because
    no single ID was dominant enough, or total rows were below min_total_rows) but
    have at most ``tiebreak_max_ids`` distinct IDs.

    For these groups the alphabetically-first artist_id is selected as canonical.
    This is a safe assumption for 2-ID cases: two Spotify profiles for the same
    artist is far more likely than two genuinely different artists sharing a name.
    Groups with 3+ IDs are always left for manual review.

    Returns a DataFrame with the same schema as build_auto_resolution_map, plus a
    ``resolution_method`` column set to ``"tiebreak"`` for all rows.
    """
    if tiebreak_max_ids < 2:
        return pd.DataFrame(
            columns=[
                "artist_name_key", "canonical_artist_id", "dominant_row_count",
                "total_row_count", "dominance_ratio", "artist_id_count", "resolution_method",
            ]
        )

    # IDs already resolved by dominance — exclude them
    dominance_resolved = build_auto_resolution_map(df, dominance_threshold, min_total_rows)
    already_resolved_keys = set(dominance_resolved["artist_name_key"])

    audit_df = _prepare_primary_artist_df(df)

    counts = (
        audit_df.groupby(["artist_name_key", "artist_ids"], dropna=False)
        .size()
        .reset_index(name="id_row_count")
    )

    totals = (
        counts.groupby("artist_name_key")
        .agg(
            total_row_count=("id_row_count", "sum"),
            artist_id_count=("artist_ids", "nunique"),
        )
        .reset_index()
    )

    counts = counts.merge(totals, on="artist_name_key")
    counts["dominance_ratio"] = counts["id_row_count"] / counts["total_row_count"]

    # Only multi-ID groups not already resolved
    candidates = counts[
        (counts["artist_id_count"] > 1)
        & (counts["artist_id_count"] <= tiebreak_max_ids)
        & (~counts["artist_name_key"].isin(already_resolved_keys))
    ].copy()

    # Pick alphabetically-first artist_id as canonical (deterministic, reproducible)
    tiebreak = (
        candidates.sort_values(["artist_name_key", "artist_ids"])
        .drop_duplicates(subset=["artist_name_key"], keep="first")
    )

    tiebreak = tiebreak.rename(
        columns={"artist_ids": "canonical_artist_id", "id_row_count": "dominant_row_count"}
    )[[
        "artist_name_key", "canonical_artist_id", "dominant_row_count",
        "total_row_count", "dominance_ratio", "artist_id_count",
    ]].reset_index(drop=True)

    tiebreak["resolution_method"] = "tiebreak"

    return tiebreak


def resolve_ambiguous_artist_ids(
    df,
    dominance_threshold: float = DOMINANCE_THRESHOLD,
    min_total_rows: int = MIN_ROWS_FOR_RESOLUTION,
    tiebreak_max_ids: int = TIEBREAK_MAX_IDS,):
    """
    Auto-resolves same-name / multi-ID ambiguity in two passes:

    Pass 1 - Dominance: groups where one artist_id accounts for >= dominance_threshold
      of rows (and total rows >= min_total_rows) are remapped to that dominant ID.

    Pass 2 - Tiebreak: groups that weren't resolved in pass 1 but have at most
      tiebreak_max_ids distinct IDs (default: 2) are resolved by picking the
      alphabetically-first artist_id. Two-ID groups are overwhelmingly likely to
      be duplicate Spotify profiles for the same artist rather than genuinely
      different artists sharing a name. Groups with 3+ IDs are left for manual
      review via REVIEWED_ARTIST_NAME_OVERRIDES.

    apply_canonical_primary_artist_names is re-run after both passes so display
    names stay consistent with updated IDs.

    Returns
    -------
    df_resolved : pd.DataFrame
        Input dataframe with corrected artist_ids (and primary_artist_name).
        Boolean column ``auto_resolved_artist_id`` marks rows that were changed.
    resolution_summary : pd.DataFrame
        One row per resolved artist_name_key. Includes a ``resolution_method``
        column: ``"dominance"`` for pass-1 resolutions, ``"tiebreak"`` for pass-2.
    """
    dominance_map = build_auto_resolution_map(df, dominance_threshold, min_total_rows)
    dominance_map["resolution_method"] = "dominance"

    tiebreak_map = build_tiebreak_resolution_map(
        df, dominance_threshold, min_total_rows, tiebreak_max_ids
    )

    resolution_map_df = pd.concat([dominance_map, tiebreak_map], ignore_index=True)

    result = df.copy()
    result["artist_ids"] = result["artist_ids"].astype(str).str.strip()
    result["_name_key"] = result["primary_artist_name"].apply(normalise_artist_key)

    if resolution_map_df.empty:
        result["auto_resolved_artist_id"] = False
        result = result.drop(columns=["_name_key"])
        return result, resolution_map_df

    key_to_canonical = dict(
        zip(resolution_map_df["artist_name_key"], resolution_map_df["canonical_artist_id"])
    )

    canonical_series = result["_name_key"].map(key_to_canonical)
    needs_remap = canonical_series.notna() & (result["artist_ids"] != canonical_series)

    result.loc[needs_remap, "artist_ids"] = canonical_series[needs_remap]
    result["auto_resolved_artist_id"] = needs_remap
    result = result.drop(columns=["_name_key"])

    # Re-canonicalize names now that IDs have changed
    result = apply_canonical_primary_artist_names(result)

    # Attach remapped row counts to the summary
    remapped_counts = (
        result[result["auto_resolved_artist_id"]]
        .assign(_name_key=lambda d: d["primary_artist_name"].apply(normalise_artist_key))
        .groupby("_name_key")
        .size()
        .reset_index(name="remapped_row_count")
        .rename(columns={"_name_key": "artist_name_key"})
    )

    # Drop the working column — it's an internal computation artifact
    result = result.drop(columns=["auto_resolved_artist_id"])

    resolution_summary = resolution_map_df.merge(remapped_counts, on="artist_name_key", how="left")
    resolution_summary["remapped_row_count"] = (
        resolution_summary["remapped_row_count"].fillna(0).astype(int)
    )

    return result, resolution_summary


# Ambiguity flagging
def flag_ambiguous_primary_artist_names(df):
    """
    Flags rows whose artist_name_key still maps to multiple IDs after auto-resolution.
    Only unresolved cases (those that didn't meet the dominance threshold, or were
    genuinely different artists) will be flagged here.
    """
    result = df.copy()

    if "primary_artist_name" not in result.columns:
        result["is_artist_name_ambiguous"] = False
        result["artist_collision_bucket"] = "not_applicable"
        return result

    ambiguous = audit_artist_name_to_id(result)
    ambiguous_keys = set(ambiguous["artist_name_key"].dropna())

    result["artist_name_key"] = result["primary_artist_name"].apply(normalise_artist_key)
    result["is_artist_name_ambiguous"] = result["artist_name_key"].isin(ambiguous_keys)

    result["artist_collision_bucket"] = "unique_name"
    result.loc[result["is_artist_name_ambiguous"], "artist_collision_bucket"] = "same_name_multi_id"

    return result


# Manual override layer

def build_reviewed_override_table():
    if not REVIEWED_ARTIST_NAME_OVERRIDES:
        return pd.DataFrame(
            columns=[
                "artist_name_key",
                "preferred_artist_id",
                "resolution_status",
                "review_notes",
            ]
        )

    return pd.DataFrame(REVIEWED_ARTIST_NAME_OVERRIDES).copy()


def apply_reviewed_artist_overrides(df):
    """
    Optional manual review layer. Annotates rows with override metadata and
    marks which rows match the preferred_artist_id.
    """
    result = df.copy()

    if "primary_artist_name" not in result.columns or "artist_ids" not in result.columns:
        raise ValueError("df must contain 'primary_artist_name' and 'artist_ids'.")

    result["artist_ids"] = result["artist_ids"].astype(str).str.strip()
    result["artist_name_key"] = result["primary_artist_name"].apply(normalise_artist_key)

    overrides = build_reviewed_override_table()
    if overrides.empty:
        result["artist_resolution_status"] = pd.NA
        result["artist_review_notes"] = pd.NA
        result["is_preferred_artist_id"] = pd.NA
        return result

    overrides["artist_name_key"] = overrides["artist_name_key"].apply(normalise_artist_key)
    overrides["preferred_artist_id"] = overrides["preferred_artist_id"].astype("string")

    result = result.merge(overrides, on="artist_name_key", how="left")

    result["is_preferred_artist_id"] = (
        result["preferred_artist_id"].notna()
        & (result["artist_ids"] == result["preferred_artist_id"])
    )

    return result


# Top-level pipeline entry point
def build_artist_quality_report(df):
    """
    Full artist data-quality pipeline. Steps in order:

    1. Canonicalize display names for same-ID / multi-name variants.
    2. Auto-resolve same-name / multi-ID cases in two passes:
         a. Dominance: one ID accounts for >= DOMINANCE_THRESHOLD of rows.
         b. Tiebreak: remaining groups with <= TIEBREAK_MAX_IDS distinct IDs
            are resolved by picking the alphabetically-first artist_id.
            Groups with 3+ IDs are left for manual review.
    3. Re-canonicalize display names after ID remapping.
    4. Flag any remaining unresolved ambiguous name->ID pairs (3+ ID groups
       that didn't meet any resolution criteria).
    5. Apply REVIEWED_ARTIST_NAME_OVERRIDES annotations for manual cases.

    Returns
    -------
    dict with keys:
      df_fixed                   - cleaned dataframe ready for downstream use
      ambiguous_name_to_id       - remaining unresolved name->multiID cases
      inconsistent_id_to_name    - same-ID / multi-name cases (post-fix)
      reviewed_artist_name_overrides  - contents of REVIEWED_ARTIST_NAME_OVERRIDES
      auto_resolved_artist_ids   - summary DataFrame of what step 2 resolved,
                                   including a ``resolution_method`` column
    """
    # Step 1 – fix same-ID name variants
    df_fixed = apply_canonical_primary_artist_names(df)

    # Step 2 – auto-resolve dominant same-name multi-ID cases
    df_fixed, auto_resolved_summary = resolve_ambiguous_artist_ids(df_fixed)

    # Step 3 – re-canonicalize names after ID remapping
    df_fixed = apply_canonical_primary_artist_names(df_fixed)

    # Step 4 – flag remaining ambiguous cases
    df_fixed = flag_ambiguous_primary_artist_names(df_fixed)

    # Step 5 – apply manual overrides
    df_fixed = apply_reviewed_artist_overrides(df_fixed)

    ambiguous_name_to_id = audit_artist_name_to_id(df_fixed)
    inconsistent_id_to_name = audit_artist_id_to_name(df_fixed)
    reviewed_artist_name_overrides = build_reviewed_override_table()

    # Drop internal pipeline columns that have no downstream use in the UI.
    # is_artist_name_ambiguous is kept — it is read by album_features.py.
    _internal_cols = [
        "artist_name_key",
        "artist_collision_bucket",
        "artist_resolution_status",
        "artist_review_notes",
        "is_preferred_artist_id",
    ]
    df_fixed = df_fixed.drop(columns=[c for c in _internal_cols if c in df_fixed.columns])

    return {
        "df_fixed": df_fixed,
        "ambiguous_name_to_id": ambiguous_name_to_id,
        "inconsistent_id_to_name": inconsistent_id_to_name,
        "reviewed_artist_name_overrides": reviewed_artist_name_overrides,
        "auto_resolved_artist_ids": auto_resolved_summary,
    }

# TOP ARTISTS BY FEATURE

def get_top_artists_by_feature(df: pd.DataFrame, feature: str, top_n=10, min_tracks=3):

    # check if selected feature exists in the dataframe
    if feature not in df.columns:
        raise ValueError(f"{feature} not found in dataframe")

    artist_col = "primary_artist_name"

    df_valid = df[[artist_col, feature]].dropna()

    #group by artist and calculate avg feature value and number of tracks per artist
    grouped = (
        df_valid
        .groupby(artist_col)
        .agg(
            avg_feature=(feature, "mean"),
            track_count=(feature, "count")
        )
        .reset_index()
    )
    
    #filter artists with too few tracks
    grouped = grouped[grouped["track_count"] >= min_tracks]

    #sort artists by feature value(highest first)
    grouped = grouped.sort_values(by="avg_feature", ascending=False)

    return grouped.head(top_n)
    
def get_bottom_artists_by_feature(df: pd.DataFrame, feature: str, top_n=10, min_tracks=3):

    if feature not in df.columns:
        raise ValueError(f"{feature} not found in dataframe")

    artist_col = "primary_artist_name"

    df_valid = df[[artist_col, feature]].dropna()

    grouped = (
        df_valid
        .groupby(artist_col)
        .agg(
            avg_feature=(feature, "mean"),
            track_count=(feature, "count")
        )
        .reset_index()
    )

    grouped = grouped[grouped["track_count"] >= min_tracks]
    
    #soart artists by feature value(lowest first)
    grouped = grouped.sort_values(by="avg_feature", ascending=True)

    return grouped.head(top_n)