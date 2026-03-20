import re
import pandas as pd
import tomllib

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def normalise_apostrophe_caps(text):
    if pd.isna(text):
        return text

    text = str(text)

    # Fix bad title-casing after apostrophes:
    # Don'T -> Don't
    # I'M -> I'm
    # Jojo'S -> Jojo's
    text = re.sub(r"'([A-Z])", lambda m: "'" + m.group(1).lower(), text)

    return text

def add_era_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "release_date" not in result.columns:
        raise ValueError("release_date column is required.")

    result["release_date"] = pd.to_datetime(result["release_date"], errors="coerce")
    result["release_year"] = result["release_date"].dt.year
    result["era_start"] = (result["release_year"] // 10) * 10
    result["era"] = result["era_start"].apply(
        lambda x: f"{int(x)}s" if pd.notna(x) else pd.NA
    )

    return result

def get_theme_colors(fill_alpha=0.45) -> tuple[str, str]:
    with open(".streamlit/config.toml", "rb") as f:
        config = tomllib.load(f)

    primary_color = config["theme"]["primaryColor"]
    primary_color_fill = hex_to_rgba(primary_color, fill_alpha)

    return primary_color, primary_color_fill

def prepare_year_column(df):
    df = df.copy()
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year

    df = df[df["release_year"].notna()].copy()

    return df

def filter_by_year_range(df, year_range):
    start_year, end_year = year_range

    return df[
        (df["release_year"] >= start_year) &
        (df["release_year"] <= end_year)
    ].copy()

def apply_iqr_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return df.copy()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[
        pd.to_numeric(df[column], errors="coerce").between(lower, upper)
    ].copy()
