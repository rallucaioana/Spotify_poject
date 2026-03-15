import re
import pandas as pd

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