import pandas as pd
import numpy as np
from typing import Tuple, Optional

def is_valid_gps(lat: float, lon: float) -> bool:
    """Checks if latitude and longitude are valid non-NaN float values within standard WGS84 bounds."""
    try:
        if pd.isna(lat) or pd.isna(lon):
            return False
        lat_f = float(lat)
        lon_f = float(lon)
        if lat_f == 0.0 and lon_f == 0.0:
            return False
        return (-90.0 <= lat_f <= 90.0) and (-180.0 <= lon_f <= 180.0)
    except Exception:
        return False

def filter_valid_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataframe to only rows containing valid GPS coordinates."""
    if df is None or df.empty:
        return pd.DataFrame()
        
    valid_mask = df.apply(lambda row: is_valid_gps(row.get("latitude"), row.get("longitude")), axis=1)
    return df[valid_mask].copy()
