import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Geocoder, Draw
import geopandas as gpd
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import rioxarray
from dask.diagnostics import ProgressBar
import plotly.express as px
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ========================
# CONFIG & CONSTANTS
# ========================

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Water Quality Thresholds for NDCI
WQ_THRESHOLDS = {
    'clear': 0.0,
    'low_algae': 0.1,
    'moderate_bloom': 0.3,
    'high_bloom': 0.5,
    'severe_bloom': 0.7
}

MAX_AREA_KM2 = 10000  # For large lakes/reservoirs

# ========================
# SESSION STATE
# ========================

if 'ndci_results_df' not in st.session_state:
    st.session_state['ndci_results_df'] = None

# ========================
# HELPER FUNCTIONS
# ========================

@st.cache_data(show_spinner=False)
def fetch_items_for_dates(bounds, date_list, cloud_limit=10, window_days=3):
    """
    Batch-fetch Sentinel-2 items for dates.
    For each target date, finds closest image within ±window_days.
    If none, finds most recent image BEFORE target date (no future).
    Returns dict: {target_date: item}
    """
    client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    target_dates = pd.to_datetime(date_list).sort_values()
    min_search = (target_dates.min() - timedelta(days=window_days + 30)).strftime("%Y-%m-%d")  # Extend back for fallback
    max_search = (target_dates.max() + timedelta(days=window_days)).strftime("%Y-%m-%d")
    date_range = f"{min_search}/{max_search}"

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_limit}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}]
    )

    all_items = list(search.get_items())
    if not all_items:
        return {}

    items_by_date = {}
    item_datetimes = [pd.to_datetime(item.datetime) for item in all_items]

    for target_date_str in date_list:
        target_date = pd.to_datetime(target_date_str)

        # 1. Try within window
        candidates = [
            item for item, dt in zip(all_items, item_datetimes)
            if abs((dt - target_date).days) <= window_days
        ]

        if candidates:
            # Pick closest
            closest = min(candidates, key=lambda item: abs(pd.to_datetime(item.datetime) - target_date))
            items_by_date[target_date_str] = closest
            continue

        # 2. Fallback: Most recent BEFORE target date
        past_candidates = [
            item for item, dt in zip(all_items, item_datetimes)
            if dt <= target_date
        ]

        if past_candidates:
            most_recent = max(past_candidates, key=lambda item: pd.to_datetime(item.datetime))
            items_by_date[target_date_str] = most_recent
        else:
            st.warning(f"No image found before {target_date_str}")
            continue

    return items_by_date