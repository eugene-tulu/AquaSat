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

@st.cache_data(show_spinner=False)
def compute_ndci_batch(_items_dict, _bounds, _gdf_wkt):
    """
    Compute NDCI for all items in batch. Returns enriched DataFrame.
    _gdf_wkt is used for caching and included in output.
    """
    if not _items_dict:
        return pd.DataFrame()

    import planetary_computer

    signed_items = [planetary_computer.sign(item) for item in _items_dict.values()]

    stack = stackstac.stack(
        signed_items,
        assets=["B04", "B05"],
        resolution=10,
        epsg=6933,
        dtype="float32",
        bounds_latlon=_bounds
    )

    # Create AOI from WKT
    gdf = gpd.GeoDataFrame({'geometry': [gpd.GeoSeries.from_wkt([_gdf_wkt])[0]]}, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(stack.rio.crs)
    aoi_geom = gdf_proj.geometry.unary_union
    clipped = stack.rio.clip([aoi_geom], crs=stack.rio.crs)

    with ProgressBar():
        data = clipped.compute()

    red = data.sel(band="B04")
    red_edge = data.sel(band="B05")
    ndci = (red_edge - red) / (red_edge + red)

    results = []
    for i, (target_date, item) in enumerate(_items_dict.items()):
        ndci_slice = ndci.isel(time=i)
        mean_val = float(np.nanmean(ndci_slice.values))

        results.append({
            "requested_date": target_date,
            "image_date": item.datetime.strftime("%Y-%m-%d"),
            "ndci_mean": mean_val,
            "cloud_cover": item.properties.get("eo:cloud_cover", "N/A"),
            "satellite": "Sentinel-2",
            "geometry_wkt": _gdf_wkt  # Store for multi-polygon studies
        })

    return pd.DataFrame(results)

def classify_water_quality(ndci_value):
    if ndci_value < WQ_THRESHOLDS['low_algae']:
        return "Clear Water"
    elif ndci_value < WQ_THRESHOLDS['moderate_bloom']:
        return "Low Algae"
    elif ndci_value < WQ_THRESHOLDS['high_bloom']:
        return "Moderate Bloom"
    elif ndci_value < WQ_THRESHOLDS['severe_bloom']:
        return "High Bloom"
    else:
        return "Severe Bloom"

def get_gemini_water_summary(ndci_df):
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=(
                "You are an AI water quality expert. Analyze NDCI trends indicating chlorophyll-a levels. "
                "Summarize current risk level, note if conditions are worsening/improving, "
                "and suggest 1–2 specific actions for water managers (e.g., sampling, public advisory, nutrient control). "
                "Keep it under 150 characters if possible for SMS."
            )
        )

        mean_ndci = ndci_df['ndci_mean'].mean()
        recent = ndci_df.iloc[-1]
        trend = "stable"
        if len(ndci_df) > 1:
            delta = ndci_df['ndci_mean'].iloc[-1] - ndci_df['ndci_mean'].iloc[-2]
            trend = "improving" if delta < -0.05 else "worsening" if delta > 0.05 else "stable"

        prompt = (
            f"Water Body NDCI: Recent={recent['ndci_mean']:.3f} ({classify_water_quality(recent['ndci_mean'])}), "
            f"Avg={mean_ndci:.3f}, Trend={trend}. "
            f"Image used from {recent['image_date']} for requested date {recent['requested_date']}. "
            f"Provide concise management advice."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=150,
                temperature=0.2,
            )
        )

        text = response.text.strip()
        return text[:397] + "..." if len(text) > 400 else text

    except Exception as e:
        return f"AI summary error: {str(e)}"
    
