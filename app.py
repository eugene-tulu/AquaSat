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
    # FIXED: Removed trailing spaces in URL
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

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(layout="wide", page_title="💧 AquaSat Pro")
st.title("💧 AquaSat Pro - Advanced Water Quality Monitor")
st.markdown("Track chlorophyll & algal blooms using Sentinel-2. Handles missing dates intelligently.")

# Sidebar
with st.sidebar:
    st.header("🛰️ Configuration")

    uploaded_file = st.file_uploader("Upload Water Body (GeoJSON)", type=["geojson"])
    
    date_option = st.radio("Date Input", ["Date Range", "Upload CSV"], horizontal=True)
    
    if date_option == "Date Range":
        end_date = st.date_input("End Date", datetime.now())
        start_date = st.date_input("Start Date", end_date - timedelta(days=60))
        freq = st.selectbox("Frequency", ["Daily", "Weekly", "Biweekly"], index=1)
        freq_map = {"Daily": 'D', "Weekly": 'W', "Biweekly": '2W'}
        date_list = pd.date_range(start=start_date, end=end_date, freq=freq_map[freq]).strftime("%Y-%m-%d").tolist()
    else:
        date_file = st.file_uploader("Upload Dates CSV", type=["csv"])
        if date_file:
            df = pd.read_csv(date_file)
            if 'date' not in df.columns:
                st.error("CSV must have 'date' column")
                date_list = []
            else:
                df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
                date_list = df['date'].dropna().unique().tolist()
                st.success(f"Loaded {len(date_list)} dates")
        else:
            date_list = []

    cloud_cover = st.slider("Max Cloud Cover (%)", 0, 50, 10)
    window_days = st.slider("Search Window (±days)", 0, 7, 3)
    run_analysis = st.button("🚀 Analyze Water Quality", type="primary")

# Map — FIXED TILE URL
st.subheader("📍 Draw or Upload Water Body")
# FIXED: Removed spaces in tile URL
m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri"
)
Draw(export=True).add_to(m)
Geocoder().add_to(m)
map_output = st_folium(m, width=700, height=400)

# Main Analysis
if run_analysis and date_list:
    try:
        # Load AOI — Supports drawing!
        if uploaded_file:
            gdf = gpd.read_file(uploaded_file)
        elif map_output and map_output.get("last_active_drawing"):
            geojson = map_output["last_active_drawing"]
            gdf = gpd.GeoDataFrame.from_features([geojson], crs="EPSG:4326")
        else:
            st.error("❌ Please draw on the map or upload a water body boundary.")
            st.stop()

        # Validate & Prep
        area_km2 = gdf.to_crs("EPSG:6933").area.sum() / 1e6
        if area_km2 > MAX_AREA_KM2:
            st.error(f"❌ Area ({area_km2:.1f} km²) exceeds limit of {MAX_AREA_KM2} km²")
            st.stop()

        bounds = gdf.total_bounds.tolist()
        gdf_wkt = gdf.geometry.iloc[0].wkt  # Supports MultiPolygon

        st.info(f"🌊 Processing {len(date_list)} dates over {area_km2:.1f} km² water body")

        # Fetch
        with st.spinner("📡 Finding best satellite images..."):
            items_dict = fetch_items_for_dates(bounds, date_list, cloud_cover, window_days)
            if not items_dict:
                st.error("❌ No usable images found for any date.")
                st.stop()
            st.success(f"✅ Found usable images for {len(items_dict)} out of {len(date_list)} requested dates")

        # Compute
        with st.spinner("🧮 Calculating NDCI values..."):
            df_ndci = compute_ndci_batch(items_dict, bounds, gdf_wkt)
            if df_ndci.empty:
                st.error("❌ Calculation failed.")
                st.stop()

        # Classify
        df_ndci['quality_class'] = df_ndci['ndci_mean'].apply(classify_water_quality)
        st.session_state['ndci_results_df'] = df_ndci

        # Results
        st.success("✅ Analysis Complete!")

        # Plot
        fig = px.line(
            df_ndci,
            x="requested_date",
            y="ndci_mean",
            color="quality_class",
            hover_data=["image_date"],
            title="NDCI Trend (Chlorophyll-a Proxy)",
            markers=True
        )
        fig.update_layout(yaxis_range=[-0.2, 1.0], xaxis_title="Requested Date", yaxis_title="NDCI")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_df = df_ndci[["requested_date", "image_date", "ndci_mean", "quality_class", "cloud_cover"]]
        st.dataframe(display_df.style.format({"ndci_mean": "{:.4f}"}))

        # Download (FULL with geometry)
        csv_full = df_ndci.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Results (CSV)",
            csv_full,
            f"water_quality_ndci_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            help="Includes geometry_wkt for multi-polygon studies"
        )

        # AI Summary
        with st.spinner("🤖 Generating AI insights..."):
            ai_summary = get_gemini_water_summary(df_ndci)
            st.info(f"**AI Water Quality Advisor:** {ai_summary}")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

# Display existing results
elif st.session_state['ndci_results_df'] is not None:
    df = st.session_state['ndci_results_df']
    st.subheader("📊 Water Quality Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest NDCI", f"{df.iloc[-1]['ndci_mean']:.3f}")
    with col2:
        st.metric("Avg NDCI", f"{df['ndci_mean'].mean():.3f}")
    with col3:
        st.metric("Status", df.iloc[-1]['quality_class'])

    fig = px.line(df, x="requested_date", y="ndci_mean", color="quality_class", markers=True, hover_data=["image_date"])
    fig.update_layout(yaxis_range=[-0.2, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    display_df = df[["requested_date", "image_date", "ndci_mean", "quality_class", "cloud_cover"]]
    st.dataframe(display_df.style.format({"ndci_mean": "{:.4f}"}))

    if st.button("🔄 Regenerate AI Summary"):
        with st.spinner("Thinking..."):
            ai_summary = get_gemini_water_summary(df)
            st.info(f"**AI Advisor:** {ai_summary}")

# Footer
st.markdown("---")
st.caption("💧 AquaSat Pro | Powered by Sentinel-2 & Microsoft Planetary Computer")