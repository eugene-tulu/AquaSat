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
import logging
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# CONFIG & CONSTANTS
# ========================

# Load environment variables
try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        logger.warning("Google API key not found. AI summaries will be disabled.")
except Exception as e:
    GEMINI_AVAILABLE = False
    logger.error(f"Error loading environment: {e}")

# Water Quality Thresholds for NDCI
WQ_THRESHOLDS = {
    'clear': 0.0,
    'low_algae': 0.1,
    'moderate_bloom': 0.3,
    'high_bloom': 0.5,
    'severe_bloom': 0.7
}

# Color mapping for water quality classes
WQ_COLORS = {
    'Clear Water': '#0066CC',
    'Low Algae': '#66CC00', 
    'Moderate Bloom': '#FFCC00',
    'High Bloom': '#FF6600',
    'Severe Bloom': '#CC0000'
}

MAX_AREA_KM2 = 10000  # For large lakes/reservoirs
MIN_AREA_KM2 = 0.001  # Minimum viable water body size
MAX_DATES = 100  # Maximum number of dates to process

# ========================
# SESSION STATE
# ========================

def initialize_session_state():
    """Initialize session state variables"""
    if 'ndci_results_df' not in st.session_state:
        st.session_state['ndci_results_df'] = None
    if 'analysis_metadata' not in st.session_state:
        st.session_state['analysis_metadata'] = None
    if 'last_error' not in st.session_state:
        st.session_state['last_error'] = None

# ========================
# HELPER FUNCTIONS
# ========================

def validate_geometry(gdf: gpd.GeoDataFrame) -> Tuple[bool, str]:
    """Validate the input geometry"""
    try:
        if gdf.empty:
            return False, "Geometry is empty"
        
        if gdf.geometry.isna().any():
            return False, "Invalid geometry detected"
        
        # Check area
        area_km2 = gdf.to_crs("EPSG:6933").area.sum() / 1e6
        if area_km2 < MIN_AREA_KM2:
            return False, f"Area too small ({area_km2:.4f} km²). Minimum: {MIN_AREA_KM2} km²"
        
        if area_km2 > MAX_AREA_KM2:
            return False, f"Area too large ({area_km2:.1f} km²). Maximum: {MAX_AREA_KM2} km²"
        
        return True, f"Valid geometry: {area_km2:.2f} km²"
    
    except Exception as e:
        return False, f"Geometry validation error: {str(e)}"

def validate_dates(date_list: List[str]) -> Tuple[bool, str, List[str]]:
    """Validate and clean date list"""
    try:
        if not date_list:
            return False, "No dates provided", []
        
        if len(date_list) > MAX_DATES:
            return False, f"Too many dates ({len(date_list)}). Maximum: {MAX_DATES}", []
        
        # Parse and validate dates
        valid_dates = []
        current_date = datetime.now().date()
        
        for date_str in date_list:
            try:
                # Parse date and ensure it's timezone-naive
                parsed_date = pd.to_datetime(date_str, utc=True).tz_localize(None)
                date_obj = parsed_date.date()
                
                # Don't allow future dates beyond tomorrow
                if date_obj > current_date + timedelta(days=1):
                    logger.warning(f"Skipping future date: {date_str}")
                    continue
                valid_dates.append(date_str)
            except Exception as e:
                logger.warning(f"Skipping invalid date {date_str}: {e}")
                continue
        
        if not valid_dates:
            return False, "No valid dates found", []
        
        return True, f"Found {len(valid_dates)} valid dates", sorted(valid_dates)
    
    except Exception as e:
        return False, f"Date validation error: {str(e)}", []

@st.cache_data(show_spinner=False)
def fetch_items_for_dates(bounds: List[float], date_list: List[str], 
                         cloud_limit: int = 10, window_days: int = 3) -> Dict:
    """
    Batch-fetch Sentinel-2 items for dates with improved error handling.
    """
    try:
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            headers={"User-Agent": "AquaSat-Pro/1.0"}
        )

        # Ensure all dates are timezone-naive
        target_dates = pd.to_datetime(date_list, utc=True).tz_localize(None).sort_values()
        min_search = (target_dates.min() - pd.Timedelta(days=window_days + 30)).strftime("%Y-%m-%d")
        max_search = (target_dates.max() + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        date_range = f"{min_search}/{max_search}"

        # Search with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                search = client.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bounds,
                    datetime=date_range,
                    query={"eo:cloud_cover": {"lt": cloud_limit}},
                    sortby=[{"field": "properties.datetime", "direction": "asc"}],
                    limit=1000  # Reasonable limit
                )
                all_items = list(search.get_items())
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                continue

        if not all_items:
            logger.warning("No items found in search")
            return {}

        items_by_date = {}
        # Ensure item datetimes are timezone-naive for comparison
        item_datetimes = [pd.to_datetime(item.datetime, utc=True).tz_localize(None) for item in all_items]

        for target_date_str in date_list:
            # Ensure target date is timezone-naive
            target_date = pd.to_datetime(target_date_str, utc=True).tz_localize(None)

            # 1. Try within window
            candidates = [
                item for item, dt in zip(all_items, item_datetimes)
                if abs((dt - target_date).days) <= window_days
            ]

            if candidates:
                closest = min(candidates, key=lambda item: abs(pd.to_datetime(item.datetime, utc=True).tz_localize(None) - target_date))
                items_by_date[target_date_str] = closest
                continue

            # 2. Fallback: Most recent BEFORE target date
            past_candidates = [
                item for item, dt in zip(all_items, item_datetimes)
                if dt <= target_date
            ]

            if past_candidates:
                most_recent = max(past_candidates, key=lambda item: pd.to_datetime(item.datetime, utc=True).tz_localize(None))
                items_by_date[target_date_str] = most_recent
            else:
                logger.warning(f"No image found before {target_date_str}")

        return items_by_date
    
    except Exception as e:
        logger.error(f"Error fetching items: {e}")
        raise

@st.cache_data(show_spinner=False)
def compute_ndci_batch(_items_dict: Dict, _bounds: List[float], _gdf_wkt: str) -> pd.DataFrame:
    """
    Compute NDCI for all items in batch with enhanced error handling.
    """
    if not _items_dict:
        return pd.DataFrame()

    try:
        import planetary_computer
        
        signed_items = []
        for item in _items_dict.values():
            try:
                signed_item = planetary_computer.sign(item)
                signed_items.append(signed_item)
            except Exception as e:
                logger.warning(f"Failed to sign item {item.id}: {e}")
                continue
        
        if not signed_items:
            raise ValueError("No items could be signed")

        # Create stack with error handling
        try:
            stack = stackstac.stack(
                signed_items,
                assets=["B04", "B05"],
                resolution=10,
                epsg=6933,
                dtype="float32",
                bounds_latlon=_bounds,
                fill_value=np.nan
            )
        except Exception as e:
            logger.error(f"Error creating stack: {e}")
            raise

        # Create AOI from WKT with validation
        try:
            gdf = gpd.GeoDataFrame({'geometry': [gpd.GeoSeries.from_wkt([_gdf_wkt])[0]]}, crs="EPSG:4326")
            gdf_proj = gdf.to_crs(stack.rio.crs)
            aoi_geom = gdf_proj.geometry.unary_union
            
            # Clip to AOI
            clipped = stack.rio.clip([aoi_geom], crs=stack.rio.crs, drop=True)
        except Exception as e:
            logger.error(f"Error clipping to AOI: {e}")
            raise

        # Compute with progress bar
        with ProgressBar():
            data = clipped.compute()

        # Calculate NDCI with validation
        red = data.sel(band="B04")
        red_edge = data.sel(band="B05")
        
        # Avoid division by zero
        denominator = red_edge + red
        denominator = denominator.where(denominator != 0, np.nan)
        ndci = (red_edge - red) / denominator

        results = []
        for i, (target_date, item) in enumerate(_items_dict.items()):
            try:
                ndci_slice = ndci.isel(time=i)
                mean_val = float(np.nanmean(ndci_slice.values))
                
                # Validate NDCI value
                if np.isnan(mean_val) or np.isinf(mean_val):
                    logger.warning(f"Invalid NDCI value for {target_date}")
                    continue

                results.append({
                    "requested_date": target_date,
                    "image_date": item.datetime.strftime("%Y-%m-%d"),
                    "image_time": item.datetime.strftime("%H:%M:%S"),
                    "ndci_mean": mean_val,
                    "cloud_cover": round(item.properties.get("eo:cloud_cover", 0), 1),
                    "satellite": "Sentinel-2",
                    "item_id": item.id,
                    "geometry_wkt": _gdf_wkt
                })
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}")
                continue

        return pd.DataFrame(results)
    
    except Exception as e:
        logger.error(f"Error in NDCI computation: {e}")
        raise

def classify_water_quality(ndci_value: float) -> str:
    """Classify water quality based on NDCI value"""
    if pd.isna(ndci_value):
        return "Unknown"
    
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

def get_trend_description(df: pd.DataFrame) -> str:
    """Analyze trend in NDCI values"""
    if len(df) < 2:
        return "Insufficient data"
    
    recent_values = df['ndci_mean'].tail(3).values
    if len(recent_values) >= 2:
        slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        if slope > 0.02:
            return "Worsening (↗)"
        elif slope < -0.02:
            return "Improving (↘)"
        else:
            return "Stable (→)"
    return "Stable"

def get_gemini_water_summary(ndci_df: pd.DataFrame) -> str:
    """Generate AI summary using Gemini"""
    if not GEMINI_AVAILABLE:
        return "AI summaries unavailable - Google API key not configured"
    
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=(
                "You are an AI water quality expert. Analyze NDCI trends indicating chlorophyll-a levels. "
                "Summarize current risk level, note if conditions are worsening/improving, "
                "and suggest 1–2 specific actions for water managers. "
                "Keep response under 200 words."
            )
        )

        mean_ndci = ndci_df['ndci_mean'].mean()
        recent = ndci_df.iloc[-1]
        trend = get_trend_description(ndci_df)

        prompt = (
            f"Water Body Analysis:\n"
            f"- Recent NDCI: {recent['ndci_mean']:.3f} ({classify_water_quality(recent['ndci_mean'])})\n"
            f"- Average NDCI: {mean_ndci:.3f}\n"
            f"- Trend: {trend}\n"
            f"- Latest image: {recent['image_date']} (requested: {recent['requested_date']})\n"
            f"- Data points: {len(ndci_df)}\n"
            "Provide management recommendations."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=200,
                temperature=0.3,
            )
        )

        return response.text.strip()

    except Exception as e:
        return f"AI summary error: {str(e)}"

def create_enhanced_plot(df: pd.DataFrame) -> px.line:
    """Create enhanced plotly figure with better styling"""
    fig = px.line(
        df,
        x="requested_date",
        y="ndci_mean",
        color="quality_class",
        color_discrete_map=WQ_COLORS,
        hover_data={
            "image_date": True,
            "cloud_cover": True,
            "ndci_mean": ":.4f"
        },
        title="Water Quality Trend (NDCI - Chlorophyll-a Proxy)",
        markers=True
    )
    
    # Add threshold lines
    for threshold, value in WQ_THRESHOLDS.items():
        if threshold != 'clear':
            fig.add_hline(
                y=value, 
                line_dash="dash", 
                line_color="gray",
                opacity=0.5,
                annotation_text=threshold.replace('_', ' ').title()
            )
    
    fig.update_layout(
        yaxis_range=[-0.1, max(1.0, df['ndci_mean'].max() + 0.1)],
        xaxis_title="Requested Date",
        yaxis_title="NDCI Value",
        hovermode='x unified',
        legend_title="Water Quality"
    )
    
    return fig

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(
    layout="wide", 
    page_title="💧 AquaSat Pro",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

initialize_session_state()

st.title("💧 AquaSat Pro - Advanced Water Quality Monitor")
st.markdown("""
Track chlorophyll & algal blooms using Sentinel-2 satellite data. 
Handles missing dates intelligently with fallback mechanisms.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("🛰️ Configuration")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Water Body (GeoJSON)", 
        type=["geojson"],
        help="Upload a GeoJSON file containing water body boundaries"
    )
    
    # Date input options
    date_option = st.radio("Date Input Method", ["Date Range", "Upload CSV"], horizontal=True)
    
    if date_option == "Date Range":
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                datetime.now().date() - timedelta(days=60),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        if start_date > end_date:
            st.error("Start date must be before end date")
            date_list = []
        else:
            freq = st.selectbox("Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly"], index=1)
            freq_map = {"Daily": 'D', "Weekly": 'W', "Bi-weekly": '2W', "Monthly": 'MS'}
            
            try:
                date_range = pd.date_range(start=start_date, end=end_date, freq=freq_map[freq])
                date_list = date_range.strftime("%Y-%m-%d").tolist()
                st.success(f"Generated {len(date_list)} dates")
            except Exception as e:
                st.error(f"Error generating dates: {e}")
                date_list = []
    else:
        # CSV upload
        date_file = st.file_uploader("Upload Dates CSV", type=["csv"])
        if date_file:
            try:
                df_dates = pd.read_csv(date_file)
                if 'date' not in df_dates.columns:
                    st.error("CSV must have 'date' column")
                    date_list = []
                else:
                    df_dates['date'] = pd.to_datetime(df_dates['date'], errors='coerce', utc=True).dt.tz_localize(None)
                    df_dates = df_dates.dropna(subset=['date'])
                    date_list = df_dates['date'].dt.strftime("%Y-%m-%d").unique().tolist()
                    st.success(f"Loaded {len(date_list)} dates")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                date_list = []
        else:
            date_list = []

    # Analysis parameters
    st.subheader("Analysis Parameters")
    cloud_cover = st.slider("Max Cloud Cover (%)", 0, 80, 20, step=5)
    window_days = st.slider("Search Window (±days)", 0, 14, 3, 
                           help="Days to search around target date for images")
    
    # Analysis button
    run_analysis = st.button("🚀 Analyze Water Quality", type="primary", use_container_width=True)
    
    # Clear results button
    if st.session_state['ndci_results_df'] is not None:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state['ndci_results_df'] = None
            st.session_state['analysis_metadata'] = None
            st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Water Body Selection")
    
    # Create map with better tile source
    m = folium.Map(
        location=[0, 20],  # Centered over Africa/Atlantic
        zoom_start=2,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )
    
    # Add drawing tools
    Draw(
        export=True,
        filename='water_body.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        }
    ).add_to(m)
    
    # Add geocoder
    Geocoder(collapsed=False, position='topright').add_to(m)
    
    map_output = st_folium(m, width=700, height=500, returned_objects=["all_drawings"])

with col2:
    # Display current analysis info
    if st.session_state['ndci_results_df'] is not None:
        st.subheader("📊 Current Analysis")
        df = st.session_state['ndci_results_df']
        
        # Key metrics
        latest_ndci = df.iloc[-1]['ndci_mean']
        avg_ndci = df['ndci_mean'].mean()
        latest_class = df.iloc[-1]['quality_class']
        trend = get_trend_description(df)
        
        st.metric(
            label="Latest NDCI", 
            value=f"{latest_ndci:.3f}",
            delta=f"{latest_ndci - avg_ndci:.3f}"
        )
        st.metric(label="Water Quality", value=latest_class)
        st.metric(label="Trend", value=trend)
        st.metric(label="Data Points", value=len(df))
        
        # Metadata
        if st.session_state['analysis_metadata']:
            metadata = st.session_state['analysis_metadata']
            st.info(f"**Area:** {metadata['area_km2']:.2f} km²")

# Main Analysis Logic
if run_analysis:
    # Validate inputs
    if not date_list:
        st.error("❌ No dates selected. Please configure date range or upload CSV.")
        st.stop()
    
    # Validate and clean dates
    dates_valid, dates_msg, clean_dates = validate_dates(date_list)
    if not dates_valid:
        st.error(f"❌ Date validation failed: {dates_msg}")
        st.stop()
    
    st.info(dates_msg)
    date_list = clean_dates

    try:
        # Get geometry from uploaded file or map drawing
        gdf = None
        if uploaded_file:
            try:
                gdf = gpd.read_file(uploaded_file)
                st.success("✅ Loaded geometry from uploaded file")
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")
                st.stop()
        elif map_output and map_output.get("all_drawings"):
            try:
                drawings = map_output["all_drawings"]
                if drawings:
                    # Get the most recent drawing
                    geojson = drawings[-1]
                    gdf = gpd.GeoDataFrame.from_features([geojson], crs="EPSG:4326")
                    st.success("✅ Using drawn geometry from map")
            except Exception as e:
                st.error(f"❌ Error processing map drawing: {e}")
                st.stop()
        
        if gdf is None or gdf.empty:
            st.error("❌ Please draw a water body on the map or upload a GeoJSON file.")
            st.stop()

        # Validate geometry
        geom_valid, geom_msg = validate_geometry(gdf)
        if not geom_valid:
            st.error(f"❌ {geom_msg}")
            st.stop()
        
        st.success(f"✅ {geom_msg}")

        # Prepare analysis parameters
        bounds = gdf.total_bounds.tolist()
        gdf_wkt = gdf.geometry.iloc[0].wkt
        area_km2 = gdf.to_crs("EPSG:6933").area.sum() / 1e6

        # Store metadata
        st.session_state['analysis_metadata'] = {
            'area_km2': area_km2,
            'bounds': bounds,
            'date_count': len(date_list),
            'analysis_time': datetime.now().isoformat()
        }

        st.info(f"🌊 Processing {len(date_list)} dates over {area_km2:.2f} km² water body")

        # Fetch satellite data
        with st.spinner("📡 Searching for satellite images..."):
            progress_bar = st.progress(0)
            progress_bar.progress(0.2)
            
            items_dict = fetch_items_for_dates(bounds, date_list, cloud_cover, window_days)
            progress_bar.progress(0.5)
            
            if not items_dict:
                st.error("❌ No usable satellite images found for any date.")
                st.stop()
            
            found_pct = len(items_dict) / len(date_list) * 100
            st.success(f"✅ Found images for {len(items_dict)}/{len(date_list)} dates ({found_pct:.0f}%)")
            progress_bar.progress(0.7)

        # Compute NDCI
        with st.spinner("🧮 Calculating water quality indices..."):
            df_ndci = compute_ndci_batch(items_dict, bounds, gdf_wkt)
            progress_bar.progress(0.9)
            
            if df_ndci.empty:
                st.error("❌ NDCI calculation failed - no valid results.")
                st.stop()

        # Classify results
        df_ndci['quality_class'] = df_ndci['ndci_mean'].apply(classify_water_quality)
        df_ndci = df_ndci.sort_values('requested_date')
        
        # Store results
        st.session_state['ndci_results_df'] = df_ndci
        progress_bar.progress(1.0)
        progress_bar.empty()

        st.success("✅ Analysis Complete!")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Analysis error: {e}", exc_info=True)
        st.session_state['last_error'] = str(e)
        if st.checkbox("Show detailed error"):
            st.exception(e)
        st.stop()

# Display Results
if st.session_state['ndci_results_df'] is not None:
    df = st.session_state['ndci_results_df']
    
    st.subheader("📈 Water Quality Visualization")
    
    # Enhanced plot
    fig = create_enhanced_plot(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    display_df = df[[
        "requested_date", "image_date", "ndci_mean", 
        "quality_class", "cloud_cover", "image_time"
    ]].copy()
    
    # Format the display
    display_df['ndci_mean'] = display_df['ndci_mean'].round(4)
    display_df.columns = [
        "Requested Date", "Image Date", "NDCI", 
        "Water Quality", "Cloud Cover %", "Image Time"
    ]
    
    st.dataframe(
        display_df, 
        use_container_width=True,
        hide_index=True
    )
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results (CSV)",
            csv_data,
            f"water_quality_ndci_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            help="Download complete results including geometry data"
        )
    
    with col2:
        # Summary statistics
        summary_stats = {
            'metric': ['Count', 'Mean NDCI', 'Std NDCI', 'Min NDCI', 'Max NDCI'],
            'value': [
                len(df),
                f"{df['ndci_mean'].mean():.4f}",
                f"{df['ndci_mean'].std():.4f}",
                f"{df['ndci_mean'].min():.4f}",
                f"{df['ndci_mean'].max():.4f}"
            ]
        }
        summary_csv = pd.DataFrame(summary_stats).to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Summary Stats",
            summary_csv,
            f"water_quality_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

    # AI Summary (if available)
    if GEMINI_AVAILABLE:
        st.subheader("🤖 AI Water Quality Assessment")
        
        if st.button("Generate AI Summary", type="secondary"):
            with st.spinner("🤖 Analyzing water quality trends..."):
                try:
                    ai_summary = get_gemini_water_summary(df)
                    st.info(f"**AI Assessment:** {ai_summary}")
                except Exception as e:
                    st.warning(f"AI summary generation failed: {e}")
    else:
        st.info("💡 **Tip:** Configure Google API key in environment to enable AI summaries")

# Footer with information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**💧 AquaSat Pro**")
    st.caption("Advanced satellite water quality monitoring")

with col2:
    st.markdown("**📊 Data Sources**")
    st.caption("Sentinel-2 L2A | Microsoft Planetary Computer")

with col3:
    st.markdown("**ℹ️ NDCI Scale**")
    st.caption("0.0: Clear | 0.1: Low Algae | 0.3+: Bloom Conditions")

# Debug info (only if there was an error)
if st.session_state.get('last_error') and st.checkbox("🔧 Show Debug Information"):
    with st.expander("Debug Details"):
        st.code(st.session_state['last_error'])
        if st.session_state.get('analysis_metadata'):
            st.json(st.session_state['analysis_metadata'])