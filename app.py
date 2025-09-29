from __future__ import annotations
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Geocoder, Draw
import geopandas as gpd
import pandas as pd
import numpy as np
import os, json, warnings, logging
from datetime import datetime, timedelta
from shapely import wkt
from shapely.geometry import shape
from typing import Tuple

# Optional AI
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        GEMINI_AVAILABLE = True
except Exception:
    pass

# Plotting
import plotly.express as px
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# CONFIG
# ========================

WQ_THRESHOLDS = {
    'clear': 0.0,
    'low_algae': 0.1,
    'moderate_bloom': 0.3,
    'high_bloom': 0.5,
    'severe_bloom': 0.7
}

WQ_COLORS = {
    'Clear Water': '#0066CC',
    'Low Algae': '#66CC00', 
    'Moderate Bloom': '#FFCC00',
    'High Bloom': '#FF6600',
    'Severe Bloom': '#CC0000'
}

MAX_AREA_KM2 = 10000
MIN_AREA_KM2 = 0.001
MAX_DATES = 100

# ========================
# HELPER FUNCTIONS
# ========================

def validate_geometry(gdf: gpd.GeoDataFrame) -> Tuple[bool, str]:
    try:
        if gdf.empty or gdf.geometry.isna().any():
            return False, "Invalid or empty geometry"
        area_km2 = gdf.to_crs("EPSG:6933").area.sum() / 1e6
        if area_km2 < MIN_AREA_KM2:
            return False, f"Area too small ({area_km2:.4f} km²)"
        if area_km2 > MAX_AREA_KM2:
            return False, f"Area too large ({area_km2:.1f} km²)"
        return True, f"Valid: {area_km2:.2f} km²"
    except Exception as e:
        return False, f"Geometry error: {e}"

def validate_dates(date_list: list[str]) -> Tuple[bool, str, list[str]]:
    if not date_list:
        return False, "No dates", []
    if len(date_list) > MAX_DATES:
        return False, f"Too many dates (>{MAX_DATES})", []
    valid = []
    now = datetime.now().date()
    for d in date_list:
        try:
            dt = pd.to_datetime(d, utc=True).tz_localize(None).date()
            if dt <= now + timedelta(days=1):
                valid.append(pd.to_datetime(d, utc=True).tz_localize(None).strftime("%Y-%m-%d"))
        except:
            continue
    if not valid:
        return False, "No valid dates", []
    return True, f"{len(valid)} valid dates", sorted(set(valid))

def classify_water_quality(ndci_value: float | None) -> str:
    if pd.isna(ndci_value) or ndci_value is None:
        return "No Data"
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
    if len(df) < 2:
        return "Insufficient data"
    recent = df['ndci_mean'].tail(3).values
    if len(recent) >= 2:
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 0.02: return "Worsening (↗)"
        elif slope < -0.02: return "Improving (↘)"
        else: return "Stable (→)"
    return "Stable"

def create_enhanced_plot(df: pd.DataFrame):
    fig = px.line(
        df, x="requested_date", y="ndci_mean",
        color="quality_class", color_discrete_map=WQ_COLORS,
        markers=True,
        hover_data={"image_date": True, "cloud_cover": True, "ndci_mean": ":.4f"},
        title="Water Quality Trend (NDCI)"
    )
    for name, val in WQ_THRESHOLDS.items():
        if name != 'clear':
            fig.add_hline(y=val, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        yaxis_range=[-0.1, max(1.0, df['ndci_mean'].max() + 0.1)],
        xaxis_title="Requested Date", yaxis_title="NDCI",
        hovermode='x unified', legend_title="Water Quality"
    )
    return fig

def get_gemini_water_summary(ndci_df: pd.DataFrame) -> str:
    if not GEMINI_AVAILABLE:
        return "AI summaries require GOOGLE_API_KEY"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Water Body Analysis:\n"
            f"- Recent NDCI: {ndci_df.iloc[-1]['ndci_mean']:.3f} ({classify_water_quality(ndci_df.iloc[-1]['ndci_mean'])})\n"
            f"- Average NDCI: {ndci_df['ndci_mean'].mean():.3f}\n"
            f"- Trend: {get_trend_description(ndci_df)}\n"
            f"- Data points: {len(ndci_df)}\n"
            "Provide concise management recommendations."
        )
        resp = model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=200, temperature=0.3))
        return resp.text.strip()
    except Exception as e:
        return f"AI error: {e}"

# ========================
# CORE LOGIC
# ========================

@st.cache_data(show_spinner=False)
def _search_s2(bounds: list[float], date_min: str, date_max: str, cloud_limit: int) -> list:
    """Return signed Sentinel-2 items for the AOI once – reused by both modes."""
    import pystac_client, planetary_computer as pc
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        headers={"User-Agent": "AquaSat-Pro/1.0"},
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bounds,
        datetime=f"{date_min}/{date_max}",
        query={"eo:cloud_cover": {"lt": cloud_limit}},
        sortby=[{"field": "properties.datetime", "direction": "asc"}],
        limit=1000,
    )
    items = list(search.items())
    return [pc.sign(i) for i in items]

def compute_ndci_batch(items_dict: dict, bounds: list[float], gdf_wkt: str) -> pd.DataFrame:
    import stackstac, rioxarray, planetary_computer as pc
    from dask.diagnostics import ProgressBar

    signed = [pc.sign(it) for it in items_dict.values()]
    stack = stackstac.stack(
        signed, assets=["B04", "B05"], resolution=10, epsg=6933,
        dtype="float", bounds_latlon=bounds, fill_value=np.nan
    )
    gdf = gpd.GeoDataFrame(geometry=[gpd.GeoSeries.from_wkt([gdf_wkt])[0]], crs=4326)
    aoi = gdf.to_crs(stack.rio.crs).geometry.unary_union
    clipped = stack.rio.clip([aoi], crs=stack.rio.crs, drop=True)

    with ProgressBar():
        data = clipped.compute()

    red = data.sel(band="B04")
    red_edge = data.sel(band="B05")
    denom = red_edge + red
    ndci = (red_edge - red).where(denom != 0) / denom.where(denom != 0)

    records = []
    for i, (req_date, item) in enumerate(items_dict.items()):
        val = float(np.nanmean(ndci.isel(time=i).values))
        if np.isnan(val) or np.isinf(val):
            continue
        records.append(
            {
                "requested_date": req_date,
                "image_date": item.datetime.strftime("%Y-%m-%d"),
                "image_time": item.datetime.strftime("%H:%M:%S"),
                "ndci_mean": val,
                "cloud_cover": round(item.properties.get("eo:cloud_cover", 0), 1),
                "item_id": item.id,
            }
        )
    return pd.DataFrame(records)

def parse_geometry_column(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = []
    col = "geometry" if "geometry" in df.columns else "geojson"
    for val in df[col]:
        if pd.isna(val):
            geoms.append(None)
            continue
        s = str(val).strip()
        try:
            if s.startswith("{"):
                geoms.append(shape(json.loads(s)))
            else:
                geoms.append(wkt.loads(s))
        except Exception:
            try:
                import ast
                geom_dict = ast.literal_eval(s)
                geoms.append(shape(geom_dict))
            except Exception:
                geoms.append(None)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:4326")
    return gdf.dropna(subset=["geometry"])

def batch_process_csv(
    df: pd.DataFrame, buffer_km: float, cloud_limit: int = 20, window_days: int = 10
) -> pd.DataFrame:
    import stackstac, rioxarray, planetary_computer as pc

    if len(df) > 5_000:
        st.warning(
            "CSV has > 5 000 rows – you may hit Planetary-Computer rate limits. "
            "Consider splitting the file."
        )

    gdf = parse_geometry_column(df)

    if buffer_km:
        utm = gdf.estimate_utm_crs()
        if utm is None:
            utm = "EPSG:32633"
        gdf["geometry"] = (
            gdf.to_crs(utm).buffer(buffer_km * 1000).to_crs(4326)
        )

    minx, miny, maxx, maxy = gdf.total_bounds
    bounds = [minx, miny, maxx, maxy]

    all_dates = pd.to_datetime(gdf["date"], errors="coerce").dropna()
    if all_dates.empty:
        st.error("No valid dates found in CSV.")
        st.stop()
    dmin = (all_dates.min() - pd.Timedelta(days=window_days + 30)).strftime("%Y-%m-%d")
    dmax = (all_dates.max() + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

    items = _search_s2(bounds, dmin, dmax, cloud_limit)
    if not items:
        st.error("No Sentinel-2 images found for the supplied dates / cloud limit.")
        st.stop()

    # Build stack once
    stack = stackstac.stack(
        items, assets=["B04", "B05"], resolution=10, epsg=6933,
        dtype="float", bounds_latlon=bounds, fill_value=np.nan
    )
    item_dts = [pd.to_datetime(it.datetime, utc=True).tz_localize(None) for it in items]

    results = []
    prog = st.progress(0)
    for idx, (_, row) in enumerate(gdf.iterrows()):
        target = pd.to_datetime(row["date"], errors="coerce")
        if pd.isna(target):
            results.append({**row.to_dict(), "ndci_mean": None, "quality_class": "No Data"})
            continue
        target = target.tz_localize(None)

        # Closest prior image
        candidates = [(i, dt) for i, dt in enumerate(item_dts) if dt <= target]
        if not candidates:
            results.append({**row.to_dict(), "ndci_mean": None, "quality_class": "No Data"})
            continue
        latest_idx, _ = max(candidates, key=lambda x: x[1])

        try:
            gdf_row = gpd.GeoDataFrame({"geometry": [row.geometry]}, crs=4326)
            geom = gdf_row.to_crs(stack.rio.crs).geometry.iloc[0]
            clipped = stack.isel(time=latest_idx).rio.clip([geom], drop=True)
            red = clipped.sel(band="B04")
            re = clipped.sel(band="B05")
            ndci = (re - red) / (re + red).where((re + red) != 0)
            val = float(np.nanmean(ndci.values))
            if np.isnan(val) or np.isinf(val):
                val = None
        except Exception:
            val = None

        results.append(
            {**row.to_dict(), "ndci_mean": val, "quality_class": classify_water_quality(val)}
        )
        prog.progress((idx + 1) / len(gdf))

    prog.empty()
    out = pd.DataFrame(results)
    if "geometry" in out:
        out["geometry"] = out["geometry"].apply(lambda g: g.wkt if g else None)
    return out

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(layout="wide", page_title="💧 AquaSat Pro", page_icon="💧")
st.title("💧 AquaSat Pro – Water Quality from Space")
mode = st.sidebar.radio("Mode", ["🔍 Explore (Single AOI)", "⚡ Batch Process (CSV)"])

if mode == "🔍 Explore (Single AOI)":
    uploaded_file = st.sidebar.file_uploader("Upload GeoJSON", type=["geojson"])
    date_option = st.sidebar.radio("Date Input", ["Date Range", "Upload CSV"])
    date_list = []
    if date_option == "Date Range":
        start = st.sidebar.date_input("Start", datetime.now().date() - timedelta(days=60))
        end = st.sidebar.date_input("End", datetime.now().date())
        if start <= end:
            freq = st.sidebar.selectbox("Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly"], index=1)
            freq_map = {"Daily": 'D', "Weekly": 'W', "Bi-weekly": '2W', "Monthly": 'MS'}
            date_list = pd.date_range(start, end, freq=freq_map[freq]).strftime("%Y-%m-%d").tolist()
    else:
        csv_file = st.sidebar.file_uploader("Dates CSV", type=["csv"])
        if csv_file:
            df_dates = pd.read_csv(csv_file)
            if 'date' in df_dates.columns:
                date_list = pd.to_datetime(df_dates['date'], errors='coerce').dropna().dt.strftime("%Y-%m-%d").tolist()
    cloud_cover = st.sidebar.slider("Max Cloud Cover (%)", 0, 80, 20)
    window_days = st.sidebar.slider("Search Window (±days)", 0, 14, 3)
    
    # Disable button until both date and geometry exist
    has_geometry = uploaded_file or (st.session_state.get("map_out") and st.session_state["map_out"].get("all_drawings"))
    run = st.sidebar.button(
        "🚀 Analyse", type="primary",
        disabled=not (date_list and has_geometry)
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
        Draw(export=True, draw_options={'polyline':False,'marker':False,'circle':False,'circlemarker':False}).add_to(m)
        Geocoder().add_to(m)
        map_out = st_folium(m, width=700, height=500, returned_objects=["all_drawings"])
        st.session_state["map_out"] = map_out

    gdf = None
    if uploaded_file:
        gdf = gpd.read_file(uploaded_file)
    elif map_out and map_out.get("all_drawings"):
        drawings = map_out["all_drawings"]
        if drawings:
            geojson = drawings[-1]
            gdf = gpd.GeoDataFrame.from_features([geojson], crs="EPSG:4326")
    if run:
        if not date_list:
            st.error("No dates selected")
            st.stop()
        _, _, clean_dates = validate_dates(date_list)
        if gdf is None or gdf.empty:
            st.error("Draw or upload a water body")
            st.stop()
        valid, msg = validate_geometry(gdf)
        if not valid:
            st.error(msg)
            st.stop()
        bounds = gdf.total_bounds.tolist()
        wkt_str = gdf.geometry.iloc[0].wkt
        with st.spinner("Fetching images..."):
            items = {}
            for d in clean_dates:
                items.update(fetch_items_for_dates(bounds, [d], cloud_cover, window_days))
        with st.spinner("Computing NDCI..."):
            df_ndci = compute_ndci_batch(items, bounds, wkt_str)
        if df_ndci.empty:
            st.error("No results")
            st.stop()
        df_ndci['quality_class'] = df_ndci['ndci_mean'].apply(classify_water_quality)
        df_ndci = df_ndci.sort_values('requested_date')
        st.session_state['ndci_results'] = df_ndci
        st.success("✅ Done!")
    if 'ndci_results' in st.session_state:
        df = st.session_state['ndci_results']
        st.plotly_chart(create_enhanced_plot(df), use_container_width=True)
        st.dataframe(df[["requested_date","image_date","ndci_mean","quality_class","cloud_cover"]], use_container_width=True)
        if GEMINI_AVAILABLE:
            if st.button("🤖 AI Summary"):
                summary = get_gemini_water_summary(df)
                st.info(summary)
                st.code(summary, language="text")  # Copyable

else:
    st.subheader("Upload CSV with `date` + `geometry` (GeoJSON/WKT) columns")
    buffer_km = st.sidebar.number_input("Buffer (km)", min_value=0.0, value=0.0, step=0.5)
    cloud_limit = st.sidebar.slider("Max Cloud Cover (%)", 0, 80, 50)
    window_days = st.sidebar.slider("Search Window (±days)", 0, 21, 10)
    csv_file = st.file_uploader("Batch CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        if "date" not in df.columns:
            st.error("CSV must include a 'date' column.")
        else:
            with st.spinner("Processing... this may take a few minutes ⏳"):
                try:
                    results = batch_process_csv(df, buffer_km, cloud_limit, window_days)
                    st.success("✅ Processing complete!")
                    st.dataframe(results, use_container_width=True)
                    csv_out = results.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results", csv_out, "batch_results.csv", "text/csv")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    daily_avg = results.groupby("date")["ndci_mean"].mean().dropna()
                    if not daily_avg.empty:
                        daily_avg.plot(ax=ax, marker="o")
                        ax.set_ylabel("Mean NDCI")
                        ax.set_title("Average NDCI Over Time")
                        st.pyplot(fig)
                except Exception as e:
                    st.exception(e)

st.markdown("---")
st.caption("Data: Sentinel-2 L2A via Microsoft Planetary Computer")