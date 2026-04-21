import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HVAC Climate Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.title("HVAC Climate Dashboard")
st.caption("Actual historical weather data — powered by Open-Meteo (ERA5 reanalysis)")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def wet_bulb_stull(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """Stull (2011) wet-bulb approximation. T in °C, RH in 0–100. Accurate to ±0.3°C."""
    return (
        T * np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH)
        - 4.686035
    )


def c_to_f(arr):
    return arr * 9 / 5 + 32


@st.cache_data(ttl=600)
def geocode(city: str):
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("results", [])


@st.cache_data(ttl=3600)
def fetch_weather(lat: float, lon: float, start: str, end: str):
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "shortwave_radiation",
                "direct_radiation",
                "wind_speed_10m",
                "precipitation",
            ]),
            "timezone": "auto",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Wet bulb from dry bulb + RH
    df["wet_bulb"] = wet_bulb_stull(
        df["temperature_2m"].values, df["relative_humidity_2m"].values
    )
    # Sun hour: direct radiation > 120 W/m² (WMO threshold)
    df["is_sun_hour"] = df["direct_radiation"] > 120

    return df, data.get("timezone", "UTC"), float(data.get("elevation", 0))


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Location")
    city_input = st.text_input("Search city", value="Reno", placeholder="e.g. London")

    lat, lon, location_label = None, None, ""

    if city_input:
        try:
            results = geocode(city_input)
        except Exception as e:
            st.error(f"Geocoding failed: {e}")
            results = []

        if results:
            options = {
                f"{r['name']}, {r.get('admin1', '')}, {r['country']}": (
                    r["latitude"],
                    r["longitude"],
                )
                for r in results
            }
            # Default to Reno, Nevada, United States when the app first loads
            default_key = "Reno, Nevada, United States"
            default_index = (
                list(options.keys()).index(default_key)
                if default_key in options
                else 0
            )
            chosen = st.selectbox("Select location", list(options.keys()), index=default_index)
            lat, lon = options[chosen]
            location_label = chosen
        else:
            st.warning("No results found. Try a different spelling.")

    st.divider()
    st.header("Date Range")
    current_year = datetime.now().year
    # Default to previous full year; current year may be incomplete
    year_options = list(range(current_year - 1, current_year - 11, -1))
    year = st.selectbox("Year", year_options, index=0)
    start_date = f"{year}-01-01"
    # Cap end date to yesterday so we don't request future data
    end_date = min(date(year, 12, 31), date.today() - pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    st.divider()
    st.header("Settings")
    bin_width = st.select_slider(
        "Bin width",
        options=[1, 2, 5],
        value=2,
        help="Width of temperature bins for frequency histograms",
    )
    hdd_cdd_base_f = st.number_input(
        "Degree-day base (°F)",
        value=65.0,
        step=1.0,
        help="ASHRAE standard base: 65°F (18.3°C)",
    )
    hdd_cdd_base_c = (hdd_cdd_base_f - 32) * 5 / 9

    st.divider()
    st.caption(
        "Data: [Open-Meteo](https://open-meteo.com/) · "
        "ERA5 reanalysis · Free & open"
    )

# ──────────────────────────────────────────────────────────────────────────────
# GATE — need a location before we can do anything
# ──────────────────────────────────────────────────────────────────────────────
if lat is None:
    st.info("Enter a location in the sidebar to get started.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# FETCH DATA
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching weather data for {location_label} ({year})…"):
    try:
        df, tz, elevation = fetch_weather(lat, lon, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to fetch weather data: {e}")
        st.stop()

df = df.dropna(subset=["temperature_2m", "relative_humidity_2m"])

# Display columns — always °F
df["oat_disp"] = c_to_f(df["temperature_2m"])
df["wb_disp"] = c_to_f(df["wet_bulb"])
unit_label = "°F"

# Daily means in °C for degree-day calcs (always metric internally)
daily_oat_c = df["temperature_2m"].resample("D").mean()
hdd_total = (hdd_cdd_base_c - daily_oat_c).clip(lower=0).sum()
cdd_total = (daily_oat_c - hdd_cdd_base_c).clip(lower=0).sum()

sun_hours_monthly = df["is_sun_hour"].resample("ME").sum()
sun_hours_total = int(df["is_sun_hour"].sum())

# ──────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(
    f"{location_label}  ·  {year}  ·  Elev {elevation * 3.28084:.0f} ft  ·  TZ: {tz}"
)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Mean OAT", f"{df['oat_disp'].mean():.1f} {unit_label}")
c2.metric("Peak OAT", f"{df['oat_disp'].max():.1f} {unit_label}")
c3.metric("Min OAT", f"{df['oat_disp'].min():.1f} {unit_label}")
c4.metric("Peak Wet Bulb", f"{df['wb_disp'].max():.1f} {unit_label}")
c5.metric("Mean Wet Bulb", f"{df['wb_disp'].mean():.1f} {unit_label}")
c6.metric("HDD", f"{hdd_total:.0f}", help="Heating degree days")
c7.metric("CDD", f"{cdd_total:.0f}", help="Cooling degree days")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# ROW 1 — Temperature Bins  |  Wet Bulb Bins
# ──────────────────────────────────────────────────────────────────────────────
col_oat, col_wb = st.columns(2)

CHART_HEIGHT = 360
BG = "rgba(0,0,0,0)"

def make_bin_chart(series: pd.Series, bw: float, colorscale: str, xtitle: str):
    lo = np.floor(series.min() / bw) * bw
    hi = np.ceil(series.max() / bw) * bw
    bins = np.arange(lo, hi + bw, bw)
    counts, edges = np.histogram(series.dropna(), bins=bins)
    labels = [f"{e:.0f}" for e in edges[:-1]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker=dict(color=counts, colorscale=colorscale, showscale=False),
            hovertemplate=f"%{{x}} {unit_label}: %{{y}} hrs<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=xtitle,
        yaxis_title="Hours / year",
        height=CHART_HEIGHT,
        margin=dict(t=10, b=40),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )
    return fig


with col_oat:
    st.subheader("Dry Bulb — Temperature Bins")
    st.plotly_chart(
        make_bin_chart(
            df["oat_disp"],
            bin_width,
            "RdBu_r",
            f"Dry Bulb Temperature ({unit_label})",
        ),
        width="stretch",
    )

with col_wb:
    st.subheader("Wet Bulb — Temperature Bins")
    st.plotly_chart(
        make_bin_chart(
            df["wb_disp"],
            bin_width,
            "Blues",
            f"Wet Bulb Temperature ({unit_label})",
        ),
        width="stretch",
    )

# ──────────────────────────────────────────────────────────────────────────────
# ROW 2 — Sun Hours  |  Monthly OAT/WB Profile
# ──────────────────────────────────────────────────────────────────────────────
col_sun, col_monthly = st.columns(2)

with col_sun:
    st.subheader(f"Monthly Sun Hours  (total: {sun_hours_total:,} hrs)")
    sun_df = sun_hours_monthly.reset_index()
    sun_df.columns = ["month", "sun_hours"]
    sun_df["label"] = sun_df["month"].dt.strftime("%b")

    fig_sun = go.Figure(
        go.Bar(
            x=sun_df["label"],
            y=sun_df["sun_hours"],
            marker=dict(
                color=sun_df["sun_hours"],
                colorscale="YlOrRd",
                showscale=False,
            ),
            hovertemplate="%{x}: %{y:.0f} hrs<extra></extra>",
        )
    )
    fig_sun.update_layout(
        xaxis_title="Month",
        yaxis_title="Sun Hours",
        height=CHART_HEIGHT,
        margin=dict(t=10, b=40),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )
    st.plotly_chart(fig_sun, width="stretch")

with col_monthly:
    st.subheader("Monthly Temperature Profile")
    monthly_mean = df["oat_disp"].resample("ME").mean()
    monthly_max = df["oat_disp"].resample("ME").max()
    monthly_min = df["oat_disp"].resample("ME").min()
    monthly_wb = df["wb_disp"].resample("ME").mean()
    months = monthly_mean.index.strftime("%b")

    fig_m = go.Figure()
    fig_m.add_trace(
        go.Scatter(
            x=months, y=monthly_max.values,
            mode="lines", name="OAT Max",
            line=dict(color="salmon", dash="dot", width=1),
            opacity=0.7,
        )
    )
    fig_m.add_trace(
        go.Scatter(
            x=months, y=monthly_mean.values,
            mode="lines+markers", name="OAT Mean",
            line=dict(color="tomato", width=2),
        )
    )
    fig_m.add_trace(
        go.Scatter(
            x=months, y=monthly_min.values,
            mode="lines", name="OAT Min",
            line=dict(color="lightblue", dash="dot", width=1),
            opacity=0.7,
        )
    )
    fig_m.add_trace(
        go.Scatter(
            x=months, y=monthly_wb.values,
            mode="lines+markers", name="WB Mean",
            line=dict(color="steelblue", width=2),
        )
    )
    fig_m.update_layout(
        yaxis_title=f"Temperature ({unit_label})",
        height=CHART_HEIGHT,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )
    st.plotly_chart(fig_m, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# ROW 3 — Degree Days  |  Coincident Dry Bulb vs RH
# ──────────────────────────────────────────────────────────────────────────────
col_dd, col_coin = st.columns(2)

with col_dd:
    st.subheader(f"Monthly Degree Days  (base {hdd_cdd_base_f:.0f}°F)")
    daily_df = pd.DataFrame(
        {
            "HDD": (hdd_cdd_base_c - daily_oat_c).clip(lower=0),
            "CDD": (daily_oat_c - hdd_cdd_base_c).clip(lower=0),
        }
    )
    monthly_dd = daily_df.resample("ME").sum()
    months_dd = monthly_dd.index.strftime("%b")

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Bar(x=months_dd, y=monthly_dd["HDD"], name="HDD", marker_color="steelblue")
    )
    fig_dd.add_trace(
        go.Bar(x=months_dd, y=monthly_dd["CDD"], name="CDD", marker_color="tomato")
    )
    fig_dd.update_layout(
        barmode="group",
        yaxis_title="°F·days",
        height=CHART_HEIGHT,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )
    st.plotly_chart(fig_dd, width="stretch")

with col_coin:
    st.subheader("Coincident Dry Bulb vs Humidity")
    fig_coin = go.Figure(
        go.Histogram2d(
            x=df["oat_disp"],
            y=df["relative_humidity_2m"],
            colorscale="Viridis",
            nbinsx=30,
            nbinsy=20,
            colorbar=dict(title="Hours"),
            hovertemplate=(
                f"OAT: %{{x:.0f}} {unit_label}<br>"
                "RH: %{y:.0f}%<br>"
                "Hours: %{z}<extra></extra>"
            ),
        )
    )
    fig_coin.update_layout(
        xaxis_title=f"Dry Bulb ({unit_label})",
        yaxis_title="Relative Humidity (%)",
        height=CHART_HEIGHT,
        margin=dict(t=10, b=40),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
    )
    st.plotly_chart(fig_coin, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# ROW 4 — Daily OAT time series (full year)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Daily Dry Bulb & Wet Bulb — Full Year")

daily_max = df["oat_disp"].resample("D").max()
daily_mean = df["oat_disp"].resample("D").mean()
daily_min = df["oat_disp"].resample("D").min()
daily_wb_mean = df["wb_disp"].resample("D").mean()

fig_ts = go.Figure()
fig_ts.add_trace(
    go.Scatter(
        x=daily_max.index, y=daily_max.values,
        fill=None, mode="lines",
        line=dict(color="salmon", width=0),
        name="OAT Max", showlegend=True,
    )
)
fig_ts.add_trace(
    go.Scatter(
        x=daily_min.index, y=daily_min.values,
        fill="tonexty", mode="lines",
        line=dict(color="lightblue", width=0),
        fillcolor="rgba(255,160,122,0.15)",
        name="OAT Min (band)", showlegend=True,
    )
)
fig_ts.add_trace(
    go.Scatter(
        x=daily_mean.index, y=daily_mean.values,
        mode="lines", name="OAT Mean",
        line=dict(color="tomato", width=1.5),
    )
)
fig_ts.add_trace(
    go.Scatter(
        x=daily_wb_mean.index, y=daily_wb_mean.values,
        mode="lines", name="WB Mean",
        line=dict(color="steelblue", width=1.5),
    )
)
fig_ts.update_layout(
    yaxis_title=f"Temperature ({unit_label})",
    xaxis_title="Date",
    height=320,
    margin=dict(t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    hovermode="x unified",
)
st.plotly_chart(fig_ts, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# ROW 5 — Daily Relative Humidity — Full Year
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Daily Relative Humidity — Full Year")

daily_rh_mean = df["relative_humidity_2m"].resample("D").mean()
daily_rh_max  = df["relative_humidity_2m"].resample("D").max()
daily_rh_min  = df["relative_humidity_2m"].resample("D").min()

fig_rh = go.Figure()
fig_rh.add_trace(
    go.Scatter(
        x=daily_rh_max.index, y=daily_rh_max.values,
        fill=None, mode="lines",
        line=dict(color="steelblue", width=0),
        name="RH Max", showlegend=True,
    )
)
fig_rh.add_trace(
    go.Scatter(
        x=daily_rh_min.index, y=daily_rh_min.values,
        fill="tonexty", mode="lines",
        line=dict(color="lightblue", width=0),
        fillcolor="rgba(70,130,180,0.15)",
        name="RH Min (band)", showlegend=True,
    )
)
fig_rh.add_trace(
    go.Scatter(
        x=daily_rh_mean.index, y=daily_rh_mean.values,
        mode="lines", name="RH Mean",
        line=dict(color="steelblue", width=1.5),
    )
)
fig_rh.update_layout(
    yaxis=dict(title="Relative Humidity (%)", range=[0, 100]),
    xaxis_title="Date",
    height=300,
    margin=dict(t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    hovermode="x unified",
)
st.plotly_chart(fig_rh, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# PSYCHROMETRIC CHART
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Psychrometric Chart — Hourly Data Distribution")


def sat_pressure_kpa(T_c: np.ndarray) -> np.ndarray:
    """Buck equation — saturation vapour pressure in kPa. T in °C."""
    return 0.61121 * np.exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c))


def hum_ratio_gkg(T_c: np.ndarray, RH: np.ndarray, P_kpa: float = 101.325) -> np.ndarray:
    """Humidity ratio in g/kg dry air. T in °C, RH in 0–100."""
    Pv = (RH / 100.0) * sat_pressure_kpa(T_c)
    return 0.622 * Pv / (P_kpa - Pv) * 1000


# Site pressure corrected for elevation
P_site = 101.325 * np.exp(-elevation / 8430.0)

# Calculate humidity ratio for every hourly data point
df["W_gkg"] = hum_ratio_gkg(
    df["temperature_2m"].values,
    df["relative_humidity_2m"].values,
    P_kpa=P_site,
)

# Temperature grid for background lines (always °C internally)
T_lo = min(float(df["temperature_2m"].min()) - 3.0, -5.0)
T_hi = float(df["temperature_2m"].max()) + 3.0
T_grid = np.linspace(T_lo, T_hi, 400)
T_grid_disp = c_to_f(T_grid)
x_data_psych = df["oat_disp"]

W_max_axis = max(25.0, float(df["W_gkg"].max()) * 1.1)

fig_psych = go.Figure()

# ── Data density ──────────────────────────────────────────────────────────────
fig_psych.add_trace(
    go.Histogram2dContour(
        x=x_data_psych,
        y=df["W_gkg"],
        colorscale="YlOrRd",
        reversescale=False,
        showscale=True,
        colorbar=dict(title="Hours", thickness=12, len=0.75),
        ncontours=15,
        contours=dict(showlabels=False, coloring="fill"),
        line=dict(width=0.3, color="rgba(0,0,0,0.1)"),
        hovertemplate=(
            f"DB: %{{x:.1f}} {unit_label}<br>"
            "W: %{y:.2f} g/kg<extra></extra>"
        ),
        name="Hour density",
    )
)

# ── Constant RH lines ─────────────────────────────────────────────────────────
for rh in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    W_rh = hum_ratio_gkg(T_grid, np.full_like(T_grid, float(rh)), P_kpa=P_site)
    mask = (W_rh >= 0) & (W_rh <= W_max_axis)
    if not mask.any():
        continue
    is_sat = rh == 100
    fig_psych.add_trace(
        go.Scatter(
            x=T_grid_disp[mask],
            y=W_rh[mask],
            mode="lines",
            line=dict(
                color="rgba(30,100,200,0.9)" if is_sat else "rgba(80,140,220,0.45)",
                width=1.6 if is_sat else 0.9,
            ),
            name="Saturation (100%)" if is_sat else f"{rh}% RH",
            showlegend=is_sat,
            hovertemplate=f"{rh}% RH — DB: %{{x:.1f}} {unit_label}, W: %{{y:.2f}} g/kg<extra></extra>",
        )
    )
    # Inline % label near the right-hand end of each curve
    valid_idx = np.where(mask)[0]
    label_idx = valid_idx[int(len(valid_idx) * 0.87)]
    fig_psych.add_annotation(
        x=T_grid_disp[label_idx],
        y=W_rh[label_idx],
        text=f"{rh}%",
        showarrow=False,
        font=dict(size=8, color="royalblue"),
        xanchor="left",
        yanchor="middle",
    )

# ── Constant enthalpy lines ───────────────────────────────────────────────────
# h = 1.006·T + (W/1000)·(2501 + 1.86·T)  →  W (g/kg) = 1000·(h − 1.006·T)/(2501 + 1.86·T)
for h_kj in range(-20, 120, 10):
    W_h = 1000.0 * (h_kj - 1.006 * T_grid) / (2501.0 + 1.86 * T_grid)
    mask = (W_h >= 0) & (W_h <= W_max_axis)
    if not mask.any():
        continue
    fig_psych.add_trace(
        go.Scatter(
            x=T_grid_disp[mask],
            y=W_h[mask],
            mode="lines",
            line=dict(color="rgba(150,0,200,0.18)", width=0.7, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Label at left edge of line
    left_idx = np.where(mask)[0][0]
    fig_psych.add_annotation(
        x=T_grid_disp[left_idx],
        y=W_h[left_idx],
        text=str(h_kj),
        showarrow=False,
        font=dict(size=7, color="purple"),
        xanchor="right",
        yanchor="middle",
    )

# ── ASHRAE 55 comfort zone (approximate summer bounds) ────────────────────────
_cb_c = [20.0, 26.0, 26.0, 20.0, 20.0]
_cb_w = [ 4.0,  4.0, 12.0, 12.0,  4.0]
_cb_disp = [c_to_f(t) for t in _cb_c]
fig_psych.add_trace(
    go.Scatter(
        x=_cb_disp,
        y=_cb_w,
        mode="lines",
        fill="toself",
        fillcolor="rgba(0,180,0,0.07)",
        line=dict(color="green", width=1.2, dash="dash"),
        name="ASHRAE 55 comfort (approx.)",
    )
)

fig_psych.update_layout(
    xaxis_title=f"Dry Bulb Temperature ({unit_label})",
    yaxis=dict(
        title="Humidity Ratio (g/kg dry air)",
        range=[0, W_max_axis],
    ),
    height=540,
    margin=dict(t=20, b=50, r=90),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    hovermode="closest",
)
st.caption(
    f"Purple dotted lines = constant enthalpy (kJ/kg dry air) · "
    f"Site pressure: {P_site:.1f} kPa (elev. {elevation * 3.28084:.0f} ft) · "
    "Green dashed box = ASHRAE 55 summer comfort zone (approximate)"
)
st.plotly_chart(fig_psych, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# RAW DATA EXPANDER
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Raw hourly data"):
    disp = df[[
        "temperature_2m", "wet_bulb", "relative_humidity_2m",
        "dew_point_2m", "shortwave_radiation", "direct_radiation",
        "wind_speed_10m", "precipitation", "is_sun_hour",
    ]].copy()
    disp.columns = [
        "OAT (°C — source)", "Wet Bulb (°C — source)", "RH (%)", "Dew Point (°C — source)",
        "SW Radiation (W/m²)", "Direct Radiation (W/m²)",
        "Wind Speed (m/s)", "Precipitation (mm)", "Sun Hour",
    ]
    st.dataframe(disp, width="stretch")
    csv = disp.to_csv()
    st.download_button("Download CSV", csv, file_name=f"weather_{location_label}_{year}.csv")
