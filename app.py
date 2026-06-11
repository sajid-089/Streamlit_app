from datetime import datetime
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    calculate_kpis,
    clean_sales_dataframe,
    format_currency,
    generate_executive_summary,
    generate_insights,
)
from auth import login, logout, signup
from ml_pipeline import compare_forecasting_models, prepare_daily_series

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Auralytix | BI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# SESSION STATE
# ============================================
DEFAULT_STATE = {
    "user": None,
    "token": None,
    "data": None,
    "clean_df": None,
    "filtered_df": None,
    "forecast_result": None,
    "schema": None,
    "background_theme": "Gradient Dark",
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "kpi_color": "#ffffff",
    "x_axis": "Date",
    "y_axis": "Revenue",
    "chart_type": "Line",
    "horizon": 30,
    "filter_by": "None",
    "filter_value": "All",
    "filter_signature": "None:All",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================
# THEMES
# ============================================
color_themes = {
    "Modern Purple": {"primary": "#667eea", "secondary": "#764ba2"},
    "Corporate Blue": {"primary": "#1e3c72", "secondary": "#2a5298"},
    "Nature Green": {"primary": "#11998e", "secondary": "#38ef7d"},
    "Sunset Orange": {"primary": "#f12711", "secondary": "#f5af19"},
    "Ocean Teal": {"primary": "#00b4db", "secondary": "#0083b0"},
    "Royal Gold": {"primary": "#8e2de2", "secondary": "#4a00e0"},
    "Rose Pink": {"primary": "#ff0844", "secondary": "#ffb199"},
    "Cyber Neon": {"primary": "#00f2fe", "secondary": "#4facfe"},
}

background_themes = {
    "Gradient Dark": "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)",
    "Deep Blue": "linear-gradient(135deg, #000428 0%, #004e92 100%)",
    "Midnight": "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
    "Forest": "linear-gradient(135deg, #134e5e 0%, #71b280 100%)",
    "Sunset": "linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%)",
    "Ocean": "linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%)",
    "Premium Black": "linear-gradient(135deg, #000000 0%, #1a1a2e 100%)",
    "Royal Purple": "linear-gradient(135deg, #4a00e0 0%, #8e2de2 100%)",
    "Carbon": "radial-gradient(circle at 30% 50%, #2d2d2d 0%, #1a1a1a 100%)",
    "Glass Effect": "rgba(0, 0, 0, 0.85)",
}

# ============================================
# HELPERS
# ============================================
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def feature_chips(items: List[str]) -> str:
    """Render small premium feature chips."""
    return "".join([f"<span class='chip'>{item}</span>" for item in items])


def render_metric_card(title: str, value: str, helper: str, accent: str = "#667eea") -> None:
    """Render a premium KPI card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {accent};">{value}</div>
            <div class="metric-helper">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_box(summary: Dict[str, object]) -> None:
    """Render an executive summary block."""
    bullets = "".join([f"<li>{item}</li>" for item in summary.get("recommendations", [])])
    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-kicker">EXECUTIVE SUMMARY</div>
            <h3 class="summary-headline">{summary.get("headline", "")}</h3>
            <p class="summary-text">{summary.get("summary_text", "")}</p>
            <div class="summary-subtitle">Recommended actions</div>
            <ul class="summary-list">{bullets}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_axis_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that are useful for chart x-axis."""
    axis_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype == "object":
            axis_cols.append(col)
    return axis_cols


def normalize_kpis(raw: dict) -> dict:
    """Normalize KPI keys so app can work with different analytics outputs."""
    return {
        "total_revenue": raw.get("total_revenue", 0.0),
        "avg_revenue": raw.get("avg_revenue", 0.0),
        "total_records": raw.get("total_records", 0),
        "profit": raw.get("profit", 0.0),
        "margin": raw.get("margin", raw.get("margin_pct", 0.0)),
        "unique_items": raw.get("unique_items", raw.get("unique_products", 0)),
        "growth": raw.get("growth", raw.get("growth_pct", 0.0)),
        "volatility": raw.get("volatility", raw.get("volatility_pct", 0.0)),
        "top_product": raw.get("top_product"),
        "top_product_value": raw.get("top_product_value", raw.get("top_product_revenue", 0.0)),
        "weak_product": raw.get("weak_product"),
        "weak_product_value": raw.get("weak_product_value", raw.get("weak_product_revenue", 0.0)),
        "top_region": raw.get("top_region"),
        "top_region_value": raw.get("top_region_value", raw.get("top_region_revenue", 0.0)),
        "weak_region": raw.get("weak_region"),
        "weak_region_value": raw.get("weak_region_value", raw.get("weak_region_revenue", 0.0)),
        "unique_regions": raw.get("unique_regions", 0),
    }


def calc_risk_label(kpis: Dict[str, float]) -> str:
    """Return a simple business risk label."""
    growth = kpis.get("growth", 0.0)
    volatility = kpis.get("volatility", 0.0)

    if growth < 0 and volatility > 40:
        return "High"
    if growth < 0 or volatility > 35:
        return "Medium"
    return "Low"


def apply_segment_filter(df: pd.DataFrame, filter_by: str, filter_value: str) -> pd.DataFrame:
    """Apply one business segment filter."""
    if filter_by == "None" or filter_value == "All":
        return df.copy()
    if filter_by not in df.columns:
        return df.copy()
    return df[df[filter_by].astype(str) == str(filter_value)].copy()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel file from uploader."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# ============================================
# DEMO DATA
# ============================================
@st.cache_data(show_spinner=False)
def generate_demo_data() -> pd.DataFrame:
    """Generate a realistic sales demo dataset with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

    products = [
        "MacBook Pro",
        "iPhone 15",
        "AirPods Pro",
        "iPad Pro",
        "Apple Watch",
    ]
    regions = ["North America", "Europe", "Asia Pacific", "Middle East", "LATAM"]

    rows = []
    for date in dates:
        seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * date.dayofyear / 365)
        weekly = 1.0 + (0.14 if date.dayofweek in [4, 5] else 0.0)
        trend = 1.0 + ((date - dates[0]).days / len(dates)) * 0.22

        for product in products:
            factor = {
                "MacBook Pro": 1.55,
                "iPhone 15": 1.25,
                "AirPods Pro": 1.05,
                "iPad Pro": 0.95,
                "Apple Watch": 1.10,
            }[product]

            revenue = 18000 * seasonal * weekly * trend * factor + np.random.normal(0, 220)
            revenue = max(250, revenue)

            # One visible anomaly for storytelling in the demo.
            if date == pd.Timestamp("2024-03-15") and product == "iPad Pro":
                revenue *= 0.35

            revenue = round(float(revenue), 2)

            rows.append(
                {
                    "Date": date,
                    "Product": product,
                    "Region": np.random.choice(regions),
                    "Revenue": revenue,
                    "Profit": round(revenue * 0.34, 2),
                    "Quantity": int(np.random.randint(10, 420)),
                }
            )

    return pd.DataFrame(rows)


# ============================================
# CSS
# ============================================
def apply_custom_css() -> None:
    """Apply premium UI styling."""
    primary = st.session_state.primary_color
    secondary = st.session_state.secondary_color
    bg_theme = background_themes.get(
        st.session_state.background_theme,
        background_themes["Gradient Dark"],
    )
    pr, pg, pb = hex_to_rgb(primary)

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', 'IBM Plex Sans', sans-serif;
            }}

            .stApp {{
                background: {bg_theme};
                color: #f8fafc;
            }}

            .hero {{
                background: rgba(15, 23, 42, 0.75);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 24px;
                padding: 22px;
                margin-bottom: 18px;
                box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
                backdrop-filter: blur(18px);
            }}

            .hero h1 {{
                margin: 0;
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: -0.04em;
            }}

            .brand-name {{
                background: linear-gradient(135deg, #ffc700 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Inter', sans-serif;
                font-weight: 800;
            }}

            .stylish-title {{
                font-family: 'Playfair Display', serif;
                font-size: 1.8rem;
                font-style: italic;
                font-weight: 700;
                background: linear-gradient(135deg, #68fc8c 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-left: 8px;
                opacity: 0.95;
                letter-spacing: 0.5px;
            }}
            .hero p {{
                color: #a0a0c0;
                margin-top: 0.5rem;
                margin-bottom: 0.8rem;
            }}

            .chip {{
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                margin-right: 0.45rem;
                margin-bottom: 0.45rem;
                font-size: 0.72rem;
                color: #e5e7eb;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.08);
            }}

            .metric-card {{
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 18px;
                padding: 16px;
                height: 100%;
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.14);
                transition: 0.25s ease;
            }}

            .metric-card:hover {{
                transform: translateY(-2px);
                border-color: rgba({pr},{pg},{pb},0.35);
            }}

            .metric-title {{
                text-transform: uppercase;
                letter-spacing: 1.7px;
                font-size: 0.72rem;
                color: #cbd5e1;
                margin-bottom: 0.3rem;
            }}

            .metric-value {{
                font-size: 1.8rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: white;
                margin-bottom: 0.25rem;
            }}

            .metric-helper {{
                font-size: 0.82rem;
                color: #94a3b8;
            }}

            .summary-box {{
                background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(168,85,247,0.12));
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 20px;
                padding: 18px;
                margin-bottom: 1rem;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(12px);
            }}

            .summary-kicker {{
                color: #93c5fd;
                text-transform: uppercase;
                letter-spacing: 2px;
                font-size: 0.7rem;
                margin-bottom: 0.35rem;
            }}

            .summary-headline {{
                margin: 0 0 0.4rem 0;
                color: white;
                font-size: 1.25rem;
                font-weight: 800;
                letter-spacing: -0.03em;
            }}

            .summary-text {{
                color: #d1d5db;
                line-height: 1.65;
                margin-bottom: 0.9rem;
            }}

            .summary-subtitle {{
                color: white;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }}

            .summary-list {{
                margin-top: 0.35rem;
                color: #e5e7eb;
                line-height: 1.6;
            }}

            .panel {{
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 20px;
                padding: 1.1rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
                backdrop-filter: blur(10px);
            }}

            .panel-title {{
                color: white;
                font-size: 0.92rem;
                font-weight: 700;
                margin-bottom: 0.9rem;
                padding-left: 0.8rem;
                border-left: 3px solid {primary};
            }}

            .subtle {{
                color: #94a3b8;
                font-size: 0.86rem;
            }}

            .stButton > button {{
                background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
                color: white;
                border: none;
                border-radius: 12px;
                width: 100%;
                font-weight: 700;
                box-shadow: 0 10px 24px rgba({pr},{pg},{pb},0.26);
            }}

            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 14px 30px rgba({pr},{pg},{pb},0.32);
            }}

            .stSelectbox > div > div {{
                background: rgba(26,26,58,0.8);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                color: white;
            }}

            .stTextInput > div > div > input, .stPasswordInput > div > div > input {{
                background: rgba(26,26,58,0.8);
                color: white;
                border-radius: 12px;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                background: rgba(255,255,255,0.05);
                padding: 0.5rem;
                border-radius: 16px;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 12px;
                color: white;
                font-weight: 600;
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
            }}

            div[data-testid="stColorPicker"] label {{
                color: white !important;
            }}

            .status-pill {{
                display: inline-block;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                font-size: 0.72rem;
                margin-right: 0.4rem;
                background: rgba(255,255,255,0.08);
                color: #e5e7eb;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_css()

# ============================================
# CHART BUILDERS
# ============================================
def create_colored_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str, title: str):
    """Create a premium chart using the selected theme colors."""
    primary = st.session_state.primary_color
    secondary = st.session_state.secondary_color

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0)
    plot_df = plot_df.groupby(x_col, as_index=False)[y_col].sum()

    if x_col == "Month":
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=month_order, ordered=True)
        plot_df = plot_df.sort_values(x_col)
    elif x_col == "Quarter":
        q_order = ["Q1", "Q2", "Q3", "Q4"]
        plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=q_order, ordered=True)
        plot_df = plot_df.sort_values(x_col)
    elif x_col == "Day":
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=day_order, ordered=True)
        plot_df = plot_df.sort_values(x_col)
    elif pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        plot_df = plot_df.sort_values(x_col)

    if chart_type == "Line":
        fig = px.line(plot_df, x=x_col, y=y_col, title=title, template="plotly_dark", markers=True)
        fig.update_traces(
            line=dict(color=primary, width=3, shape="spline"),
            marker=dict(size=8, color=secondary),
        )
    elif chart_type == "Bar":
        fig = px.bar(
            plot_df,
            x=x_col,
            y=y_col,
            title=title,
            template="plotly_dark",
            color=y_col,
            color_continuous_scale=[primary, secondary],
        )
    elif chart_type == "Area":
        fig = px.area(plot_df, x=x_col, y=y_col, title=title, template="plotly_dark")
        pr, pg, pb = hex_to_rgb(primary)
        fig.update_traces(
            fill="tozeroy",
            line=dict(color=primary, width=3),
            fillcolor=f"rgba({pr},{pg},{pb},0.28)",
        )
    else:
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            title=title,
            template="plotly_dark",
            size=y_col,
            color=y_col,
            color_continuous_scale=[primary, secondary],
            size_max=16,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(color="white", size=15),
        font=dict(color="white"),
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    return fig


def build_3d_chart(df: pd.DataFrame):
    """Build a premium 3D performance view."""
    perf = (
        df.groupby(["Product", "Region"], as_index=False)[["Revenue", "Profit", "Quantity"]]
        .sum()
        .sort_values("Revenue", ascending=False)
    )

    if perf.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=500)
        return fig

    max_rev = float(perf["Revenue"].max()) if float(perf["Revenue"].max()) > 0 else 1.0
    perf["PointSize"] = np.clip(perf["Revenue"] / max_rev * 22, 8, 22)

    fig = px.scatter_3d(
        perf,
        x="Revenue",
        y="Profit",
        z="Quantity",
        color="Region",
        symbol="Product",
        size="PointSize",
        opacity=0.9,
        title="3D Performance View",
    )
    fig.update_traces(marker=dict(line=dict(width=0.35, color="white")))
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis_title="Revenue",
            yaxis_title="Profit",
            zaxis_title="Quantity",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def build_heatmap(df: pd.DataFrame):
    """Build a weekday vs month heatmap."""
    heat = df.pivot_table(
        index="Day",
        columns="Month",
        values="Revenue",
        aggfunc="sum",
        fill_value=0,
    )

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    heat = heat.reindex(weekdays)
    heat = heat[[m for m in months if m in heat.columns]]

    if heat.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=400, title="Sales Intensity Heatmap")
        return fig

    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Sales Intensity Heatmap",
    )
    fig.update_layout(template="plotly_dark", height=420)
    return fig


def build_forecast_chart(actual_series: pd.Series, forecast_df: pd.DataFrame, forecast_title: str):
    """Build a forecast chart with confidence band."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_series.index,
            y=actual_series.values,
            mode="lines",
            name="Actual",
            line=dict(color=st.session_state.primary_color, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f472b6", width=3, dash="dot"),
        )
    )

    if "upper" in forecast_df.columns and "lower" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(244,114,182,0.18)",
                line=dict(width=0),
                name="Confidence Band",
            )
        )

    fig.update_layout(
        title=forecast_title,
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    return fig


# ============================================
# SIDEBAR AUTH
# ============================================
def render_auth_panel() -> None:
    """Render login/signup controls in the sidebar."""
    st.subheader("Workspace Access")

    if st.session_state.user:
        st.success(f"Signed in as {st.session_state.user['email']}")
        st.caption(f"Role: {st.session_state.user.get('role', 'viewer')}")

        if st.button("Logout", use_container_width=True):
            if st.session_state.token:
                logout(st.session_state.token)
            st.session_state.user = None
            st.session_state.token = None
            st.rerun()
        return

    auth_mode = st.radio("Mode", ["Login", "Sign up"], horizontal=True)
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    role = "viewer"
    if auth_mode == "Sign up":
        role = st.selectbox("Role", ["viewer", "analyst", "admin"], index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Submit", use_container_width=True):
            if auth_mode == "Login":
                result = login(email, password)
            else:
                result = signup(email, password, role)

            if result["success"]:
                st.session_state.user = result["user"]
                st.session_state.token = result.get("token")
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])

    with col_b:
        if st.button("Guest", use_container_width=True):
            st.info("Guest preview mode is active.")


# ============================================
# MAIN APP
# ============================================
def main() -> None:
    """Render the InsightFlow dashboard."""
    workspace = (
        f"{st.session_state.user['email']} • {st.session_state.user.get('role', 'viewer')}"
        if st.session_state.user
        else "Guest preview"
    )

    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;">
                <div>
                    <h1>
                       <span class="brand-name">AURALYTIX</span>
                        <span class="stylish-title">| Business Intelligence & Decision Maker</span>
                    </h1>
                    <p>
                        Premium sales intelligence for executives, analysts, and growth teams.
                        Clean uploads, smart forecasting, 3D analytics, and business-ready summaries.
                    </p>
                    <div>
                        {feature_chips(["Secure Auth", "3D Analytics", "Model Comparison", "Export Ready"])}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="status-pill">● LIVE</div>
                    <div class="status-pill">Workspace: {workspace}</div>
                    <div class="status-pill">Forecast Lab</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Workspace")
        render_auth_panel()

        st.markdown("---")
        st.markdown("### Appearance")

        st.markdown("#### Background Theme")
        bg_choice = st.selectbox("Choose background", list(background_themes.keys()), key="bg_choice")
        if st.button("Apply Background", use_container_width=True):
            st.session_state.background_theme = bg_choice
            st.rerun()

        st.markdown("#### KPI Heading Color")
        kpi_color = st.color_picker("KPI Text Color", st.session_state.kpi_color, key="kpi_color_picker")
        if st.button("Apply KPI Color", use_container_width=True):
            st.session_state.kpi_color = kpi_color
            st.rerun()

        st.markdown("#### Chart Theme")
        theme_choice = st.selectbox("Color Theme", list(color_themes.keys()), key="theme_choice")
        if st.button("Apply Theme", use_container_width=True):
            theme = color_themes[theme_choice]
            st.session_state.primary_color = theme["primary"]
            st.session_state.secondary_color = theme["secondary"]
            st.rerun()

        st.markdown("#### Custom Colors")
        custom_primary = st.color_picker("Primary Color", st.session_state.primary_color, key="custom_primary")
        custom_secondary = st.color_picker("Secondary Color", st.session_state.secondary_color, key="custom_secondary")
        if st.button("Apply Custom Colors", use_container_width=True):
            st.session_state.primary_color = custom_primary
            st.session_state.secondary_color = custom_secondary
            st.rerun()

        st.markdown("---")
        st.markdown("### Data")

        uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
        st.caption("Recommended columns: Date, Product, Region, Revenue, Profit, Quantity")

        if uploaded is not None:
            try:
                st.session_state.data = read_uploaded_file(uploaded)
                st.session_state.filtered_df = None
                st.session_state.forecast_result = None
                st.session_state.filter_by = "None"
                st.session_state.filter_value = "All"
                st.session_state.filter_signature = "None:All"
                st.session_state.x_axis = "Date"
                st.session_state.y_axis = "Revenue"
                st.session_state.chart_type = "Line"
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

        if st.button("Load Demo Dataset", use_container_width=True):
            st.session_state.data = generate_demo_data()
            st.session_state.filtered_df = None
            st.session_state.forecast_result = None
            st.session_state.filter_by = "None"
            st.session_state.filter_value = "All"
            st.session_state.filter_signature = "None:All"
            st.session_state.x_axis = "Date"
            st.session_state.y_axis = "Revenue"
            st.session_state.chart_type = "Line"
            st.success("Demo dataset loaded")
            st.rerun()

        if st.button("Reset Workspace", use_container_width=True):
            st.session_state.data = None
            st.session_state.clean_df = None
            st.session_state.filtered_df = None
            st.session_state.forecast_result = None
            st.session_state.schema = None
            st.session_state.filter_by = "None"
            st.session_state.filter_value = "All"
            st.session_state.filter_signature = "None:All"
            st.session_state.x_axis = "Date"
            st.session_state.y_axis = "Revenue"
            st.session_state.chart_type = "Line"
            st.success("Workspace reset")
            st.rerun()

        st.markdown("---")
        st.markdown("### Chart Settings")

        if st.session_state.clean_df is not None:
            df_for_controls = st.session_state.clean_df
        elif st.session_state.data is not None:
            df_for_controls = st.session_state.data
        else:
            df_for_controls = None

        if df_for_controls is not None and not df_for_controls.empty:
            axis_candidates = detect_axis_columns(df_for_controls)
            numeric_cols = df_for_controls.select_dtypes(include=[np.number]).columns.tolist()

            if not axis_candidates:
                axis_candidates = ["Date"] if "Date" in df_for_controls.columns else list(df_for_controls.columns[:1])

            if numeric_cols:
                y_index = numeric_cols.index("Revenue") if "Revenue" in numeric_cols else 0
                y_axis = st.selectbox("Y-Axis", numeric_cols, index=min(y_index, len(numeric_cols) - 1))
            else:
                y_axis = "Revenue"

            x_default_index = axis_candidates.index("Date") if "Date" in axis_candidates else 0
            x_axis = st.selectbox(
                "X-Axis",
                axis_candidates,
                index=min(x_default_index, len(axis_candidates) - 1),
            )
            chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area", "Scatter"])
            horizon = st.slider("Forecast horizon (days)", 7, 90, st.session_state.horizon, 1)
        else:
            x_axis = st.session_state.x_axis
            y_axis = st.session_state.y_axis
            chart_type = st.session_state.chart_type
            horizon = st.session_state.horizon

        st.session_state.x_axis = x_axis
        st.session_state.y_axis = y_axis
        st.session_state.chart_type = chart_type
        st.session_state.horizon = horizon

        st.markdown("---")
        st.markdown("### Segment Filter")

        if df_for_controls is not None and not df_for_controls.empty:
            available_filters = [c for c in ["Product", "Region", "Quarter", "Day"] if c in df_for_controls.columns]

            if available_filters:
                filter_options = ["None"] + available_filters
                current_filter_by = st.selectbox(
                    "Filter by",
                    filter_options,
                    index=filter_options.index(st.session_state.filter_by)
                    if st.session_state.filter_by in filter_options
                    else 0,
                    key="filter_by_widget",
                )

                if current_filter_by != st.session_state.filter_by:
                    st.session_state.filter_by = current_filter_by
                    st.session_state.filter_value = "All"
                    st.session_state.filter_signature = f"{current_filter_by}:All"
                    st.session_state.filtered_df = None
                    st.rerun()

                if current_filter_by != "None":
                    values = ["All"] + sorted(df_for_controls[current_filter_by].dropna().astype(str).unique().tolist())
                    default_value = (
                        st.session_state.filter_value
                        if st.session_state.filter_value in values
                        else "All"
                    )
                    current_filter_value = st.selectbox(
                        f"Select {current_filter_by}",
                        values,
                        index=values.index(default_value),
                        key=f"filter_value_widget_{current_filter_by}",
                    )

                    if current_filter_value != st.session_state.filter_value:
                        st.session_state.filter_value = current_filter_value
                        st.session_state.filter_signature = f"{current_filter_by}:{current_filter_value}"
                        if current_filter_value == "All":
                            st.session_state.filtered_df = None
                        else:
                            st.session_state.filtered_df = apply_segment_filter(
                                df_for_controls, current_filter_by, current_filter_value
                            )
                else:
                    st.session_state.filtered_df = None
            else:
                st.caption("No segment filter available for this file.")
        else:
            st.session_state.filter_by = "None"
            st.session_state.filter_value = "All"
            st.session_state.filter_signature = "None:All"

    # ============================================
    # DATA PIPELINE
    # ============================================
    if st.session_state.data is None:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">What this demo delivers</div>
                <div class="subtle">
                    Upload a file or load the demo dataset to see premium sales analytics, model comparison,
                    business summaries, 3D views, and forecasting insights.
                </div>
                <div style="margin-top:1rem;">
                    <span class="chip">Secure login</span>
                    <span class="chip">3D charts</span>
                    <span class="chip">Forecasting lab</span>
                    <span class="chip">Business summary</span>
                    <span class="chip">Data quality checks</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Use the sidebar to sign in, load the demo dataset, or upload your own sales file.")
        return

    # Clean the data for business use.
    raw_df = st.session_state.data
    clean_df, schema = clean_sales_dataframe(raw_df)
    st.session_state.clean_df = clean_df
    st.session_state.schema = schema

    if clean_df.empty:
        st.warning("The uploaded data did not produce usable sales rows.")
        return

    # Apply filter if any.
    if st.session_state.filtered_df is not None and not st.session_state.filtered_df.empty:
        view_df = st.session_state.filtered_df.copy()
    else:
        view_df = clean_df.copy()

    if view_df.empty:
        st.warning("No records match the current filter.")
        return

    # ============================================
    # KPI SECTION
    # ============================================
    raw_kpis = calculate_kpis(view_df)
    kpis = normalize_kpis(raw_kpis)
    summary = generate_executive_summary(view_df, st.session_state.forecast_result)
    insights = generate_insights(view_df)
    risk_label = calc_risk_label(kpis)

    st.markdown("### Business Performance Snapshot")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Total Revenue",
            format_currency(kpis["total_revenue"]),
            f"Growth vs previous period: {kpis['growth']:.1f}%",
            accent=st.session_state.primary_color,
        )
    with col2:
        render_metric_card(
            "Profit Margin",
            f"{kpis['margin']:.1f}%",
            f"Estimated profit: {format_currency(kpis['profit'])}",
            accent="#22c55e",
        )
    with col3:
        render_metric_card(
            "Transactions",
            f"{kpis['total_records']:,}",
            f"Unique products: {kpis['unique_items']}",
            accent="#f59e0b",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        render_metric_card(
            "Average Order",
            format_currency(kpis["avg_revenue"]),
            f"Volatility: {kpis['volatility']:.0f}%",
            accent="#f472b6",
        )
    with col5:
        render_metric_card(
            "Top Region",
            kpis["top_region"] or "Unknown",
            format_currency(kpis["top_region_value"]),
            accent="#60a5fa",
        )
    with col6:
        render_metric_card(
            "Top Product",
            kpis["top_product"] or "Unknown",
            format_currency(kpis["top_product_value"]),
            accent="#a855f7",
        )

    col7, col8, col9 = st.columns(3)
    with col7:
        render_metric_card(
            "Business Risk",
            risk_label,
            "Based on growth and volatility",
            accent="#ef4444" if risk_label == "High" else "#f59e0b" if risk_label == "Medium" else "#22c55e",
        )
    with col8:
        render_metric_card(
            "Top Region Share",
            f"{(kpis['top_region_value'] / kpis['total_revenue'] * 100):.1f}%" if kpis["total_revenue"] > 0 else "0.0%",
            "Contribution to total revenue",
            accent="#22c55e",
        )
    with col9:
        render_metric_card(
            "Revenue Stability",
            f"{max(0, 100 - min(100, kpis['volatility'])):.0f}%",
            "Lower volatility is better",
            accent="#38bdf8",
        )

    render_summary_box(summary)

    # ============================================
    # TABS
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Deep Dive", "Forecast Lab", "Data Quality"]
    )

    # --------------------------------------------
    # TAB 1: OVERVIEW
    # --------------------------------------------
    with tab1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Custom Chart View</div>', unsafe_allow_html=True)

        if x_axis and y_axis:
            try:
                custom_chart = create_colored_chart(
                    view_df,
                    x_axis,
                    y_axis,
                    chart_type,
                    f"{y_axis} by {x_axis}",
                )
                st.plotly_chart(custom_chart, use_container_width=True, config={"displayModeBar": True})
            except Exception as exc:
                st.warning(f"Chart could not be rendered: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Revenue Trend</div>', unsafe_allow_html=True)
            daily = view_df.groupby("Date", as_index=False)["Revenue"].sum().sort_values("Date")
            fig = px.line(daily, x="Date", y="Revenue", template="plotly_dark")
            fig.update_traces(line=dict(color=st.session_state.primary_color, width=3, shape="spline"))
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Monthly Performance</div>', unsafe_allow_html=True)
            monthly = view_df.groupby(["MonthNum", "Month"], as_index=False)["Revenue"].sum().sort_values("MonthNum")
            fig = px.bar(
                monthly,
                x="Month",
                y="Revenue",
                template="plotly_dark",
                color="Revenue",
                color_continuous_scale=[st.session_state.primary_color, st.session_state.secondary_color],
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------
    # TAB 2: DEEP DIVE
    # --------------------------------------------
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Top Products</div>', unsafe_allow_html=True)
            top_products = (
                view_df.groupby("Product", as_index=False)["Revenue"]
                .sum()
                .sort_values("Revenue", ascending=False)
                .head(6)
            )
            fig = px.bar(
                top_products,
                x="Revenue",
                y="Product",
                orientation="h",
                template="plotly_dark",
                color="Revenue",
                color_continuous_scale=[st.session_state.primary_color, st.session_state.secondary_color],
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Regional View</div>', unsafe_allow_html=True)
            region = (
                view_df.groupby("Region", as_index=False)["Revenue"]
                .sum()
                .sort_values("Revenue", ascending=False)
            )
            fig = px.pie(
                region,
                values="Revenue",
                names="Region",
                template="plotly_dark",
                color_discrete_sequence=[
                    st.session_state.primary_color,
                    st.session_state.secondary_color,
                    "#f093fb",
                    "#4facfe",
                    "#43e97b",
                ],
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Sales by Weekday</div>', unsafe_allow_html=True)
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekday = (
                view_df.groupby("Day", as_index=False)["Revenue"]
                .sum()
                .set_index("Day")
                .reindex(weekday_order)
                .fillna(0)
                .reset_index()
            )
            fig = px.bar(weekday, x="Day", y="Revenue", template="plotly_dark")
            fig.update_traces(marker_color=st.session_state.primary_color)
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Performance Signals</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom:1rem;">
                    <div class="metric-title">Current Growth</div>
                    <div class="metric-value" style="color:{st.session_state.primary_color};">{kpis['growth']:.1f}%</div>
                    <div class="metric-helper">Trend versus previous period</div>
                </div>
                <div class="metric-card" style="margin-bottom:1rem;">
                    <div class="metric-title">Unique Regions</div>
                    <div class="metric-value" style="color:#22c55e;">{kpis['unique_regions']}</div>
                    <div class="metric-helper">Coverage across markets</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Data Confidence</div>
                    <div class="metric-value" style="color:#f59e0b;">{risk_label}</div>
                    <div class="metric-helper">Simple risk signal for the demo</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">3D Performance View</div>', unsafe_allow_html=True)
        st.caption("Revenue, profit, and quantity together across products and regions.")
        st.plotly_chart(build_3d_chart(view_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Sales Intensity Heatmap</div>', unsafe_allow_html=True)
        st.caption("Strong days and months appear brighter on the grid.")
        st.plotly_chart(build_heatmap(view_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------
    # TAB 3: FORECAST LAB
    # --------------------------------------------
    with tab3:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Forecasting Models</div>
                <div class="subtle">
                    SARIMAX | Exponential Smoothing | Random Forest | Gradient Boosting
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("Compare multiple models and automatically select the best one for the selected horizon.")

        if st.button("Generate Forecast", type="primary"):
            try:
                daily_series = prepare_daily_series(view_df, "Date", "Revenue")
                result = compare_forecasting_models(daily_series, horizon=st.session_state.horizon)
                st.session_state.forecast_result = result
                st.success(result["message"])
            except Exception as exc:
                st.error(f"Forecasting failed: {exc}")

        if st.session_state.forecast_result:
            result = st.session_state.forecast_result
            forecast_df = pd.DataFrame(result.get("forecast", []))
            comparison_df = pd.DataFrame(result.get("comparison", []))

            st.info(f"Best model selected: {result.get('best_model', 'N/A')}")
            st.metric(
                "Projected revenue over selected horizon",
                format_currency(result.get("forecast_sum", 0.0)),
            )

            if not forecast_df.empty:
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                actual_series = prepare_daily_series(view_df, "Date", "Revenue")

                fig = build_forecast_chart(
                    actual_series,
                    forecast_df,
                    "Revenue Forecast with Confidence Band",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    f"Forecast average per day: {format_currency(result.get('forecast_average', 0.0))}"
                )

            if not comparison_df.empty:
                st.subheader("Model Comparison")
                st.dataframe(comparison_df, use_container_width=True, height=250)

            with st.expander("Forecast records", expanded=False):
                st.dataframe(forecast_df, use_container_width=True, height=260)

            with st.expander("Model artifacts", expanded=False):
                st.write(f"Saved artifact: {result.get('artifact_path', 'Not saved')}")
                st.caption("Best model is saved as an artifact for deployment or later serving.")
        else:
            st.warning("Click 'Generate Forecast' to run model comparison.")

    # --------------------------------------------
    # TAB 4: DATA QUALITY
    # --------------------------------------------
    with tab4:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Data Quality Snapshot</div>', unsafe_allow_html=True)

        missing_cells = int(view_df.isna().sum().sum())
        total_cells = int(view_df.shape[0] * view_df.shape[1])
        missing_rate = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

        date_start = view_df["Date"].min()
        date_end = view_df["Date"].max()
        date_span_days = (date_end - date_start).days + 1 if pd.notna(date_start) and pd.notna(date_end) else 0

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            render_metric_card("Rows", f"{len(view_df):,}", "Usable records after cleaning", accent="#60a5fa")
        with q2:
            render_metric_card("Date Span", f"{date_span_days:,}", "Days of coverage", accent="#22c55e")
        with q3:
            render_metric_card("Missing Rate", f"{missing_rate:.1f}%", "Remaining null cells", accent="#f59e0b")
        with q4:
            render_metric_card("Schema Fields", f"{len(schema.get('standard_columns', []))}", "Standard columns mapped", accent="#a855f7")

        st.markdown("### Cleaned Data Preview")
        st.dataframe(view_df, use_container_width=True, height=420)

        st.download_button(
            "Download cleaned CSV",
            view_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_sales_data.csv",
            mime="text/csv",
        )

        with st.expander("Detected schema", expanded=False):
            st.json(schema)

        with st.expander("Business insights", expanded=False):
            for item in insights:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom:0.7rem;">
                        <div class="metric-title">{item['title']}</div>
                        <div class="metric-helper">{item['detail']}</div>
                        <div style="margin-top:0.35rem;"><strong>Action:</strong> {item['action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
