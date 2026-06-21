from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

st.set_page_config(
    page_title="Energy Demand Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#0E0E10",
    "surface": "#1A1A1F",
    "surface_raised": "#242429",
    "border": "#2E2E34",
    "text_primary": "#E8E5E0",
    "text_secondary": "#9C9A95",
    "accent": "#D4A054",
    "accent_dim": "#A37B3F",
    "positive": "#5EA87A",
    "negative": "#C75D5D",
    "chart_actual": "#5EA87A",
    "chart_prophet": "#8B8BCC",
    "chart_lgbm": "#D4A054",
    "chart_et": "#6BA3BE",
    "chart_forward": "#E8A87C",
}

MODEL_COLORS = {
    "Actual": COLORS["chart_actual"],
    "Prophet": COLORS["chart_prophet"],
    "LightGBM": COLORS["chart_lgbm"],
    "ExtraTrees": COLORS["chart_et"],
}

CUSTOM_CSS = f"""
<style>
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }}
    div[data-testid="stMetric"] label {{
        color: {COLORS["text_secondary"]};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        font-weight: 600;
    }}
    .section-gap {{ margin-top: 2.5rem; margin-bottom: 0.5rem; }}
    h3 {{
        color: {COLORS["text_primary"]} !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.01em;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    h1 {{ font-weight: 600 !important; letter-spacing: -0.02em; }}
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
"""


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """
    Load forecast data from Supabase, deduplicated by date + timestamp.

    Returns:
        DataFrame with all columns from the energy_forecast table, or empty
        DataFrame if unavailable.
    """
    try:
        client = get_supabase_client()
    except ValueError:
        return pd.DataFrame()
    response = (
        client.table(SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    data = getattr(response, "data", None)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["as_of_date", "created_at"], ascending=[True, False])
    df = df.drop_duplicates(subset=["as_of_date", "timestamp"], keep="first")
    return df


def demand_trend_chart(df: pd.DataFrame) -> alt.Chart | None:
    if df.empty:
        return None
    chart_df = df.sort_values("timestamp")
    records = []
    for _, row in chart_df.iterrows():
        d = row["timestamp"]
        records.append({"timestamp": d, "series": "Actual", "demand": row.get("actual_demand")})
        if pd.notna(row.get("prophet_prediction")):
            records.append(
                {"timestamp": d, "series": "Prophet", "demand": row["prophet_prediction"]}
            )
        if pd.notna(row.get("lightgbm_prediction")):
            records.append(
                {"timestamp": d, "series": "LightGBM", "demand": row["lightgbm_prediction"]}
            )
        if pd.notna(row.get("extratrees_prediction")):
            records.append(
                {"timestamp": d, "series": "ExtraTrees", "demand": row["extratrees_prediction"]}
            )
    long_df = pd.DataFrame(records).dropna(subset=["demand"])
    if long_df.empty:
        return None
    series_present = long_df["series"].unique().tolist()
    domain = [s for s in MODEL_COLORS if s in series_present]
    range_colors = [MODEL_COLORS[s] for s in domain]
    chart = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=30, filled=True), strokeWidth=2)
        .encode(
            x=alt.X(
                "timestamp:T",
                title=None,
                axis=alt.Axis(
                    labelColor=COLORS["text_secondary"],
                    gridColor=COLORS["border"],
                    domainColor=COLORS["border"],
                    format="%b %d",
                ),
            ),
            y=alt.Y(
                "demand:Q",
                title="Demand (MW)",
                axis=alt.Axis(
                    labelColor=COLORS["text_secondary"],
                    titleColor=COLORS["text_secondary"],
                    gridColor=COLORS["border"],
                    domainColor=COLORS["border"],
                ),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=domain, range=range_colors),
                legend=alt.Legend(labelColor=COLORS["text_secondary"], orient="top"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("demand:Q", title="Demand (MW)", format=".1f"),
            ],
        )
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .properties(height=320)
    )
    return chart


def model_mae_bar(date_df: pd.DataFrame) -> alt.Chart | None:
    mae_cols = {
        "Prophet": "mae_prophet",
        "LightGBM": "mae_lightgbm",
        "ExtraTrees": "mae_extratrees",
        "Last Value": "mae_last_value",
    }
    records = []
    for model_name, col in mae_cols.items():
        if col in date_df.columns:
            val = date_df[col].dropna()
            if not val.empty:
                records.append({"Model": model_name, "MAE": float(val.iloc[0])})
    if not records:
        return None
    chart_df = pd.DataFrame(records).sort_values("MAE", ascending=True)
    best_model = chart_df.iloc[0]["Model"] if not chart_df.empty else ""
    chart_df["is_best"] = chart_df["Model"] == best_model
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "MAE:Q",
                title="Mean Absolute Error",
                axis=alt.Axis(
                    labelColor=COLORS["text_secondary"],
                    titleColor=COLORS["text_secondary"],
                    gridColor=COLORS["border"],
                    domainColor=COLORS["border"],
                    format=".1f",
                ),
            ),
            y=alt.Y(
                "Model:N",
                title=None,
                sort=alt.EncodingSortField(field="MAE", order="ascending"),
                axis=alt.Axis(labelColor=COLORS["text_primary"], domainColor=COLORS["border"]),
            ),
            color=alt.condition(
                alt.datum.is_best, alt.value(COLORS["positive"]), alt.value(COLORS["accent_dim"])
            ),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("MAE:Q", title="MAE", format=".2f"),
            ],
        )
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .properties(height=max(len(chart_df) * 40, 160))
    )
    return chart


def forward_forecast_chart(fdf: pd.DataFrame) -> alt.Chart | None:
    if fdf.empty:
        return None
    chart = (
        alt.Chart(fdf.sort_values("timestamp"))
        .mark_line(
            point=alt.OverlayMarkDef(size=30, filled=True),
            strokeWidth=2,
            strokeDash=[6, 3],
        )
        .encode(
            x=alt.X(
                "timestamp:T",
                title=None,
                axis=alt.Axis(
                    labelColor=COLORS["text_secondary"],
                    gridColor=COLORS["border"],
                    domainColor=COLORS["border"],
                    format="%b %d %H:%M",
                ),
            ),
            y=alt.Y(
                "forward_prediction:Q",
                title="Predicted Demand (MW)",
                axis=alt.Axis(
                    labelColor=COLORS["text_secondary"],
                    titleColor=COLORS["text_secondary"],
                    gridColor=COLORS["border"],
                    domainColor=COLORS["border"],
                ),
            ),
            color=alt.ColorValue(COLORS["chart_forward"]),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("forward_prediction:Q", title="Forecast (MW)", format=".1f"),
            ],
        )
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .properties(height=280)
    )
    return chart


def run_dashboard() -> None:
    """Render the Streamlit UI for energy demand forecast visualisation."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    df = load_data()
    if df.empty:
        st.title("Energy Demand Forecast")
        st.info("No data available yet. Run the forecast pipeline to populate Supabase.")
        return

    with st.sidebar:
        st.markdown(
            f"<p style='color:{COLORS['accent']};font-weight:600;font-size:1.1rem;"
            f"letter-spacing:-0.01em;margin-bottom:0.25rem;'>Energy Demand Forecast</p>",
            unsafe_allow_html=True,
        )
        st.caption("Short-term electricity demand forecasting")
        st.markdown("---")
        available_dates = sorted(df["as_of_date"].unique(), reverse=True)
        selected_date = st.selectbox(
            "AS-OF DATE", options=available_dates, format_func=lambda d: d.strftime("%Y-%m-%d")
        )
        date_df = df[df["as_of_date"] == selected_date].copy().sort_values("timestamp")
        backtest_df = date_df[date_df.get("forecast_type", "backtest") != "forward"]
        forward_df = date_df[date_df.get("forecast_type") == "forward"]
        st.markdown("---")
        st.caption(
            f"{len(backtest_df)} backtest + {len(forward_df)} forward points. Refreshes every 5 min."
        )

    st.markdown("<h1 style='margin-bottom:0;'>Energy Demand Forecast</h1>", unsafe_allow_html=True)
    st.caption(f"As of {selected_date.strftime('%B %d, %Y')}")

    if not backtest_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            actuals = backtest_df["actual_demand"].dropna()
            st.metric(
                "Avg Actual Demand", f"{actuals.mean():.0f} MW" if not actuals.empty else "N/A"
            )
        with m2:
            st.metric("Data Points", str(len(backtest_df)))
        with m3:
            best = backtest_df["best_model"].dropna()
            st.metric("Best Model", best.iloc[0] if not best.empty else "N/A")
        with m4:
            mae = backtest_df["mae_lightgbm"].dropna()
            st.metric("LightGBM MAE", f"{mae.iloc[0]:.1f}" if not mae.empty else "N/A")

    if not forward_df.empty:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.subheader("Forward Forecast (Next 24 Hours)")
        fwd_meta = forward_df.iloc[0]
        f_model = fwd_meta.get("forecast_model", "N/A")
        fm1, fm2, fm3 = st.columns(3)
        with fm1:
            st.metric("Forecast Model", str(f_model))
        with fm2:
            peak = forward_df["forward_prediction"].max()
            st.metric("Peak Forecast", f"{peak:.0f} MW")
        with fm3:
            avg = forward_df["forward_prediction"].mean()
            st.metric("Avg Forecast", f"{avg:.0f} MW")
        fwd_chart = forward_forecast_chart(forward_df)
        if fwd_chart:
            st.altair_chart(fwd_chart, width="stretch")

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("Demand Forecast Trend (Backtest)")
    trend = demand_trend_chart(backtest_df)
    if trend:
        st.altair_chart(trend, width="stretch")
    else:
        st.info("Trend data requires at least two daily runs.")

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("Model Benchmarking")
    mae_chart = model_mae_bar(backtest_df)
    if mae_chart:
        st.altair_chart(mae_chart, width="stretch")
    else:
        st.info("Model benchmarking requires accumulated data.")

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.subheader("Forecast Data")
    if not date_df.empty:
        cols_to_show = [
            "timestamp",
            "actual_demand",
            "prophet_prediction",
            "lightgbm_prediction",
            "extratrees_prediction",
            "forward_prediction",
            "forecast_type",
        ]
        available = [c for c in cols_to_show if c in date_df.columns]
        display = date_df[available].copy()
        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(display, hide_index=True, width="stretch")


def main() -> None:
    """Entry point for the Streamlit dashboard application."""
    load_dotenv()
    run_dashboard()


if __name__ == "__main__":
    main()
