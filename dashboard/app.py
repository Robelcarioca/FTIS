"""Streamlit operations center for the Flight Turbulence Intelligence System."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ftis.config import MODEL_PATH, RISK_COLORS
from ftis.explain_model import feature_importance_ranking
from ftis.model_monitoring import model_metrics_summary
from ftis.routes.airports import AIRPORTS
from ftis.routes.route_models import RouteAnalysis, RouteRequest
from ftis.routes.simulator import route_simulator
from ftis.weather.weather_service import weather_service


st.set_page_config(
    page_title="FTIS | Aviation Operations Center",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
body, .stApp {
    background: #050914;
    color: #e5edf7;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #08111f;
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}
.ftis-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0;
}
.ftis-subtitle {
    color: #94a3b8;
    margin: 0.1rem 0 0.85rem;
    font-size: 0.92rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 0.6rem 0 1rem;
}
.metric-card {
    background: #0b1728;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 8px;
    padding: 0.85rem;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.73rem;
    text-transform: uppercase;
}
.metric-value {
    color: #f8fafc;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0.18rem;
}
.alert-card {
    border-left: 4px solid #38bdf8;
    background: #0b1728;
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.65rem;
}
.risk-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.18rem 0.6rem;
    color: #06111f;
    font-size: 0.8rem;
    font-weight: 700;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc;
}
</style>
"""


AIRPORT_OPTIONS = [
    "LAX",
    "JFK",
    "DEN",
    "ORD",
    "SFO",
    "SEA",
    "ATL",
    "DFW",
    "MIA",
    "LHR",
    "CDG",
    "HND",
    "DXB",
    "SIN",
]


def metric_card(label: str, value: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
    )


def analysis_frame(analysis: RouteAnalysis) -> pd.DataFrame:
    return pd.DataFrame([waypoint.as_dict() for waypoint in analysis.waypoints])


def run_route_analysis(
    departure: str,
    destination: str,
    altitude_m: float,
    speed_kt: float,
    waypoints: int,
    use_live_weather: bool,
) -> RouteAnalysis:
    request = RouteRequest(
        departure_airport=departure,
        destination_airport=destination,
        cruising_altitude_m=altitude_m,
        aircraft_speed_kt=speed_kt,
        waypoint_count=waypoints,
    )
    return route_simulator.analyze_route_sync(request, use_live_weather=use_live_weather)


def make_ops_map(route: pd.DataFrame, playback_index: int) -> folium.Map:
    """Build a dark operational map with route, heat zones, and playback."""

    center = [float(route["latitude"].mean()), float(route["longitude"].mean())]
    flight_map = folium.Map(
        location=center,
        zoom_start=4,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    points = list(zip(route["latitude"], route["longitude"]))
    for index in range(len(points) - 1):
        risk = route.iloc[index + 1]["risk"]
        color = RISK_COLORS.get(risk, "#94a3b8") if index < playback_index else "#475569"
        folium.PolyLine(
            locations=[points[index], points[index + 1]],
            color=color,
            weight=5 if index < playback_index else 3,
            opacity=0.9 if index < playback_index else 0.5,
        ).add_to(flight_map)

    for index, row in route.iterrows():
        risk = row["risk"]
        color = RISK_COLORS.get(risk, "#94a3b8")
        radius = 5 + float(row["FTI"]) / 18
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.25 if index > playback_index else 0.62,
            popup=(
                f"{row['waypoint_id']} | {risk}<br>"
                f"FTI {row['FTI']:.1f}<br>"
                f"Wind {row['wind_speed_kmh']:.1f} km/h<br>"
                f"Provider {row['provider']}"
            ),
        ).add_to(flight_map)

    current = route.iloc[min(playback_index, len(route) - 1)]
    folium.Marker(
        location=[current["latitude"], current["longitude"]],
        popup=f"Playback: {current['waypoint_id']}",
        icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
    ).add_to(flight_map)

    high_risk = route[route["FTI"] >= 66]
    for _, row in high_risk.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=38_000,
            color=RISK_COLORS["High"],
            fill=True,
            fill_opacity=0.09,
            weight=1,
        ).add_to(flight_map)
    return flight_map


def fti_profile(route: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=route["distance_from_origin_nm"],
            y=route["altitude_m"],
            mode="lines",
            name="Altitude",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=route["distance_from_origin_nm"],
            y=route["FTI"],
            mode="lines+markers",
            name="FTI",
            yaxis="y2",
            line=dict(color="#f97316", width=3),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=12, r=12, t=25, b=12),
        xaxis_title="Distance from origin (NM)",
        yaxis=dict(title="Altitude (m)", gridcolor="rgba(148,163,184,0.15)"),
        yaxis2=dict(title="FTI", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", y=1.13),
    )
    return fig


def confidence_gauge(value: float, title: str = "Confidence") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 55], "color": "#172033"},
                    {"range": [55, 80], "color": "#1f3a4d"},
                    {"range": [80, 100], "color": "#14532d"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=8, r=8, t=35, b=8),
    )
    return fig


def weather_chart(route: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=route["waypoint_id"],
            y=route["wind_speed_kmh"],
            name="Wind km/h",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=route["waypoint_id"],
            y=route["jet_stream_indicator"],
            name="Jet indicator",
            line=dict(color="#facc15", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=route["waypoint_id"],
            y=route["turbulence_indicator"],
            name="Turbulence indicator",
            line=dict(color="#ef4444", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310,
        margin=dict(l=12, r=12, t=25, b=12),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        legend=dict(orientation="h", y=1.13),
    )
    return fig


def risk_distribution(route: pd.DataFrame) -> go.Figure:
    counts = route["risk"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)
    fig = px.bar(
        counts.reset_index(),
        x="risk",
        y="count",
        color="risk",
        color_discrete_map=RISK_COLORS,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=12, r=12, t=25, b=12),
        showlegend=False,
    )
    return fig


def operational_alerts(analysis: RouteAnalysis) -> list[str]:
    alerts = []
    if analysis.route_risk == "High":
        alerts.append("High route risk detected. Dispatch review recommended before release.")
    if analysis.turbulence_corridors:
        corridor = analysis.turbulence_corridors[0]
        alerts.append(
            "Elevated corridor "
            f"{corridor['start_waypoint']} to {corridor['end_waypoint']} "
            f"peaks at FTI {corridor['max_fti']:.1f}."
        )
    if analysis.max_fti < 50:
        alerts.append("Route remains below moderate tactical thresholds.")
    return alerts


def remember_prediction(analysis: RouteAnalysis) -> None:
    history = st.session_state.setdefault("prediction_history", [])
    row = {
        "route": f"{analysis.request.departure_airport}-{analysis.request.destination_airport}",
        "risk": analysis.route_risk,
        "cumulative_fti": analysis.cumulative_fti,
        "max_fti": analysis.max_fti,
        "distance_nm": analysis.route_distance_nm,
        "eta_minutes": analysis.estimated_time_minutes,
    }
    if not history or history[-1] != row:
        history.append(row)
    st.session_state["prediction_history"] = history[-25:]


def render_metric_grid(analysis: RouteAnalysis, route: pd.DataFrame) -> None:
    risk_color = RISK_COLORS.get(analysis.route_risk, "#94a3b8")
    st.markdown(
        '<div class="metric-grid">'
        + metric_card("Route Risk", f"<span style='color:{risk_color}'>{analysis.route_risk}</span>")
        + metric_card("Cumulative FTI", f"{analysis.cumulative_fti:.1f}")
        + metric_card("Distance", f"{analysis.route_distance_nm:.0f} NM")
        + metric_card("Mean Wind", f"{route['wind_speed_kmh'].mean():.1f} km/h")
        + "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="ftis-title">FTIS Flight Turbulence Intelligence System</div>'
        '<div class="ftis-subtitle">Aviation operations center for route risk, weather intelligence, model confidence, and system health.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Flight Controls")
        departure = st.selectbox("Departure", AIRPORT_OPTIONS, index=0)
        destination = st.selectbox("Destination", AIRPORT_OPTIONS, index=1)
        altitude_m = st.slider("Cruise altitude (m)", 6000, 14000, 11000, 100)
        speed_kt = st.slider("Aircraft speed (kt)", 250, 560, 455, 5)
        waypoint_count = st.slider("Waypoints", 12, 80, 36)
        use_live_weather = st.toggle("Live weather providers", value=False)
        simulate = st.button("Run route analysis", use_container_width=True)

    try:
        analysis = run_route_analysis(
            departure,
            destination,
            altitude_m,
            speed_kt,
            waypoint_count,
            use_live_weather,
        )
    except Exception as exc:
        st.error(f"Route analysis unavailable: {exc}")
        analysis = run_route_analysis("LAX", "JFK", 11000, 455, 36, False)

    if simulate:
        st.toast("Route analysis refreshed")

    remember_prediction(analysis)
    route = analysis_frame(analysis)
    render_metric_grid(analysis, route)

    tabs = st.tabs(
        [
            "Flight Ops",
            "Model Analytics",
            "Weather Intelligence",
            "Historical Analysis",
            "System Health",
        ]
    )

    with tabs[0]:
        playback_index = st.slider(
            "Route playback",
            0,
            len(route) - 1,
            min(len(route) - 1, len(route) // 2),
        )
        map_col, alert_col = st.columns([2.15, 1])
        with map_col:
            flight_map = make_ops_map(route, playback_index)
            components.html(flight_map._repr_html_(), height=560, scrolling=False)
        with alert_col:
            current = route.iloc[playback_index]
            st.plotly_chart(confidence_gauge(float(current["confidence"])), use_container_width=True)
            st.markdown(
                f"<span class='risk-pill' style='background:{RISK_COLORS[current['risk']]}'>{current['risk']}</span>",
                unsafe_allow_html=True,
            )
            st.metric("Current FTI", f"{current['FTI']:.1f}")
            st.metric("Elapsed", f"{current['elapsed_minutes']:.0f} min")
            st.subheader("Operational Alerts")
            for alert in operational_alerts(analysis):
                st.markdown(f"<div class='alert-card'>{alert}</div>", unsafe_allow_html=True)
        st.plotly_chart(fti_profile(route), use_container_width=True)
        st.dataframe(
            route[
                [
                    "waypoint_id",
                    "distance_from_origin_nm",
                    "altitude_m",
                    "wind_speed_kmh",
                    "pressure_hpa",
                    "temperature_c",
                    "FTI",
                    "risk",
                    "provider",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        metric_col, importance_col = st.columns([1, 1.4])
        with metric_col:
            metrics = model_metrics_summary()
            st.metric("Model Available", "Yes" if metrics.get("model_available") else "No")
            st.metric("Artifact", Path(str(metrics.get("model_path", MODEL_PATH))).name)
            st.json(metrics, expanded=False)
        with importance_col:
            try:
                ranking = pd.DataFrame(feature_importance_ranking(top_n=12))
                fig = px.bar(
                    ranking.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=430,
                    margin=dict(l=12, r=12, t=25, b=12),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Feature importance unavailable: {exc}")
        st.plotly_chart(risk_distribution(route), use_container_width=True)

    with tabs[2]:
        weather_cols = st.columns(4)
        weather_cols[0].metric("Provider Mix", ", ".join(sorted(route["provider"].unique()))[:38])
        weather_cols[1].metric("Min Visibility", f"{route['visibility_m'].fillna(0).min():.0f} m")
        weather_cols[2].metric("Max Jet Indicator", f"{route['jet_stream_indicator'].fillna(0).max():.1f}")
        weather_cols[3].metric("Weather Cache", weather_service.status()["cache"]["entries"])
        st.plotly_chart(weather_chart(route), use_container_width=True)
        st.dataframe(
            route[
                [
                    "waypoint_id",
                    "latitude",
                    "longitude",
                    "wind_direction_deg",
                    "humidity_percent",
                    "visibility_m",
                    "turbulence_indicator",
                    "jet_stream_indicator",
                    "provider",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        history = pd.DataFrame(st.session_state.get("prediction_history", []))
        if history.empty:
            st.info("No prediction history yet.")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
            hist_fig = px.line(
                history.reset_index(),
                x="index",
                y=["cumulative_fti", "max_fti"],
                markers=True,
            )
            hist_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=310,
                margin=dict(l=12, r=12, t=25, b=12),
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    with tabs[4]:
        status = weather_service.status()
        health_cols = st.columns(4)
        health_cols[0].metric("API Model", "Available" if MODEL_PATH.exists() else "Missing")
        health_cols[1].metric("Weather Providers", len(status["providers"]))
        health_cols[2].metric("Cache Entries", status["cache"]["entries"])
        health_cols[3].metric("Route Waypoints", len(route))
        st.json(
            {
                "model_path": str(MODEL_PATH),
                "weather": status,
                "known_airports": len({airport.code for airport in AIRPORTS.values()}),
                "live_weather_enabled": use_live_weather,
            },
            expanded=False,
        )


if __name__ == "__main__":
    main()
