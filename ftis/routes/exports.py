"""Export helpers for route analysis outputs."""

from __future__ import annotations

import csv
import io
from typing import Any

from ftis.routes.route_models import RouteAnalysis


def route_to_geojson(analysis: RouteAnalysis) -> dict[str, Any]:
    """Return route analysis as GeoJSON features for dashboard overlays."""

    coordinates = [
        [waypoint.longitude, waypoint.latitude, waypoint.altitude_m]
        for waypoint in analysis.waypoints
    ]
    waypoint_features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [waypoint.longitude, waypoint.latitude, waypoint.altitude_m],
            },
            "properties": waypoint.as_dict(),
        }
        for waypoint in analysis.waypoints
    ]
    route_feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "route_risk": analysis.route_risk,
            "cumulative_fti": analysis.cumulative_fti,
            "max_fti": analysis.max_fti,
            "distance_nm": analysis.route_distance_nm,
        },
    }
    return {"type": "FeatureCollection", "features": [route_feature, *waypoint_features]}


def route_to_csv(analysis: RouteAnalysis) -> str:
    """Return route waypoints as CSV text."""

    output = io.StringIO()
    rows = [waypoint.as_dict() for waypoint in analysis.waypoints]
    if not rows:
        return ""
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def route_to_map_overlay(analysis: RouteAnalysis) -> dict[str, Any]:
    """Return a compact overlay structure for Folium, Plotly, or PyDeck."""

    return {
        "path": [
            {
                "lat": waypoint.latitude,
                "lon": waypoint.longitude,
                "fti": waypoint.FTI,
                "risk": waypoint.risk,
                "altitude_m": waypoint.altitude_m,
            }
            for waypoint in analysis.waypoints
        ],
        "corridors": analysis.turbulence_corridors,
    }
