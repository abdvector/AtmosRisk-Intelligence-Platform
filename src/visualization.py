"""
visualization.py

Plotly-based visualization module for the AtmosRisk Intelligence Platform.

Provides:
  - build_geo_map()        → Interactive Scattermapbox geographic risk map
  - build_trend_chart()    → 72h temporal PM2.5 trend for top-k cities
  - build_risk_bar()       → Horizontal risk score ranking bar chart
  - build_network_graph()  → Abstract spring-layout network topology (Plotly)
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.utils.constants import CITY_COORDINATES


def build_geo_map(graph, risk_scores, stat_results):
    """
    Build an interactive Plotly Scattermapbox showing the MST backbone
    over India with nodes coloured by risk score and MST edges overlaid.

    Parameters
    ----------
    graph : nx.Graph
        MST graph. Each node must have 'lon' and 'lat' attributes.
    risk_scores : dict[str, float]
    stat_results : dict[str, dict]
        Keyed by city; each value is a dict with keys:
        ci_low, ci_high, p_val, status

    Returns
    -------
    go.Figure
    """
    # ── Edge traces ────────────────────────────────────────────────────
    edge_lons, edge_lats = [], []

    for u, v in graph.edges():
        u_lon = CITY_COORDINATES[u][1]
        u_lat = CITY_COORDINATES[u][0]
        v_lon = CITY_COORDINATES[v][1]
        v_lat = CITY_COORDINATES[v][0]

        edge_lons.extend([u_lon, v_lon, None])
        edge_lats.extend([u_lat, v_lat, None])

    edge_trace = go.Scattermapbox(
        lon=edge_lons,
        lat=edge_lats,
        mode="lines",
        line=dict(width=2.5, color="#00FFCC"),
        hoverinfo="none",
        name="Transport Links",
    )

    # ── Node traces ────────────────────────────────────────────────────
    node_lons, node_lats, node_colors, hover_texts, node_labels = (
        [], [], [], [], []
    )

    for node in graph.nodes():
        lat, lon = CITY_COORDINATES[node]
        risk = risk_scores.get(node, 0.0)
        stats = stat_results.get(node, {})

        ci_low = stats.get("ci_low", 0.0)
        ci_high = stats.get("ci_high", 0.0)
        p_val = stats.get("p_val", 1.0)
        status = stats.get("status", "OPERATIONAL")

        node_lons.append(lon)
        node_lats.append(lat)
        node_colors.append(risk)
        node_labels.append(node)

        hover_texts.append(
            f"<b>{node}</b><br>"
            f"Risk Index: {risk:.2f}<br>"
            f"95% CI: [{ci_low:.0f}, {ci_high:.0f}]<br>"
            f"Permutation p-val: {p_val:.3f}<br>"
            f"Status: {status}"
        )

    node_trace = go.Scattermapbox(
        lon=node_lons,
        lat=node_lats,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color="white", family="Arial Black"),
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(font_size=13, font_family="Arial"),
        marker=dict(
            showscale=True,
            colorscale="YlOrRd",
            reversescale=False,
            color=node_colors,
            size=16,
            colorbar=dict(
                thickness=18,
                title=dict(text="Risk Score", font=dict(size=13)),
                tickfont=dict(size=11),
            ),
        ),
        name="City Nodes",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text="TOPOLOGICAL BACKBONE",
            font=dict(size=14, color="#A0AEC0", family="Inter, sans-serif"),
        ),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=26.5, lon=80.0),
            zoom=5.5,
        ),
    )

    return fig


def build_trend_chart(df_master, risk_scores, current_hour, top_k=4, window_hours=72):
    """
    72-hour trailing PM2.5 time-series for the top-k riskiest cities.

    Parameters
    ----------
    df_master : pd.DataFrame
    risk_scores : dict[str, float]
    current_hour : int
    top_k : int
    window_hours : int

    Returns
    -------
    go.Figure
    """
    top_cities = sorted(risk_scores, key=risk_scores.get, reverse=True)[:top_k]

    df_trend = df_master[
        (df_master["hour"] > current_hour - window_hours)
        & (df_master["hour"] <= current_hour)
        & (df_master["city"].isin(top_cities))
    ].copy()

    fig = px.line(
        df_trend,
        x="hour",
        y="PM25",
        color="city",
        template="plotly_dark",
    )

    fig.update_layout(
        title=dict(
            text=f"TEMPORAL TRAJECTORY (TOP {top_k})",
            font=dict(size=14, color="#A0AEC0", family="Inter, sans-serif"),
        ),
        xaxis_title=dict(text="Time (Hour)", font=dict(size=12, color="#718096")),
        yaxis_title=dict(text="PM₂.₅ (µg/m³)", font=dict(size=12, color="#718096")),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(size=11, color="#CBD5E0", family="Inter, sans-serif"),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        xaxis=dict(showgrid=True, gridcolor="#1A202C", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1A202C", zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def build_risk_bar(risk_scores):
    """
    Horizontal bar chart ranking all cities by computed risk score.

    Parameters
    ----------
    risk_scores : dict[str, float]

    Returns
    -------
    go.Figure
    """
    df_risk = (
        pd.DataFrame(
            list(risk_scores.items()),
            columns=["City", "Risk_Score"],
        )
        .sort_values("Risk_Score", ascending=True)
        .reset_index(drop=True)
    )

    fig = px.bar(
        df_risk,
        x="Risk_Score",
        y="City",
        orientation="h",
        color="Risk_Score",
        color_continuous_scale="YlOrRd",
        template="plotly_dark",
    )

    fig.update_layout(
        title=dict(text="LATENT RISK RANKING", font=dict(size=14, color="#A0AEC0", family="Inter, sans-serif")),
        xaxis_title=dict(text="Risk Score", font=dict(size=12, color="#718096")),
        yaxis_title=dict(text="", font=dict(size=12)),
        coloraxis_showscale=False,
        font=dict(size=11, color="#CBD5E0", family="Inter, sans-serif"),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        xaxis=dict(showgrid=True, gridcolor="#1A202C", zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def build_network_graph(graph, risk_scores):
    """
    Abstract spring-layout network topology rendered as a Plotly figure.
    Node size and colour are proportional to risk score.

    Parameters
    ----------
    graph : nx.Graph
    risk_scores : dict[str, float]

    Returns
    -------
    go.Figure
    """
    pos = nx.spring_layout(graph, weight="weight", k=1.5, iterations=100, seed=42)

    # ── Edges ──────────────────────────────────────────────────────────
    edge_x, edge_y = [], []

    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#555555"),
        hoverinfo="none",
    )

    # ── Nodes ──────────────────────────────────────────────────────────
    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_colors = [risk_scores.get(n, 0) for n in graph.nodes()]
    node_sizes = [12 + risk_scores.get(n, 1) * 6 for n in graph.nodes()]
    node_labels = list(graph.nodes())

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        hoverinfo="text",
        hovertext=[
            f"{n}<br>Risk: {risk_scores.get(n, 0):.2f}"
            for n in graph.nodes()
        ],
        marker=dict(
            showscale=True,
            colorscale="YlOrRd",
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title=dict(text="Risk", font=dict(size=11)),
                tickfont=dict(size=10),
            ),
            line=dict(width=1, color="#333333"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text="ABSTRACT GRAPH TOPOLOGY", font=dict(size=14, color="#A0AEC0", family="Inter, sans-serif")),
        showlegend=False,
        hovermode="closest",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig  