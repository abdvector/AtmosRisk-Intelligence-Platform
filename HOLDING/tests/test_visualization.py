"""
test_visualization.py

Tests for the visualization module.
Verifies that all chart builders return valid Plotly Figure objects
with the expected traces, without requiring a running browser.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from src.data import DataProcessor
from src.network import AtmosphericNetwork
from src.risk import RiskEngine
from src.stats import StatisticalValidator
from src.visualization import (
    build_geo_map,
    build_trend_chart,
    build_risk_bar,
    build_network_graph,
)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE
# ──────────────────────────────────────────────────────────────────────────────

def _fixture():
    """Build a minimal pipeline to feed into visualization tests."""
    processor = DataProcessor()
    df_master = processor.generate_telemetry_stream()
    df_window = processor.get_window(df_master, 167)
    pca_df = processor.compute_pca(df_window)

    net = AtmosphericNetwork(threshold=0.2)
    sim_df = net.compute_similarity_matrix(pca_df)
    net.build_graph(sim_df)
    mst = net.maximum_spanning_tree()

    from src.utils.constants import CITY_COORDINATES
    for node in mst.nodes():
        if node in CITY_COORDINATES:
            mst.nodes[node]["lat"] = CITY_COORDINATES[node][0]
            mst.nodes[node]["lon"] = CITY_COORDINATES[node][1]

    engine = RiskEngine(alpha=0.7)
    risk_df, risk_scores = engine.run(mst, pca_df)

    validator = StatisticalValidator()
    stat_results = {
        city: validator.compute_all(city, df_window, df_master, 167)
        for city in mst.nodes()
    }

    return df_master, mst, risk_scores, stat_results


# ──────────────────────────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_geo_map_returns_figure():
    """build_geo_map must return a go.Figure with at least 2 traces."""
    df_master, mst, risk_scores, stat_results = _fixture()
    fig = build_geo_map(mst, risk_scores, stat_results)
    assert isinstance(fig, go.Figure), "Expected a Plotly Figure"
    assert len(fig.data) >= 2, "Expected edge trace + node trace (≥ 2 traces)"
    print("  ✅  test_geo_map_returns_figure: valid Figure with traces")


def test_geo_map_node_count():
    """Geo map node trace must have as many points as MST nodes."""
    df_master, mst, risk_scores, stat_results = _fixture()
    fig = build_geo_map(mst, risk_scores, stat_results)
    # Last trace is the node trace (Scattermapbox with markers)
    node_trace = fig.data[-1]
    assert len(node_trace.lon) == mst.number_of_nodes(), (
        f"Expected {mst.number_of_nodes()} node points, got {len(node_trace.lon)}"
    )
    print(f"  ✅  test_geo_map_node_count: {len(node_trace.lon)} nodes rendered")


def test_trend_chart_returns_figure():
    """build_trend_chart must return a go.Figure."""
    df_master, _, risk_scores, _ = _fixture()
    fig = build_trend_chart(df_master, risk_scores, current_hour=167)
    assert isinstance(fig, go.Figure), "Expected a Plotly Figure"
    assert len(fig.data) >= 1, "Expected at least one line trace"
    print(f"  ✅  test_trend_chart_returns_figure: {len(fig.data)} line traces")


def test_trend_chart_top_k_cities():
    """Trend chart should have exactly top_k line traces (one per city)."""
    df_master, _, risk_scores, _ = _fixture()
    for k in (2, 4):
        fig = build_trend_chart(df_master, risk_scores, current_hour=167, top_k=k)
        assert len(fig.data) == k, (
            f"Expected {k} traces for top_k={k}, got {len(fig.data)}"
        )
    print("  ✅  test_trend_chart_top_k_cities: correct trace count for k=2,4")


def test_risk_bar_returns_figure():
    """build_risk_bar must return a go.Figure with one bar trace."""
    _, _, risk_scores, _ = _fixture()
    fig = build_risk_bar(risk_scores)
    assert isinstance(fig, go.Figure), "Expected a Plotly Figure"
    assert len(fig.data) == 1, "Expected exactly one bar trace"
    bar_trace = fig.data[0]
    assert len(bar_trace.x) == len(risk_scores), (
        f"Expected {len(risk_scores)} bars, got {len(bar_trace.x)}"
    )
    print(f"  ✅  test_risk_bar_returns_figure: {len(bar_trace.x)} bars rendered")


def test_risk_bar_ascending_order():
    """Risk bar chart must render cities in ascending risk order (bottom-to-top)."""
    _, _, risk_scores, _ = _fixture()
    fig = build_risk_bar(risk_scores)
    y_labels = list(fig.data[0].y)
    x_values = list(fig.data[0].x)
    # x_values should be non-decreasing
    assert all(
        x_values[i] <= x_values[i + 1] for i in range(len(x_values) - 1)
    ), "Risk bar chart must be sorted ascending"
    print("  ✅  test_risk_bar_ascending_order: bars sorted ascending ✓")


def test_network_graph_returns_figure():
    """build_network_graph must return a go.Figure with 2 traces (edges + nodes)."""
    _, mst, risk_scores, _ = _fixture()
    fig = build_network_graph(mst, risk_scores)
    assert isinstance(fig, go.Figure), "Expected a Plotly Figure"
    assert len(fig.data) == 2, "Expected edge trace + node trace (2 traces)"
    print("  ✅  test_network_graph_returns_figure: valid Figure with 2 traces")


def test_network_graph_node_count():
    """Network graph node trace should have same count as MST nodes."""
    _, mst, risk_scores, _ = _fixture()
    fig = build_network_graph(mst, risk_scores)
    node_trace = fig.data[1]
    assert len(node_trace.x) == mst.number_of_nodes(), (
        f"Expected {mst.number_of_nodes()} node positions"
    )
    print(f"  ✅  test_network_graph_node_count: {len(node_trace.x)} nodes")


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_geo_map_returns_figure,
    test_geo_map_node_count,
    test_trend_chart_returns_figure,
    test_trend_chart_top_k_cities,
    test_risk_bar_returns_figure,
    test_risk_bar_ascending_order,
    test_network_graph_returns_figure,
    test_network_graph_node_count,
]

if __name__ == "__main__":
    sep = "─" * 60
    print(f"\n{sep}")
    print("  AtmosRisk — Visualization Test Suite")
    print(sep)

    passed, failed = 0, 0
    failures = []

    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            failures.append((name, str(exc)))
            print(f"  ❌  {name}: {exc}")

    print(f"\n{sep}")
    print(f"  Results: {passed} passed / {failed} failed / {len(ALL_TESTS)} total")
    print(sep)

    if failures:
        print("\n  FAILURES:")
        for name, msg in failures:
            print(f"    • {name}: {msg}")
        sys.exit(1)
    else:
        print("\n  All visualization tests passed. ✅\n")
        sys.exit(0)