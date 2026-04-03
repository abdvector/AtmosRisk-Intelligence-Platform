# -*- coding: utf-8 -*-
"""
test.py

Unit tests for the AtmosRisk Intelligence Platform core pipeline.

Tests:
  - DataProcessor: stream shape, PCA output, window extraction
  - AtmosphericNetwork: similarity matrix, graph construction, MST
  - RiskEngine: risk formula correctness, DataFrame output
  - MisraGries: heavy hitter detection
  - StatisticalValidator: bootstrap CI, permutation test
"""

import sys
import io

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import networkx as nx

from src.data import DataProcessor
from src.network import AtmosphericNetwork
from src.risk import RiskEngine, MisraGries
from src.stats import StatisticalValidator
from src.utils.constants import CITY_COORDINATES, POLLUTANT_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline(edge_threshold=0.2, alpha=0.7, hour=167):
    """Shared pipeline fixture used across multiple tests."""
    processor = DataProcessor()
    df_master = processor.generate_telemetry_stream()
    df_window = processor.get_window(df_master, hour)
    pca_df = processor.compute_pca(df_window)

    net = AtmosphericNetwork(threshold=edge_threshold)
    sim_df = net.compute_similarity_matrix(pca_df)
    net.build_graph(sim_df)
    mst = net.maximum_spanning_tree()

    engine = RiskEngine(alpha=alpha)
    risk_df, risk_scores = engine.run(mst, pca_df)

    return df_master, df_window, pca_df, mst, risk_df, risk_scores


# ──────────────────────────────────────────────────────────────────────────────
# DATA PROCESSOR TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_stream_shape():
    """Telemetry stream must have 168h × 20 cities rows."""
    processor = DataProcessor()
    df = processor.generate_telemetry_stream()
    expected_rows = 168 * len(CITY_COORDINATES)
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(df)}"
    )
    assert set(POLLUTANT_FEATURES).issubset(df.columns), (
        "Missing pollutant columns"
    )
    print(f"  ✅  test_stream_shape: {len(df)} rows, {len(df.columns)} columns")


def test_stream_reproducible():
    """Two separate DataProcessor instances should produce identical data."""
    p1 = DataProcessor()
    p2 = DataProcessor()
    df1 = p1.generate_telemetry_stream()
    df2 = p2.generate_telemetry_stream()
    assert df1["PM25"].sum() == df2["PM25"].sum(), "Streams are not reproducible"
    print("  ✅  test_stream_reproducible: streams are deterministic")


def test_window_extraction():
    """Window should contain only hours in (current_hour-24, current_hour]."""
    processor = DataProcessor()
    df = processor.generate_telemetry_stream()
    window = processor.get_window(df, current_hour=100)
    assert window["hour"].min() > 76, "Window lower bound incorrect"
    assert window["hour"].max() <= 100, "Window upper bound incorrect"
    print(f"  ✅  test_window_extraction: hours {window['hour'].min()}–{window['hour'].max()}")


def test_pca_output():
    """PCA must return one Z_index per city, all strictly positive."""
    processor = DataProcessor()
    df = processor.generate_telemetry_stream()
    window = processor.get_window(df, 167)
    pca_df = processor.compute_pca(window)
    assert "Z_index" in pca_df.columns, "Z_index column missing"
    assert len(pca_df) == len(CITY_COORDINATES), "City count mismatch"
    assert (pca_df["Z_index"] > 0).all(), "Z_index values must all be positive"
    print(f"  ✅  test_pca_output: Z_index range [{pca_df['Z_index'].min():.3f}, {pca_df['Z_index'].max():.3f}]")


# ──────────────────────────────────────────────────────────────────────────────
# NETWORK TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_similarity_matrix():
    """Similarity matrix should have n*(n-1)/2 rows for n cities."""
    processor = DataProcessor()
    df = processor.generate_telemetry_stream()
    pca_df = processor.compute_pca(processor.get_window(df, 167))
    net = AtmosphericNetwork()
    sim_df = net.compute_similarity_matrix(pca_df)
    n = len(CITY_COORDINATES)
    expected = n * (n - 1) // 2
    assert len(sim_df) == expected, f"Expected {expected} pairs, got {len(sim_df)}"
    assert (sim_df["weight"] > 0).all(), "All weights must be positive"
    assert (sim_df["weight"] <= 1).all(), "RBF weights must be ≤ 1"
    print(f"  ✅  test_similarity_matrix: {len(sim_df)} pairs, weights in (0, 1]")


def test_mst_is_tree():
    """Kruskal MaxST must be a spanning forest: n - k edges (k = connected components)."""
    _, _, _, mst, _, _ = _run_pipeline()
    n = mst.number_of_nodes()
    e = mst.number_of_edges()
    k = nx.number_connected_components(mst)
    # A spanning forest of a graph with k components has exactly n-k edges
    assert e == n - k, f"Spanning forest should have {n-k} edges ({k} components), got {e}"
    # Each component must itself be a tree (acyclic)
    for component in nx.connected_components(mst):
        sub = mst.subgraph(component)
        assert nx.is_tree(sub), f"Component {component} is not a tree"
    print(f"  ✅  test_mst_is_tree: {n} nodes, {e} edges, {k} component(s) — valid spanning forest")


def test_graph_node_coverage():
    """All 20 Indo-Gangetic cities must appear as graph nodes."""
    _, _, _, mst, _, _ = _run_pipeline()
    missing = set(CITY_COORDINATES.keys()) - set(mst.nodes())
    assert len(missing) == 0, f"Missing nodes in MST: {missing}"
    print(f"  ✅  test_graph_node_coverage: all {mst.number_of_nodes()} cities present")


# ──────────────────────────────────────────────────────────────────────────────
# RISK ENGINE TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_risk_scores_positive():
    """All risk scores must be strictly positive."""
    _, _, _, _, risk_df, risk_scores = _run_pipeline()
    assert all(v > 0 for v in risk_scores.values()), "Risk scores must be positive"
    print(f"  ✅  test_risk_scores_positive: min={min(risk_scores.values()):.3f}")


def test_risk_dataframe_columns():
    """Risk DataFrame must have required columns and correct ranking."""
    _, _, _, _, risk_df, _ = _run_pipeline()
    required = {"city", "Z_index", "risk_score", "risk_rank"}
    assert required.issubset(risk_df.columns), f"Missing columns: {required - set(risk_df.columns)}"
    assert risk_df["risk_rank"].iloc[0] == 1, "Top-ranked city must have rank 1"
    assert risk_df["risk_score"].is_monotonic_decreasing, "Risk scores must be descending"
    print("  ✅  test_risk_dataframe_columns: all columns present, ranks correct")


def test_high_risk_cities_ranked_high():
    """Delhi, Noida, Gurgaon, Patna should appear in the top half."""
    _, _, _, _, risk_df, _ = _run_pipeline()
    high_risk = {"Delhi", "Noida", "Gurgaon", "Patna"}
    half = len(risk_df) // 2
    top_half = set(risk_df.head(half)["city"].tolist())
    overlap = high_risk & top_half
    assert len(overlap) >= 2, (
        f"Expected at least 2 high-risk cities in top half, found: {overlap}"
    )
    print(f"  ✅  test_high_risk_cities_ranked_high: {overlap} in top-{half}")


# ──────────────────────────────────────────────────────────────────────────────
# MISRA-GRIES TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_misra_gries_output():
    """Misra-Gries should return a non-empty DataFrame for known high-emission cities."""
    processor = DataProcessor()
    df = processor.generate_telemetry_stream()
    df_mg = MisraGries.from_dataframe(df, k=6, threshold=150)
    assert not df_mg.empty, "MG heavy hitters should not be empty"
    assert "City" in df_mg.columns, "City column missing"
    assert "Severe Breaches (Lower Bound)" in df_mg.columns
    # Delhi or Patna should always appear
    cities_found = set(df_mg["City"].tolist())
    known = {"Delhi", "Noida", "Gurgaon", "Patna"}
    assert len(cities_found & known) >= 1, (
        f"Expected at least 1 known high-risk city in MG output, got: {cities_found}"
    )
    print(f"  ✅  test_misra_gries_output: heavy hitters = {cities_found}")


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICAL VALIDATOR TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_range():
    """Bootstrap CI must be ordered and finite."""
    np.random.seed(42)
    data = np.random.normal(150, 20, 100)
    validator = StatisticalValidator()
    ci_low, ci_high = validator.bootstrap_ci(data, B=200)
    assert ci_low < ci_high, "CI lower must be less than upper"
    assert np.isfinite(ci_low) and np.isfinite(ci_high), "CI values must be finite"
    print(f"  ✅  test_bootstrap_ci_range: CI = [{ci_low:.1f}, {ci_high:.1f}]")


def test_permutation_test_anomaly():
    """Clearly anomalous data (high current vs low baseline) should yield p < 0.05."""
    np.random.seed(42)
    curr = np.random.normal(200, 10, 50)     # high current window
    hist = np.random.normal(50, 10, 200)     # low historical baseline
    validator = StatisticalValidator()
    p_val = validator.permutation_test(curr, hist, B=500)
    assert p_val < 0.05, f"Expected p<0.05 for clear anomaly, got p={p_val:.4f}"
    print(f"  ✅  test_permutation_test_anomaly: p-value = {p_val:.4f} (< 0.05)")


def test_permutation_test_null():
    """Identical distributions should NOT yield p < 0.05."""
    np.random.seed(42)
    curr = np.random.normal(80, 15, 50)
    hist = np.random.normal(80, 15, 200)
    validator = StatisticalValidator()
    p_val = validator.permutation_test(curr, hist, B=500)
    assert p_val >= 0.05, f"Expected p≥0.05 for null case, got p={p_val:.4f}"
    print(f"  ✅  test_permutation_test_null: p-value = {p_val:.4f} (≥ 0.05)")


def test_compute_all_keys():
    """compute_all must return dict with all required keys."""
    processor = DataProcessor()
    df_master = processor.generate_telemetry_stream()
    df_window = processor.get_window(df_master, 167)
    validator = StatisticalValidator()
    result = validator.compute_all("Delhi", df_window, df_master, 167)
    required_keys = {"ci_low", "ci_high", "p_val", "status"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    assert result["status"] in ("CRITICAL", "OPERATIONAL"), (
        f"Invalid status string: {result['status']}"
    )
    print(
        f"  ✅  test_compute_all_keys: Delhi → "
        f"CI=[{result['ci_low']:.1f},{result['ci_high']:.1f}] "
        f"p={result['p_val']:.4f} status={result['status']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # DataProcessor
    test_stream_shape,
    test_stream_reproducible,
    test_window_extraction,
    test_pca_output,
    # Network
    test_similarity_matrix,
    test_mst_is_tree,
    test_graph_node_coverage,
    # RiskEngine
    test_risk_scores_positive,
    test_risk_dataframe_columns,
    test_high_risk_cities_ranked_high,
    # MisraGries
    test_misra_gries_output,
    # StatisticalValidator
    test_bootstrap_ci_range,
    test_permutation_test_anomaly,
    test_permutation_test_null,
    test_compute_all_keys,
]


if __name__ == "__main__":
    sep = "-" * 60
    print(f"\n{sep}")
    print("  AtmosRisk Intelligence Platform - Test Suite")
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
            print(f"    * {name}: {msg}")
        sys.exit(1)
    else:
        print("\n  All tests passed. [OK]\n")
        sys.exit(0)