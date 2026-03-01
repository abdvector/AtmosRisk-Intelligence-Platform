"""
main.py

CLI entry point for the AtmosRisk Intelligence Platform.

Usage:
    python app/main.py
    python app/main.py --hour 120 --alpha 0.6 --threshold 0.3
"""

import sys
import io
import argparse
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import DataProcessor
from src.network import AtmosphericNetwork
from src.risk import RiskEngine, MisraGries
from src.stats import StatisticalValidator
from src.utils.config import (
    DEFAULT_ALPHA,
    DEFAULT_EDGE_THRESHOLD,
    SEVERE_PM25_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AtmosRisk Intelligence Platform — CLI Runner"
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=167,
        help="Current epoch hour (24-167). Defines right boundary of 24h window.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Local vs transport weight α (0.1–0.9).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help="RBF graph sparsification threshold (0.1–0.9).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top-risk cities to display.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(current_hour, alpha, edge_threshold, top_k):
    """
    Execute the full AtmosRisk pipeline and print results to stdout.
    """
    sep = "─" * 60

    print(f"\n{sep}")
    print("  AtmosRisk Intelligence Platform — CLI Report")
    print(f"  Epoch window : h{current_hour - 24} → h{current_hour}")
    print(f"  α (alpha)    : {alpha}")
    print(f"  Edge thresh  : {edge_threshold}")
    print(sep)

    # Step 1 — Telemetry stream
    print("\n[1/5] Generating telemetry stream…")
    processor = DataProcessor()
    df_master = processor.generate_telemetry_stream()
    df_window = processor.get_window(df_master, current_hour)
    print(f"      Master rows : {len(df_master):,}  |  Window rows : {len(df_window):,}")

    # Step 2 — PCA
    print("[2/5] Computing PCA latent pollution index…")
    pca_df = processor.compute_pca(df_window)
    print(f"      Cities processed : {len(pca_df)}")

    # Step 3 — Graph & MST
    print("[3/5] Building RBF similarity graph + Kruskal MaxST…")
    net = AtmosphericNetwork(threshold=edge_threshold)
    sim_df = net.compute_similarity_matrix(pca_df)
    net.build_graph(sim_df)
    mst = net.maximum_spanning_tree()
    stats_net = net.network_statistics(mst)
    print(
        f"      Nodes : {stats_net['nodes']}  |  "
        f"Edges : {stats_net['edges']}  |  "
        f"Density : {stats_net['density']:.3f}"
    )

    # Step 4 — Risk scores
    print("[4/5] Computing risk scores…")
    engine = RiskEngine(alpha=alpha)
    risk_df, risk_scores = engine.run(mst, pca_df)

    # Step 5 — Statistical inference
    print("[5/5] Running bootstrap CI + permutation tests…")
    validator = StatisticalValidator()
    stat_results = {
        city: validator.compute_all(city, df_window, df_master, current_hour)
        for city in mst.nodes()
    }
    anomalies = [c for c, s in stat_results.items() if s["p_val"] < 0.05]

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TOP {top_k} RISK CITIES")
    print(sep)
    display = risk_df[["risk_rank", "city", "Z_index", "risk_score"]].head(top_k)
    display.columns = ["Rank", "City", "Z-Index", "Risk Score"]
    print(display.to_string(index=False))

    print(f"\n{sep}")
    print("  MISRA-GRIES HEAVY HITTERS")
    print(f"  (PM₂.₅ > {SEVERE_PM25_THRESHOLD} µg/m³ chronic offenders)")
    print(sep)
    df_mg = MisraGries.from_dataframe(df_master, k=6)
    if not df_mg.empty:
        print(df_mg.to_string(index=False))
    else:
        print("  No heavy hitters detected.")

    print(f"\n{sep}")
    print("  STATISTICAL VALIDATION ENGINE")
    print(sep)
    if anomalies:
        print(f"  🚨  {len(anomalies)} ANOMALOUS NODES detected (p < 0.05):\n")
        for city in anomalies:
            s = stat_results[city]
            print(
                f"      {city:15s}  "
                f"p={s['p_val']:.4f}  "
                f"CI=[{s['ci_low']:.1f}, {s['ci_high']:.1f}]  "
                f"Risk={risk_scores[city]:.3f}"
            )
    else:
        print("  ✅  No statistically significant anomalies. Null hypothesis retained.")

    print(f"\n{sep}\n")

    return risk_df, risk_scores, stat_results


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        current_hour=args.hour,
        alpha=args.alpha,
        edge_threshold=args.threshold,
        top_k=args.top,
    ) 