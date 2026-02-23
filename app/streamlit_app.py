"""
streamlit_app.py

AtmosRisk Intelligence Platform — Professional Interactive Dashboard.

Sections:
  - Sidebar: Stream epoch, alpha, sparsification
  - Main Layout: 3 Tabs (System Overview, Spatial Analytics, Statistical Inference)
"""

import sys
from pathlib import Path
import time

# Ensure project root is on PYTHONPATH regardless of launch directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd

from src.data import DataProcessor
from src.network import AtmosphericNetwork
from src.risk import RiskEngine, MisraGries
from src.stats import StatisticalValidator
from src.visualization import (
    build_geo_map,
    build_trend_chart,
    build_risk_bar,
    build_network_graph,
)
from src.utils.config import (
    DEFAULT_ALPHA,
    DEFAULT_EDGE_THRESHOLD,
    SEVERE_PM25_THRESHOLD,
)
from src.utils.constants import CITY_COORDINATES

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AtmosRisk Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Vecto Shield-style premium dark theme
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif !important; }
        
        /* Global dark background */
        .stApp { background-color: #0B0F19; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #06090F;
            border-right: 1px solid #1A202C;
        }

        /* Section headers */
        h1, h2, h3 { color: #E2E8F0 !important; font-weight: 600 !important; }

        /* KPI metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #00FFCC !important;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
        }
        [data-testid="stMetricLabel"] {
            color: #A0AEC0 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Alert boxes */
        .anomaly-row {
            background: rgba(255, 82, 82, 0.1);
            border: 1px solid #FF5252;
            border-radius: 6px;
            padding: 10px 15px;
            margin-bottom: 8px;
            color: #FF5252;
            font-weight: 600;
        }
        .stable-row {
            background: rgba(0, 200, 83, 0.05);
            border: 1px solid #00C853;
            border-radius: 6px;
            padding: 10px 15px;
            margin-bottom: 8px;
            color: #00C853;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #111827;
            border-radius: 4px 4px 0 0;
            border: 1px solid #1F2937;
            border-bottom: none;
            color: #9CA3AF;
            padding: 0 25px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1F2937;
            color: #00FFCC;
            border-top: 2px solid #00FFCC;
        }

        /* Divider */
        hr { border-color: #1A202C !important; }
        
        /* Top header bar */
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: #111827;
            border-radius: 8px;
            border: 1px solid #1F2937;
            margin-bottom: 25px;
        }
        .top-bar-title {
            color: #00FFCC;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 1px;
            margin: 0;
        }
        .top-bar-status {
            color: #00C853;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

current_date = time.strftime("%d %b %Y")

st.markdown(
    f"""
    <div class="top-bar">
      <div>
        <h1 class="top-bar-title">ATMOSRISK INTELLIGENCE PLATFORM</h1>
        <p style="color:#A0AEC0; font-size:0.85rem; margin:0;">Advanced Autonomous Vector & Risk Control System</p>
      </div>
      <div style="text-align: right;">
        <span style="color:#A0AEC0; font-size:0.8rem; margin-right: 15px;">SYSTEM STATUS: <span class="top-bar-status">OPERATIONAL</span></span>
        <span style="color:#A0AEC0; font-size:0.8rem;">DATE: <span style="color:#E2E8F0; font-weight:600;">{current_date}</span></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — STREAM CONTROLS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h3 style='color:#E2E8F0; font-size:1.1rem; text-transform:uppercase; letter-spacing:1px;'>Stream Constraints</h3>",
        unsafe_allow_html=True,
    )

    current_hour = st.slider(
        "Current Epoch (Hour)",
        min_value=24,
        max_value=167,
        value=167,
        help="Defines the right boundary of the 24h sliding window.",
    )

    alpha = st.slider(
        "Local vs. Transport Weight (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_ALPHA,
        step=0.05,
        help=(
            "Higher Alpha = local emission dominates risk. "
            "Lower Alpha = network-transported smog dominates."
        ),
    )

    edge_threshold = st.slider(
        "Graph Sparsification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=DEFAULT_EDGE_THRESHOLD,
        step=0.05,
        help="RBF similarity threshold. Edges below this weight are pruned.",
    )

    st.divider()
    st.markdown(
        "<small style='color:#718096;'>Sliding window: trailing 24h from epoch.</small>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_master_data():
    """Generate and cache the full 168-hour telemetry stream."""
    processor = DataProcessor()
    return processor.generate_telemetry_stream()

df_master = load_master_data()

with st.spinner("Computing spatial topology via Kruskal's algorithm..."):
    # 1. Sliding window extraction
    processor = DataProcessor()
    df_window = processor.get_window(df_master, current_hour)

    # 2. PCA latent pollution index
    pca_df = processor.compute_pca(df_window)

    # 3. Build base similarity graph + MST
    net = AtmosphericNetwork(threshold=edge_threshold)
    sim_df = net.compute_similarity_matrix(pca_df)
    base_graph = net.build_graph(sim_df)

    try:
        mst = net.maximum_spanning_tree()
    except ValueError:
        st.warning("Edge threshold too aggressive — falling back to full similarity graph.")
        import networkx as nx
        mst = base_graph.copy()

    # Attach geo coordinates
    for node in mst.nodes():
        if node in CITY_COORDINATES:
            mst.nodes[node]["lat"] = CITY_COORDINATES[node][0]
            mst.nodes[node]["lon"] = CITY_COORDINATES[node][1]

    # 4. Risk scoring
    engine = RiskEngine(alpha=alpha)
    risk_df, risk_scores = engine.run(mst, pca_df)

    # 5. Statistical inference (bootstrap CI + permutation test)
    validator = StatisticalValidator()
    stat_results = {
        city: validator.compute_all(city, df_window, df_master, current_hour)
        for city in mst.nodes()
    }

    # 6. Misra-Gries heavy hitter detection
    df_mg = MisraGries.from_dataframe(
        df_master,
        k=6,
        threshold=SEVERE_PM25_THRESHOLD,
    )

anomalies = [city for city, stats in stat_results.items() if stats["p_val"] < 0.05]

# ──────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["1 SYSTEM OVERVIEW", "2 SPATIAL ANALYTICS", "3 STATISTICAL INFERENCE"])

# ── TAB 1: SYSTEM OVERVIEW ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### PLATFORM CAPABILITIES")
    st.markdown(
        """
        <div style="color: #A0AEC0; font-size: 1rem; line-height: 1.6; max-width: 900px; margin-bottom: 30px;">
            The AtmosRisk Intelligence Platform provides real-time, network-propagated air quality risk assessment 
            across the Indo-Gangetic Plain. By utilizing Principal Component Analysis (PCA) and Kruskal's Maximum Spanning Tree, 
            the system isolates dominant atmospheric transport corridors. Statistical permutations and Bootstrap Confidence 
            Intervals ensure rigorous anomaly detection.
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("### LIVE METRICS")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("CITIES MONITORED", len(risk_df))
    kpi2.metric("ACTIVE MST NODES", mst.number_of_nodes())
    kpi3.metric("TRANSPORT EDGES", mst.number_of_edges())
    kpi4.metric("CRITICAL ANOMALIES", len(anomalies))
    kpi5.metric("EPOCH WINDOW", f"H{current_hour - 24} - H{current_hour}")

# ── TAB 2: SPATIAL ANALYTICS ───────────────────────────────────────────────────
with tab2:
    col_map, col_net = st.columns([1.5, 1])
    
    with col_map:
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 0;'>GEOGRAPHIC RISK TOPOLOGY</h4>", unsafe_allow_html=True)
        st.caption("Maximum Spanning Tree backbone overlaid on satellite terrain.")
        fig_map = build_geo_map(mst, risk_scores, stat_results)
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_net:
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 0;'>FORCE-DIRECTED NETWORK</h4>", unsafe_allow_html=True)
        st.caption("Abstract spring layout of the topological transport graph.")
        fig_net = build_network_graph(mst, risk_scores)
        st.plotly_chart(fig_net, use_container_width=True)

# ── TAB 3: STATISTICAL INFERENCE ───────────────────────────────────────────────
with tab3:
    col_trend, col_bar, col_mg = st.columns([2, 1.5, 1.5])
    
    with col_trend:
        fig_trend = build_trend_chart(df_master, risk_scores, current_hour)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_bar:
        fig_bar = build_risk_bar(risk_scores)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_mg:
        st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 0; font-size: 14px;'>MISRA-GRIES HEAVY HITTERS</h4>", unsafe_allow_html=True)
        st.caption(f"Chronic PM₂.₅ > {SEVERE_PM25_THRESHOLD} µg/m³ offenders.")
        if not df_mg.empty:
            st.dataframe(df_mg, hide_index=True, use_container_width=True)
        else:
            st.info("No cities exceeded the severe threshold.")
            
    st.divider()
    
    st.markdown("<h4 style='color:#E2E8F0; margin-bottom: 0;'>VALIDATION ENGINE & ANOMALY DETECTION</h4>", unsafe_allow_html=True)
    st.caption("Permutation test (B=1000) comparing current window against historical baseline.")
    
    anomaly_records = []
    for node in mst.nodes():
        stats = stat_results.get(node, {})
        if stats.get("p_val", 1.0) < 0.05:
            anomaly_records.append({
                "City": node,
                "Risk Score": f"{risk_scores.get(node, 0):.3f}",
                "95% CI": f"[{stats['ci_low']:.1f}, {stats['ci_high']:.1f}]",
                "p-value": f"{stats['p_val']:.4f}",
                "Status": stats["status"],
            })

    if anomaly_records:
        df_anomalies = pd.DataFrame(anomaly_records)
        st.markdown(f"<div class='anomaly-row'>CRITICAL: {len(anomaly_records)} anomalous node(s) detected in the current epoch.</div>", unsafe_allow_html=True)
        st.dataframe(df_anomalies, hide_index=True, use_container_width=True)
    else:
        st.markdown("<div class='stable-row'>OPERATIONAL: No statistically significant spikes detected. Null hypothesis retained.</div>", unsafe_allow_html=True)

    with st.expander("VIEW FULL INFERENCE TABLE"):
        all_records = []
        for node in sorted(mst.nodes()):
            stats = stat_results.get(node, {})
            all_records.append({
                "City": node,
                "Risk Score": round(risk_scores.get(node, 0), 3),
                "Z-Index": round(pca_df.set_index("city").loc[node, "Z_index"] if node in pca_df["city"].values else 0, 3),
                "CI Low": round(stats.get("ci_low", 0), 1),
                "CI High": round(stats.get("ci_high", 0), 1),
                "p-value": round(stats.get("p_val", 1.0), 4),
                "Status": stats.get("status", "OPERATIONAL"),
            })

        df_full = pd.DataFrame(all_records).sort_values("Risk Score", ascending=False)
        st.dataframe(df_full, hide_index=True, use_container_width=True) 