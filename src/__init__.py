"""
AtmosRisk Intelligence Platform — src package.

Public API:
    DataProcessor    — telemetry simulation, PCA, windowing
    AtmosphericNetwork — RBF graph, Kruskal MST
    RiskEngine       — alpha-weighted risk scoring
    MisraGries       — heavy-hitter detection
    StatisticalValidator — bootstrap CI + permutation test
"""

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

__all__ = [
    "DataProcessor",
    "AtmosphericNetwork",
    "RiskEngine",
    "MisraGries",
    "StatisticalValidator",
    "build_geo_map",
    "build_trend_chart",
    "build_risk_bar",
    "build_network_graph",
]
