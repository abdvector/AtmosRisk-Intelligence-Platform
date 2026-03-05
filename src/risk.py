"""
risk.py

Risk scoring engine for the AtmosRisk Intelligence Platform.

Implements:
  - RiskEngine: computes per-city risk scores using the original
    alpha-weighted local + transport formula:
      risk(i) = alpha * Z_i + (1 - alpha) * sum_j(w_ij * Z_j)
  - MisraGries: streaming heavy-hitter detection on PM2.5 threshold breaches
"""

import numpy as np
import pandas as pd
import networkx as nx

from src.utils.config import (
    MISRA_GRIES_K,
    DEFAULT_ALPHA,
    SEVERE_PM25_THRESHOLD,
)


class MisraGries:
    """
    Misra-Gries deterministic heavy-hitter algorithm.

    Maintains exactly k-1 counters and deterministically identifies
    the top-k frequent items in a stream in O(n) time and O(k) space.

    Parameters
    ----------
    k : int
        Number of counters to maintain.
    """

    def __init__(self, k=MISRA_GRIES_K):
        self.k = k
        self.counters = {}

    def fit(self, stream):
        """
        Process a stream of items and maintain the top-k counter set.

        Parameters
        ----------
        stream : iterable
            Sequence of hashable items (city names from high-PM25 events).

        Returns
        -------
        self
        """
        for item in stream:

            if item in self.counters:
                self.counters[item] += 1

            elif len(self.counters) < self.k - 1:
                self.counters[item] = 1

            else:
                keys_to_remove = []

                for key in self.counters:
                    self.counters[key] -= 1

                    if self.counters[key] == 0:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.counters[key]

        return self

    def heavy_hitters(self):
        """
        Return current counter state sorted by frequency descending.

        Returns
        -------
        dict[str, int]
        """
        return dict(
            sorted(
                self.counters.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    @classmethod
    def from_dataframe(cls, df, k=MISRA_GRIES_K, threshold=SEVERE_PM25_THRESHOLD):
        """
        Convenience constructor: builds stream from df where PM25 > threshold,
        fits the algorithm, and returns a results DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Full telemetry DataFrame with columns ['city', 'PM25'].
        k : int
        threshold : float

        Returns
        -------
        pd.DataFrame with columns ['City', 'Severe Breaches (Lower Bound)']
        """
        mg = cls(k=k)

        # Build stream: each row where PM25 > threshold contributes the city
        stream = df.loc[df["PM25"] > threshold, "city"].tolist()

        mg.fit(stream)
        hitters = mg.heavy_hitters()

        if not hitters:
            return pd.DataFrame(
                columns=["City", "Severe Breaches (Lower Bound)"]
            )

        return (
            pd.DataFrame(
                [
                    {"City": city, "Severe Breaches (Lower Bound)": count}
                    for city, count in hitters.items()
                ]
            )
            .sort_values("Severe Breaches (Lower Bound)", ascending=False)
            .reset_index(drop=True)
        )


class RiskEngine:
    """
    Computes atmospheric risk scores using the transport-contagion formula:

        risk(i) = alpha * Z_i + (1 - alpha) * sum_{j ∈ N(i)} w_ij * Z_j

    This matches the original code5.py formulation exactly.

    Parameters
    ----------
    alpha : float
        Weight for local emission vs. transported pollution.
        alpha=1.0 → purely local; alpha=0.0 → purely network-transported.
    """

    def __init__(self, alpha=DEFAULT_ALPHA):
        self.alpha = alpha

    def compute_risk_scores(self, graph, pca_df):
        """
        Compute per-city risk scores over the given graph.

        Parameters
        ----------
        graph : nx.Graph
            A NetworkX graph where node attribute 'Z' holds the PCA index.
            (Use the full base graph or the MST — both work.)
        pca_df : pd.DataFrame
            DataFrame with columns ['city', 'Z_index'].

        Returns
        -------
        dict[str, float]
            Mapping of city → risk score.
        """
        # Attach Z_index as node attribute 'Z'
        if "city" in pca_df.columns:
            z_lookup = dict(zip(pca_df["city"], pca_df["Z_index"]))
        else:
            z_lookup = pca_df["Z_index"].to_dict()

        for node in graph.nodes():
            graph.nodes[node]["Z"] = z_lookup.get(node, 1.0)

        risk_scores = {}

        for i in graph.nodes():
            Z_i = graph.nodes[i]["Z"]

            neighbor_sum = sum(
                graph[i][j]["weight"] * graph.nodes[j]["Z"]
                for j in graph.neighbors(i)
            )

            risk_scores[i] = (
                self.alpha * Z_i + (1 - self.alpha) * neighbor_sum
            )

        nx.set_node_attributes(graph, risk_scores, "Risk_Score")

        return risk_scores

    def risk_dataframe(self, risk_scores, pca_df):
        """
        Convert raw risk_scores dict to a tidy ranked DataFrame.

        Parameters
        ----------
        risk_scores : dict[str, float]
        pca_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame with columns:
            city, Z_index, risk_score, risk_rank
        """
        if "city" in pca_df.columns:
            z_lookup = dict(zip(pca_df["city"], pca_df["Z_index"]))
        else:
            z_lookup = pca_df["Z_index"].to_dict()

        records = []
        for city, score in risk_scores.items():
            records.append(
                {
                    "city": city,
                    "Z_index": round(z_lookup.get(city, 0.0), 4),
                    "risk_score": round(score, 4),
                }
            )

        df = (
            pd.DataFrame(records)
            .sort_values("risk_score", ascending=False)
            .reset_index(drop=True)
        )

        df["risk_rank"] = np.arange(1, len(df) + 1)

        return df

    def run(self, graph, pca_df):
        """
        Full pipeline: compute risk scores + Misra-Gries heavy hitters.

        Parameters
        ----------
        graph : nx.Graph
            MST or base graph with node attribute 'Z' (added internally).
        pca_df : pd.DataFrame

        Returns
        -------
        tuple[pd.DataFrame, dict]
            (risk_df, risk_scores)
        """
        risk_scores = self.compute_risk_scores(graph, pca_df)
        risk_df = self.risk_dataframe(risk_scores, pca_df)

        return risk_df, risk_scores 