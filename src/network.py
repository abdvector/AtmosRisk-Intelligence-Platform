import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

from src.utils.helpers import rbf_similarity
from src.utils.config import DEFAULT_EDGE_THRESHOLD


class AtmosphericNetwork:
    """
    Builds an atmospheric influence graph from city-level PCA scores.
    """

    def __init__(self, threshold=DEFAULT_EDGE_THRESHOLD):
        self.threshold = threshold
        self.graph = nx.Graph()

    def compute_similarity_matrix(self, pca_df):
        """
        Compute pairwise RBF similarities.
        """
        if "city" in pca_df.columns:
            pca_df = pca_df.set_index("city")
        cities = pca_df.index.tolist()

        similarities = []

        for city_a, city_b in combinations(cities, 2):

            score_a = pca_df.loc[city_a, "Z_index"]
            score_b = pca_df.loc[city_b, "Z_index"]

            similarity = rbf_similarity(score_a, score_b)

            similarities.append(
                {
                    "source": city_a,
                    "target": city_b,
                    "weight": similarity,
                }
            )

        return pd.DataFrame(similarities)

    def build_graph(self, similarity_df):
        """
        Build weighted graph.
        """

        G = nx.Graph()

        for _, row in similarity_df.iterrows():

            if row["weight"] >= self.threshold:

                G.add_edge(
                    row["source"],
                    row["target"],
                    weight=float(row["weight"]),
                )

        self.graph = G

        return G

    def maximum_spanning_tree(self):
        """
        Kruskal Maximum Spanning Tree.
        """

        if self.graph.number_of_edges() == 0:
            raise ValueError("Graph has no edges.")

        mst = nx.maximum_spanning_tree(
            self.graph,
            algorithm="kruskal",
            weight="weight",
        )

        return mst

    def add_node_attributes(self, graph, pca_df):
        """
        Attach risk scores to graph nodes.
        """

        if "city" in pca_df.columns:
            pca_df = pca_df.set_index("city")

        for city in graph.nodes():

            graph.nodes[city]["risk_score"] = float(
                pca_df.loc[city, "Z_index"]
            )

        return graph

    def network_statistics(self, graph):

        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "average_degree": np.mean(
                [d for _, d in graph.degree()]
            ),
        }