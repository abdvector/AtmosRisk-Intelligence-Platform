"""
data.py

Data generation, streaming simulation,
Misra-Gries heavy hitter detection,
and PCA preprocessing.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    CITY_COORDINATES,
    POLLUTANT_FEATURES,
    HIGH_RISK_CITIES
)

from src.utils.config import (
    RANDOM_SEED,
    WINDOW_HOURS,
    PCA_COMPONENTS
)

from src.utils.helpers import normalize_positive


class DataProcessor:

    def __init__(self):
        pass

    def generate_telemetry_stream(self):
        """
        Simulates historical AQI telemetry.
        Always reseeds before generation to guarantee reproducibility.
        """
        np.random.seed(RANDOM_SEED)
        data = []

        for city in CITY_COORDINATES.keys():

            base = (
                150
                if city in HIGH_RISK_CITIES
                else np.random.uniform(50, 100)
            )

            for t in range(WINDOW_HOURS):

                cycle = np.sin(
                    t / 24 * 2 * np.pi
                ) * 20

                data.append(
                    {
                        "city": city,
                        "hour": t,
                        "PM25": max(
                            10,
                            np.random.normal(
                                base + cycle,
                                20
                            )
                        ),
                        "PM10": max(
                            15,
                            np.random.normal(
                                (base + cycle) * 1.2,
                                25
                            )
                        ),
                        "NO2": np.random.normal(
                            35,
                            10
                        ),
                        "SO2": np.random.normal(
                            15,
                            5
                        ),
                    }
                )

        return pd.DataFrame(data)

    def get_window(
        self,
        dataframe,
        current_hour,
        window_size=24
    ):
        """
        Extract rolling window.
        """

        return dataframe[
            (dataframe["hour"] > current_hour - window_size)
            &
            (dataframe["hour"] <= current_hour)
        ]

    def compute_pca(
        self,
        dataframe
    ):
        """
        Compute latent pollution index.
        """

        df_mean = (
            dataframe
            .groupby("city")[POLLUTANT_FEATURES]
            .mean()
            .reset_index()
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            df_mean[POLLUTANT_FEATURES]
        )

        pca = PCA(
            n_components=PCA_COMPONENTS
        )

        z_index = pca.fit_transform(
            X_scaled
        )

        df_mean["Z_index"] = (
            normalize_positive(
                z_index.flatten()
            )
        )

        return df_mean


class MisraGries:

    def __init__(
        self,
        k=4,
        threshold=150
    ):
        self.k = k
        self.threshold = threshold

    def fit(
        self,
        dataframe
    ):
        """
        Misra-Gries heavy hitter detection.
        """

        counters = {}

        for _, row in dataframe.iterrows():

            city = row["city"]
            pm25 = row["PM25"]

            if pm25 <= self.threshold:
                continue

            if city in counters:

                counters[city] += 1

            elif len(counters) < self.k:

                counters[city] = 1

            else:

                remove_keys = []

                for key in counters:

                    counters[key] -= 1

                    if counters[key] == 0:
                        remove_keys.append(key)

                for key in remove_keys:
                    del counters[key]

        return counters

    def dataframe(
        self,
        dataframe
    ):
        """
        Convert results into dashboard dataframe.
        """

        results = self.fit(dataframe)

        output = [
            {
                "City": city,
                "Severe Breaches (Lower Bound)": count
            }
            for city, count in results.items()
        ]

        if len(output) == 0:
            return pd.DataFrame()

        return (
            pd.DataFrame(output)
            .sort_values(
                "Severe Breaches (Lower Bound)",
                ascending=False
            )
        ) 