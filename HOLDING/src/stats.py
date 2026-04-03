"""
stats.py

Statistical inference engine for the AtmosRisk Intelligence Platform.

Implements:
  - Bootstrap 95% Confidence Intervals for PM2.5 window means
  - Permutation significance test comparing current window vs historical baseline
"""

import numpy as np

from src.utils.config import (
    BOOTSTRAP_ITERATIONS,
    PERMUTATION_ITERATIONS,
)


class StatisticalValidator:
    """
    Computes non-parametric statistical inference for each city sensor node.

    Methods
    -------
    bootstrap_ci(city_data, B)
        Generates a 95% confidence interval via resampling.

    permutation_test(curr_pm, hist_pm, B)
        Tests whether the current 24h window is significantly anomalous
        compared to the historical baseline (p < 0.05 → anomaly).

    compute_all(city, df_window, df_master, current_hour)
        Convenience wrapper: runs both tests and returns a result dict.
    """

    def bootstrap_ci(self, city_data, B=BOOTSTRAP_ITERATIONS):
        """
        Generate a 95% bootstrap confidence interval for the PM2.5 mean.

        Parameters
        ----------
        city_data : np.ndarray
            PM2.5 readings for the city in the current window.
        B : int
            Number of bootstrap resamples.

        Returns
        -------
        tuple[float, float]
            (ci_low, ci_high) at the 2.5th and 97.5th percentiles.
        """
        n = len(city_data)

        if n == 0:
            return (0.0, 0.0)

        boot_means = np.empty(B)

        for i in range(B):
            sample = np.random.choice(city_data, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)

        return ci_low, ci_high

    def permutation_test(self, curr_pm, hist_pm, B=PERMUTATION_ITERATIONS):
        """
        Two-sample permutation test: is the current window mean significantly
        higher than the historical baseline mean?

        Parameters
        ----------
        curr_pm : np.ndarray
            PM2.5 readings in the current 24h sliding window.
        hist_pm : np.ndarray
            PM2.5 readings from all prior hours (historical baseline).
        B : int
            Number of permutation iterations.

        Returns
        -------
        float
            p-value. Values < 0.05 indicate a statistically significant anomaly.
        """
        if len(hist_pm) == 0 or len(curr_pm) == 0:
            return 1.0

        observed_diff = np.mean(curr_pm) - np.mean(hist_pm)
        combined = np.concatenate([curr_pm, hist_pm])
        n_curr = len(curr_pm)
        n_hist = len(hist_pm)

        count_extreme = 0

        for _ in range(B):
            perm = np.random.permutation(combined)
            perm_curr_mean = np.mean(perm[:n_curr])
            perm_hist_mean = np.mean(perm[n_curr:n_curr + n_hist])
            if (perm_curr_mean - perm_hist_mean) >= observed_diff:
                count_extreme += 1

        return count_extreme / B

    def compute_all(self, city, df_window, df_master, current_hour):
        """
        Run bootstrap CI and permutation test for a single city.

        Parameters
        ----------
        city : str
        df_window : pd.DataFrame
            Current sliding window DataFrame (all cities).
        df_master : pd.DataFrame
            Full historical DataFrame (all cities, all hours).
        current_hour : int
            The current epoch hour used as the window boundary.

        Returns
        -------
        dict with keys: ci_low, ci_high, p_val, status
        """
        curr_pm = df_window[df_window["city"] == city]["PM25"].values

        hist_pm = df_master[
            (df_master["city"] == city)
            & (df_master["hour"] < current_hour - 24)
        ]["PM25"].values

        ci_low, ci_high = self.bootstrap_ci(curr_pm)
        p_val = self.permutation_test(curr_pm, hist_pm)

        status = "CRITICAL" if p_val < 0.05 else "OPERATIONAL"

        return {
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_val": p_val,
            "status": status,
        }