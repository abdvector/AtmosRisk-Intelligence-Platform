"""
helpers.py

Reusable helper functions.
"""

import numpy as np


def normalize_positive(values):
    """
    Shifts values so all become positive.
    """
    return values - values.min() + 1


def rbf_similarity(x, y):
    """
    Radial basis similarity.
    """
    return np.exp(-abs(x - y))