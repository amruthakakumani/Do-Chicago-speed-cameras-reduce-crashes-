"""Single-model machine-learning layer for crash-count prediction.

The ML layer is deliberately independent of the causal estimators: it predicts
observed crash counts under observed patterns and never claims a causal effect.
"""

from __future__ import annotations

__all__ = [
    "dataset",
    "evaluate",
    "explain",
    "external",
    "predict",
    "train",
    "tune",
    "validation",
]
