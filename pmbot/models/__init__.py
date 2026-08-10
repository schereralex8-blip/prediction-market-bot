"""Two independent prop models plus the gate that makes them agree."""

from .base import PropModel
from .consensus import evaluate_consensus
from .model_a import NAME as MODEL_A_NAME, BayesRateModel
from .model_b import NAME as MODEL_B_NAME, BootstrapModel

__all__ = [
    "PropModel",
    "BayesRateModel",
    "BootstrapModel",
    "MODEL_A_NAME",
    "MODEL_B_NAME",
    "evaluate_consensus",
]
