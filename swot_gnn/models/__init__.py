# Main model: spatio-temporal GNN (InputEncoder -> STBlocks -> Readout)
from .swot_gnn import SWOTGNN
# Probabilistic variant: Gaussian mean + log-std output
from .swot_gnn_gauss import SWOTGNNGauss
# GraphGPS: local GAT + global attention (used inside STBlock)
from .graph_gps_layer import GraphGPSLayer
# Baselines for comparison
from .baselines import GPSGNN, LSTMBaseline, drainage_area_ratio
# Multi-day LSTM-only baseline (deterministic and probabilistic)
from .lstm_baseline_nd import LSTMBaselineMultiStep, LSTMBaselineMultiStepGauss
# Model registry: maps model_type string to ModelSpec (class, loss, slug)
from .registry import MODEL_REGISTRY, ModelSpec

__all__ = [
    "SWOTGNN", "SWOTGNNGauss",
    "GraphGPSLayer",
    "GPSGNN", "LSTMBaseline", "drainage_area_ratio",
    "LSTMBaselineMultiStep", "LSTMBaselineMultiStepGauss",
    "MODEL_REGISTRY", "ModelSpec",
]
