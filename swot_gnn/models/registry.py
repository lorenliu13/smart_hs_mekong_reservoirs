"""
Model registry for SWOT-GNN experiments.

Maps model_type string (used in YAML configs) to a ModelSpec that bundles
the model class, its corresponding loss class, output mode, and run-name slug.

To add a new model type:
    1. Implement the model class (e.g. models/swot_gnn_foo.py).
    2. Add a new ModelSpec entry to MODEL_REGISTRY below.
    3. No changes needed in training or inference scripts.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import torch.nn as nn

# Ensure the swot_gnn package root is importable when this module is loaded
# both as part of the package and from scripts at the root level.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.swot_gnn import SWOTGNN
from models.swot_gnn_gauss import SWOTGNNGauss
from training.train import ObservedMSELoss, ObservedGaussianNLLLoss


@dataclass
class ModelSpec:
    """Bundles everything needed to instantiate and train a model variant."""
    model_cls:   Type[nn.Module]  # model class; instantiated with **model_cfg
    loss_cls:    Type[nn.Module]  # loss class; instantiated with no arguments
    output_mode: str              # 'point' | 'gaussian' | future: 'quantile'
    slug:        str              # short id appended to run name, e.g. 'swotgnn_gauss'


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "SWOTGNN": ModelSpec(
        model_cls=SWOTGNN,
        loss_cls=ObservedMSELoss,
        output_mode="point",
        slug="swotgnn",
    ),
    "SWOTGNNGauss": ModelSpec(
        model_cls=SWOTGNNGauss,
        loss_cls=ObservedGaussianNLLLoss,
        output_mode="gaussian",
        slug="swotgnn_gauss",
    ),
}
