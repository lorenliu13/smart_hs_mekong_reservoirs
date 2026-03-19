"""
Compatibility shim — feature assembler logic has been merged into
training_data_processing_lake_based_20260319.py.

Import directly from there:
    from training_data_processing_lake_based_20260319 import (
        assemble_lake_features_from_datacubes,
        ERA5_INPUT_VARS,
        ECMWF_CLIMATE_VARS,
        SWOT_DIM,
        CLIMATE_DIM,
        WSE_LAKE_DYNAMIC_INDICES,
    )
"""

from training_data_processing_lake_based_20260319 import (  # noqa: F401
    assemble_lake_features_from_datacubes,
    ERA5_INPUT_VARS,
    ECMWF_CLIMATE_VARS,
    SWOT_DIM,
    CLIMATE_DIM,
    WSE_LAKE_DYNAMIC_INDICES,
)
