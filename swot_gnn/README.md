# SWOT-GNN

Replication of "SWOT-based Simulation of River Discharge with Temporal Graph Neural Networks" (Osanlou et al., NeurIPS 2024).

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `torch-geometric`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`

## Structure

- `data/`: Graph construction, discharge loading, feature assembly, temporal dataset
- `models/`: SWOT-GNN, GraphGPS layer, ST-block, baselines (GPS-GNN, LSTM, drainage-area ratio)
- `training/`: Training loop, KGE evaluation
- `configs/`: Hyperparameters

## Usage

```bash
python run_training.py --grit-path /path/to/grit_reaches.csv \
  --swot-path /path/to/swot_wse.csv \
  --climate-path /path/to/climate_csv_folder \
  [--discharge-path /path/to/discharge.csv]
```

Without discharge data, the dataset will have zero gauged nodes; add discharge CSV or USGS integration for training.

## Segment-based mode

For coarser segment-level inputs (one node per river segment instead of per reach):

1. **Create segment-reach mapping** via `training_data_processing/aggregate_wse_by_segments.ipynb`. Saves `segment_mapping_df.csv` with `segment_id`, `selected_reach_id`.

2. **Generate segment datacubes**:
   ```bash
   python -m data.training_data_processing_segment_based_20260222 --segment-mapping /path/to/segment_mapping_df.csv --save-folder /path/to/output
   ```
   Use `--swot-wse`, `--mswx-folder`, `--reach-attrs`, etc. to override default paths. Set `SMART_HS_ROOT` env var if data lives elsewhere.

3. **Train with segment datacubes**:
   ```bash
   python run_training.py --grit-path /path/to/grit_reaches.csv \
     --dynamic-datacube /path/to/ba_river_swot_dynamic_datacube_wse_norm.nc \
     --static-datacube /path/to/ba_river_swot_static_datacube_wse_norm.nc \
     --segment-based --segment-mapping /path/to/segment_reach_mapping.csv
   ```
   The script saves `segment_reach_mapping.csv` in its output folder; you can also use `segment_mapping_df.csv` from the notebook.

## Data Format

- **GRIT reach CSV**: Must have `fid`, `segment_id`, `downstre_1`, `length`, `grwl_width`, `upstream_n`, `centroid_x`, `centroid_y`
- **SWOT WSE CSV**: `fid`, `date`, `wse`
- **Climate CSVs**: `{Var}_Past_Daily_combined_catchment_avg_reach_level.csv` with `time`, `fid`, `var` (for reach-level; segment-based uses mapping)
- **Discharge CSV**: `date`, `discharge`, `reach_id` (or `site_id` with gauge-to-reach mapping)
