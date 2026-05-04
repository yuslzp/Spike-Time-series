# Final HybridSNN Plots

This directory was assembled from completed runs under `/projects/SNN/leo_workspace/saved_stuff/HybridSNN_saved`.

## Contents

- `summary/`: generated metric summary plots and CSV files.
- `final_method/`: final HybridSNN analysis plots and selected final-epoch diagnostics from `/projects/SNN/leo_workspace/saved_stuff/HybridSNN_saved/HybridSNN_v5/apr_16_last_residual_revert`.
- `ablations/`: copied analysis plots for each ablation run from the April 20, April 22 no-residual encoder, and April 28 no-residual ablation campaigns.
- `manifest.json`: source-to-destination mapping for copied files.

## Final Method Source

The final-method plots use `HybridSNN_v5/apr_16_last_residual_revert`, which matches the method described in `methodology.md`: delta-conv encoder, TS-LIF/AOHA HybridBlocks, horizon-decoder forecasting readout, and SHD attention-pooling readout.

## Ablation Sources

- `HybridSNN_v5/apr_20_ablation_metrla_h6_seed40`
- `HybridSNN_v5/apr_22_ablation_metrla_h6_seed40_clean_encoder_noresidual`
- `HybridSNN_v5/apr_28_ablation_metrla_h6_seed40_noresidual`

## Counts

- Final runs: 5
- Ablation runs: 21
- Copied plot files: 247
- Missing expected diagnostic files: 0
