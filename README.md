# CyTOF Time and Batch Correction & Visualization Pipeline

A modular Python package for CyTOF (Cytometry by Time-of-Flight) data processing, including time-drift correction, batch effect normalization, and systematic evaluation metrics.

## Project Structure

- `src/pipeline.py`: Command-line interface.
- `src/correction.py`: Core algorithms for time-drift and batch effect correction.
- `src/utils.py`: Shared utilities for configuration management, data transformation, and metadata handling.
- `src/time_correction_visualization.py`: Statistical and visual metrics for time-drift correction evaluation.
- `src/batch_correction_visualization.py`: Statistical and visual metrics for batch effect correction evaluation.
- `examples/`: Output-cleared preparation, correction, and visualization notebooks.
- `metadata/`: Generic JSON and spreadsheet configuration examples.

## Getting Started

### 1. Prerequisites
The frozen handover environment uses Python 3.8.20. Recreate it with Conda:

```powershell
conda env create -f environment.yml
conda activate cytof-ci
```

Alternatively, in an existing Python 3.8 environment:

```powershell
pip install -r requirements.txt
```

> **Note on Data Compatibility**: If your raw FCS/CSV data lacks `Time` or `Event_length` columns, or does not have pre-assigned cell types, the pipeline will still work.
> - **Missing `Time`**: The pipeline will automatically skip the time-drift correction step but will still perform batch correction using the anchor samples.
> - **Missing cell-type labels**: Set `celltype2correct` to an empty list to skip time correction, or provide cell-type annotations/gating before running it. Keep `stable_marker` as a non-empty string even when time correction is skipped.

### 2. Configuration (`meta.json`)
The pipeline is driven by a configuration file (`meta.json`) located in the `metadata/` directory. Below is the explanation for each parameter:
- `data_dir`: The directory containing the raw input CSV files.
- `output_dir`: The root directory where processed data and diagnostic plots will be stored.
- `ref_plate`: The ID of the plate chosen as the baseline (reference) for batch alignment. All other plates will be normalized to match this plate's anchor distribution.
- `anchors`: A mapping of `dataset` names (e.g., "D0", "D24") to the `sample_name` used as the reference Anchor in that specific dataset.
- `meta_cols`: List of columns that contain metadata rather than scientific signal (e.g., Time, Event_length, panel). These are excluded from mathematical correction logic.
- `celltypes`: The complete list of cell population labels expected in the data. 
- `celltype_gating`: The specific cell types used for high-level gating analysis. (If the cell_type column already exists in the raw data, it can be ignored.)
- `celltype2correct`: The subset of biological cell types used for time-drift correction. An empty list skips time correction; anchor-based batch correction still applies at plate level.
- `stable_marker`: One reference-marker column name as a non-empty string, for example `"DNA"`. Choose the channel appropriate for your data; lists and empty strings are invalid. If time correction runs, this column must exist in the panel.
- `true_markers`: The list of markers (proteins) included in the core analysis and visualization.

Time correction uses the group mean of this single reference channel divided by its centered rolling median. The default window is 500 cells (adapted to group size and made odd), and factors are clipped to 0.3–3.0. The subsequent arcsinh cofactor of 5 and anchor-based correction are unchanged.

### 3. Metadata Mapping
To ensure visual consistency across disparate panels:
1. Provide a `metadata.xlsx` file in the `metadata/` directory with the following columns:
    - `file_name`: The exact name of the input CSV/FCS file.
    - `sample_name`: The biological identity of the sample (e.g., patient ID).
    - `panel`: The antibody panel used for the acquisition (e.g., Panel1, Panel2).
    - `plate`: The technical batch or plate ID (e.g., Plate1, Plate2).
    - `dataset`: Experimental groups that can represent each dataset. (e.g., D0 for dataset 0, D24 for dataset 24. If there is only one dataset, you can set all of them to D0.).
    - `condition`: Experimental condition (e.g., Healthy, Treatment).

## Execution via Command Line Interface (CLI)

Ensure the current working directory is the project root.

### Modular Pipeline Execution
The pipeline can be executed in distinct stages:

**Stage 1: Correction (Data Processing)**
```powershell
python -m src.pipeline correction --dataset D0 --config metadata/meta.json
```
*Applies correction algorithms and saves processed files to the output directory.*

**Stage 2: Time Analysis (Visualization and Evaluation)**
```powershell
python -m src.pipeline time-analysis --dataset D0 --config metadata/meta.json
```
*Generates PCA, density plots... to evaluate time-drift correction.*

**Stage 3: Batch Analysis (Visualization and Evaluation)**
```powershell
python -m src.pipeline batch-analysis --dataset D0 --config metadata/meta.json
```
*Generates UMAP projections, EMD scores, marker correlations... for batch effect evaluation.*

### Full Pipeline Execution
```powershell
python -m src.pipeline all --dataset D0 --config metadata/meta.json
```

## Result Analysis Guide

Batch metrics use repeated, proportional plate x cell-type subsampling **without replacement** (plate only when labels are unavailable). Missing batch/label values form an `Unassigned` category in both stratification and metric inputs. `compute_metrics` defaults to 10,000 cells, five iterations and seed 42; the batch pipeline requests 20 iterations. Before/After must contain corresponding events in the same row order, with matching metadata. Counts, shared identifiers/metadata and usable indices are checked; mismatches raise errors rather than truncating or matching cells. Without verifiable identifiers/indices, a warning states the positional-pairing assumption. Per-file correction preserves event order, but concatenation, grouped correction and independently sampled CSVs can change global order; preserve event identities when preparing evaluation inputs.

The core environment is Python-only. On Windows, Somoclu may import while its training DLL is unavailable; EMD's existing fallback then uses KMeans, so population assignments/results can differ. This is reported at runtime; dependency pins alone do not verify native DLL availability.

Pipeline diagnostics are stored under `{output_dir}/{dataset}/{panel}/`, with cross-panel summaries under `{output_dir}/{dataset}/`. Per-file correction outputs are under `{output_dir}/time_corrected/` and `{output_dir}/corrected/`. Below are the diagnostic folders and outputs:

### 1. Time-Drift Correction Evaluation (`time-analysis`)
These visualizations verify if signal drift occurring *during* the acquisition of a single sample (time-drift) have been repaired.

#### **Folder: TimeCorrection/**
*   **`{panel}_{marker}_BeforeCorrection_density.png`** & **`_AfterCorrection_density.png`**: (from `plot_time_effect_density`)
    *   **What it shows**: Hexbin density plots of Arcsinh-transformed intensity vs. Time across all plates.
    *   **Significance**: Provides a "heat" view of the cell population. In the `Before` plot, you might see the population "drift" up or down. In the `After` plot, the population should form a stable horizontal band.

#### **Folder: TimeCorrection_Alignment/**
*   **`{panel}_{marker}_PlateAlignment_TimeDrift.png`**: (from `plot_time_effect_density_merged`)
    *   **What it shows**: Hexbin density plots where the X-axis is split into sequential segments for each Plate. 
    *   **Significance**: Visualizes global alignment across the entire project. Each plate occupies its own space, allowing you to see if intensities match across technical batches and if intra-plate drift has been flattened.

#### **Folder: TimeCorrectionSignal/**
*   **`{panel}_{plate}_distribution.png`**: (from `plot_signal_distributions_joint`)
    *   **What it shows**: Overlaid Kernel Density Estimation (KDE) curves with statistical annotations:
        *   **P-adj**: FDR-corrected P-value (Mann-Whitney U) indicating if the signal shift is significant.
        *   **KD**: Kullback-Leibler divergence measuring the magnitude of the distribution change.

    *   **Significance**: Ensures that the time-drift correction didn't "squash" or distort the biological distribution of markers. The curves should look similar in shape.

#### **Folder: Time_Variance/**
*   **`kd_distribution_{dataset}_{panel}.png`**: (from `plot_kd_distribution`)
    *   **What it shows**: Bar chart of KL Divergence (KD) for each marker.
    *   **Significance**: Measures how much the distribution changed. A low KD might suggest the correction didn't "squash" or distort the biological distribution of markers.
*   **`variance_marker_boxplot_{dataset}_{panel}_Time.png`**: (from `plot_variance_marker_boxplot`)
    *   **What it shows**: Boxplot of variance explained by factors (Plate, Time, Sample, etc.) across all markers.
    *   **Significance**: Shows the distribution of technical vs. biological variance contribution for each marker. Time correction should specifically reduce the **Time** variance across markers.

#### **Folder: Time_UMAP/**
UMAP visualization of time-drift correction.
*   **`umap_comparison_{dataset}_{panel}_TimeCorrection.png`**: (from `umap_by_celltype`)
    *   **What it shows**: Side-by-side UMAP projections of "Before" vs "After" time-drift correction, colored by **Cell Type**.
    *   **Significance**: Used to confirm that biological cell clusters remain intact or become more distinct after identifying and removing technical signal drift within the acquisition duration.

#### **Folder: Time_MarkerCorrelation/**
*   **`scatter_plate_{plate}.png`**: (from `marker_corr`)
    *   **What it shows**: Scatter plot of marker-marker correlations before vs. after time-drift correction.
    *   **Significance**: Points should align along the diagonal $y=x$. This proves the time-drift correction didn't distort biological relationships between markers.

---

### 2. Batch Effect Evaluation (`batch-analysis`)
These visualizations verify if the Batch Effects between different plates have been removed using the Anchor samples.

#### **Folder: Batch_AnchorSignal/**
The raw baseline distribution of Anchor samples across plates.
*   **`{panel}_Anchor_{marker}.png`**: (from `visualize_anchor_across_plates`)
    *   **What it shows**: Overlaid distributions of the Anchor sample across all plates.
    *   **Significance**: Highlights the magnitude of the batch effect before adjustment.

#### **Folder: Batch_AnchorCorrectionSignal/**
Direct validation of the anchor-based alignment.
*   **`{panel}_{plate}_distribution.png`**: (from `plot_signal_distributions`)
    *   **What it shows**: Overlaid density curves of the **Anchor sample only**, comparing its distributions before and after batch correction.
    *   **Significance**: Since the algorithm calculates normalization factors based on the Anchor, these curves should align after correction, demonstrating that the reference point is now consistent across all plates.

#### **Folder: Batch_UMAP/**
UMAP visual proof of batch removal.
*   **`umap_plate_effect_{panel}.png`**: (from `umap_plate_effect`)
    *   **What it shows**: Two UMAPs (Before vs After) using only the shared anchor sample from plates where it is available, balanced to the same number of cells per plate and colored by **Plate**.
    *   **Significance**: Because the biological source is held constant, improved plate-color mixing after correction directly reflects reduced technical plate effects.
*   **`umap_batch_evaluation_{panel}.png`**: (from `umap_batch_evaluation`)
    *   **What it shows**: A grid showing Raw vs Corrected, colored by Plate and Cell Type.
    *   **Significance**: Confirms that while plate effects are removed (intermixing in Plate plot), biological structures are preserved (distinct clusters in Cell Type plot).
*   **`umap_celltype_by_plate_{panel}.png`**: (from `umap_celltype_by_plate`)
    *   **What it shows**: Side-by-side UMAPs for each individual plate.
    *   **Significance**: Used to check whether the celltype of each plate still clusters after batch correction.

#### **Folder: Batch_EMD/**
*   **`EMD_global_KDE.png`**, **`EMD_per_plate_boxplot.png`**, **`EMD_KDE_{plate}.png`**, **`EMD_marker_boxplot_{plate}.png`**: (from `plot_emd` and `plot_emd_per_marker`)
    *   **What it shows**: Earth Mover's Distance (EMD) distributions between pairs of plates.
    *   **Significance**: Lower EMD = higher similarity. 
        *   **Global KDE**: Overall shift in EMD distributions (Raw vs Corrected).
        *   **Per-Plate Boxplot**: Each point is one marker's mean EMD across cell populations; Before and After are compared as marker-paired values for each plate.
        *   **Per-Marker Boxplot**: Detailed row-level EMD distribution within a target plate.
    *   **Statistics**: The global two-sided Wilcoxon pairs marker x target-plate x SOM-node/population rows, without BH adjustment. The per-target-plate boxplot first averages populations for each marker, pairs those marker means, then applies BH across successfully computed target-plate tests in that call. These are different statistical units, not independent biological replicates; no paper-specific five-test family is used. `emd_results.csv` preserves the row-level inputs and `emd_p_value.txt` describes the global test. MAD score is a median absolute change, not a signed reduction.
    *   **All-zero paired differences**: When all finite Before/After pairs have zero differences, the test is reported as NaN / Not testable. Plotting continues without significance stars for that comparison. Per-plate BH includes only valid tests; untestable plates are listed as excluded in `emd_p_value.txt`. No replacement p-value is assigned.
    *   **Interpretation**: A successful batch correction will show the "Corrected" (Blue) KDE/Boxplot shifted significantly to the **left** (lower values) compared to "Raw" (Red).

#### **Folder: Batch_KDE_JSD/**

`compute_kde_jsd` compares each target plate with the reference, optionally within an anchor. The retained KDE-JSD name/CSV fields contain the minimum **base-2 Jensen–Shannon distance**, i.e. the square root of the divergence, over bounded translations of the smoothed target histogram. The existing pooled bandwidth, 512-bin grid and shift search are unchanged; the distance ranges from 0 to 1.

Finite values are sampled without replacement, independently per state/plate/marker, up to 20,000 cells with seed 42. Equal seeds do not establish cell-identity pairing. `kde_jsd_results.csv` stores distances, shifts and sample counts; `kde_jsd_statistics.csv` reports the existing unadjusted two-sided Wilcoxon over matched marker-target-plate summaries. Those summaries share references and are not independent biological replicates. Insufficient finite samples or failed density calculations yield NaN; all-invalid pairs produce statistics with no plot/test.

#### **Folder: Batch_MarkerCorrelation/**
*   **`scatter_overall.png`**, **`scatter_plate_{plate}.png`** & **`scatter_group_{celltype}.png`**: (from `marker_corr`)
    *   **What it shows**: Scatter plots of marker-marker correlations (e.g., Correlation of CD3 vs CD4) "Before" vs "After".
    *   **Overall plot**: Correlation preservation across all cells and all marker pairs.
    *   **Output**: `marker_correlation_summary.csv` includes overall, plate and requested group summaries. Pearson correlation compares the upper triangles of Before/After marker-correlation matrices; BH covers all valid summary tests in the call. Overall correlation is enabled by the batch pipeline and does not replace plate/group outputs.
    *   **Significance**: Points should fall on the diagonal red line ($y=x$), proving that the correction didn't destroy biological co-expression patterns.

#### **Folder: Variance/** (Batch Phase)
*   **`{dataset}_global_variance.png`**: (from `plot_variance_bar`)
    *   **What it shows**: Factor variance contribution (Plate, Sample, Cell Type, Time) for both panels.
    *   **Significance**: Ideally, Plate variance should decrease after batch correction.
*   **`{dataset}_celltype_variance.png`**: (from `plot_variance_bar_celltype`)
    *   **What it shows**: Mean marker variance within Cell Types.
    *   **Significance**: Validates that correction reduces technical noise within biological groups.
*   **`variance_marker_boxplot_{dataset}_{panel}_Batch.png`**: (from `plot_variance_marker_boxplot`)
    *   **What it shows**: Distribution of variance explained by factors across all individual markers.
    *   **Significance**: Allows identification of specific markers that are highly affected by batch effects (**Plate** factor). High-quality correction should minimize the Plate variance component for most markers.
*   **`{panel}_metrics_summary_plot.png`**: (from `plot_metrics_comparison`)
    *   **What it shows**: scIB metric means and standard deviations across paired repeated subsampling iterations (20 in the pipeline):
        *   **Batch Removal** (Goal: Minimal technical variance): 
            *   **ASW (Batch)**: Measures how well the batches are mixed. (Lower is Better - Expect **DECREASE**)
            *   **PCR comparison**: Principal Component Regression of batch variance. (Lower is Better - Expect **DECREASE**)
            *   **Graph Connectivity**: Measures the connectivity of cells from the same biological population across batches. (Higher is Better - Expect **INCREASE**)
        *   **Bio-conservation** (Goal: Preserve biological signal): 
            *   **ASW (Cell-type)**: Measures biological separation between different cell types. (Higher is Better - Expect **STABILITY/INCREASE**)
            *   **NMI**: Normalized Mutual Information of clustering vs. ground truth. (Higher is Better - Expect **STABILITY/INCREASE**)
            *   **ARI**: Adjusted Rand Index of clustering vs. ground truth. (Higher is Better - Expect **STABILITY/INCREASE**)
*   **`{panel}_metrics_boxplot.png`** and **`{panel}_metrics_subsampling_values.csv`**:
    *   **What they show**: An additional boxplot and its source data. Each point is one paired cell-subsampling iteration. The original mean ± SD summary plot is retained unchanged.
    *   **Statistical Annotations**: 
        *   **Stars (`*` to `****`)**: Retain the existing BH-adjusted directional empirical fractions across paired iteration differences: fraction >= 0 for ASW-batch/PCR, fraction <= 0 for the other metrics. BH covers displayed metrics in the call. Iterations overlap; these are not independent biological replicates or calibrated bootstrap/permutation p-values. Thresholds are <0.05, <0.01, <0.001 and <0.0001.
        *   **ns**: Non-significant changes.
    *   **Source data**: `{panel}_metrics_subsampling_values.csv` stores each finite metric/stage/iteration value used in the boxplot; `{panel}_metrics_summary.csv` stores means and standard deviations.

---

### 3. Cross-Panel Analysis (Global Evaluation)
These visualizations are located at the `results/{dataset}/` and appear only when both **Panel 1** and **Panel 2** are processed successfully.

#### **Folder: SharedMarker_Comparison/**
Evaluates how well markers that are common to both panels have been normalized.
*   **`umap_shared_by_plate.png`**: Uses only the shared anchor sample from plates where it is available, balances cell counts per panel/plate, and colors cells by Plate. Improved color mixing after correction directly reflects reduced technical plate effects.
*   **`umap_shared_by_celltype.png`**: Multi-panel UMAP colored by Cell Type. Essential to verify that the same cell type from different panels clusters together.
*   **`umap_shared_by_panel.png`**: Specifically colored by Panel. Ideally, the "Panel 1" and "Panel 2" populations should overlap significantly for the shared markers.

#### **Folder: SharedMarker_Correlation/**
*   **`marker_expression_correlation_plot.png`**: (from `marker_expression_correlation_plot`)
    *   **What it shows**: Medians of all markers in Panel 1 vs. Panel 2.
    *   **Significance**: Validates that the global expression profiles are consistent.

#### **Folder: Variance/** (Cross-Panel)
*   **`{dataset}_global_variance.png`**: (from `plot_variance_bar`)
    *   **What it shows**: A side-by-side bar chart of variance contribution for both panels.
    *   **Significance**: Allows for direct comparison of correction efficiency. Ideally, correction should reduce batch and time variance while preserving other variances.
*   **`{dataset}_celltype_variance.png`**: (from `plot_variance_bar_celltype`)
    *   **What it shows**: Mean marker variance broken down by Cell Type for both panels.
    *   **Significance**: Ideally, the variance within cell types should be reduced.

---
