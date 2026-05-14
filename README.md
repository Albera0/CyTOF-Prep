# CyTOF Time and Batch Correction & Visualization Pipeline

A modular Python package for CyTOF (Cytometry by Time-of-Flight) data processing, including time-drift correction, batch effect normalization, and systematic evaluation metrics.

## Project Structure

- `pipeline.py`: Command-line interface.
- `correction.py`: Core algorithms for time-drift and batch effect correction.
- `utils.py`: Shared utilities for configuration management, data transformation, and metadata handling.
- `time_correction_visualization.py`: Statistical and visual metrics for time-drift correction evaluation.
- `batch_correction_visualization.py`: Statistical and visual metrics for batch effect correction evaluation.

## Getting Started

### 1. Prerequisites
The package requires Python 3.8 or higher. Install the necessary dependencies using pip:

```powershell
pip install pandas numpy scanpy matplotlib seaborn openpyxl scipy FlowCytometryTools
```

> **Note on Data Compatibility**: If your raw FCS/CSV data lacks `Time` or `Event_length` columns, or does not have pre-assigned cell types, the pipeline will still work.
> - **Missing `Time`**: The pipeline will automatically skip the time-drift correction step but will still perform batch correction using the anchor samples.
> - **Missing `cell_type`**: You can set `celltype_gating, celltype2correct, stable_marker` to an empty list in `meta.json` or perform a preliminary clustering/gating before running the pipeline.

### 2. Configuration (`meta.json`)
The pipeline is driven by a configuration file (`meta.json`) located in the `metadata/` directory. Below is the explanation for each parameter:
- `data_dir`: The directory containing the raw input CSV files.
- `output_dir`: The root directory where processed data and diagnostic plots will be stored.
- `ref_plate`: The ID of the plate chosen as the baseline (reference) for batch alignment. All other plates will be normalized to match this plate's anchor distribution.
- `anchors`: A mapping of `dataset` names (e.g., "D0", "D24") to the `sample_name` used as the reference Anchor in that specific dataset.
- `meta_cols`: List of columns that contain metadata rather than scientific signal (e.g., Time, Event_length, panel). These are excluded from mathematical correction logic.
- `celltypes`: The complete list of cell population labels expected in the data. 
- `celltype_gating`: The specific cell types used for high-level gating analysis. (If the cell_type column already exists in the raw data, it can be ignored.)
- `celltype2correct`: The subset of biological cell types that should undergo batch correction (If the cell type is not clear, do not perform time correction for this cell type.).
- `stable_marker`: A marker (like DNA) expected to have constant intensity, used for time-drift correction.
- `true_markers`: The list of markers (proteins) included in the core analysis and visualization.

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
python -m CyTof_BatchCorrection.pipeline correction --dataset D0
```
*Applies correction algorithms and saves processed files to the output directory.*

**Stage 2: Time Analysis (Visualization and Evaluation)**
```powershell
python -m CyTof_BatchCorrection.pipeline time-analysis --dataset D0
```
*Generates PCA, density plots... to evaluate time-drift correction.*

**Stage 3: Batch Analysis (Visualization and Evaluation)**
```powershell
python -m CyTof_BatchCorrection.pipeline batch-analysis --dataset D0
```
*Generates UMAP projections, EMD scores, marker correlations... for batch effect evaluation.*

### Full Pipeline Execution
```powershell
python -m CyTof_BatchCorrection.pipeline all --dataset D0
```

## Result Analysis Guide

All analytical results are stored in `output/results/{dataset}/{panel}/`. Below is a detailed explanation of the significance of each folder and every image generated:

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
    *   **What it shows**: Two UMAPs (Before vs After) colored by **Plate**.
    *   **Significance**: "Before," cells often cluster by plate (technical bias). "After," cells from different plates should intermix, forming clusters based on biology (cell type) rather than plate.
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
        *   **Per-Plate Boxplot**: Statistical comparison of EMD reduction across all markers for each plate.
        *   **Per-Marker Boxplot**: Detailed view of how different markers' EMD values changed within a specific plate.
    *   **Interpretation**: A successful batch correction will show the "Corrected" (Blue) KDE/Boxplot shifted significantly to the **left** (lower values) compared to "Raw" (Red).

#### **Folder: Batch_MarkerCorrelation/**
*   **`scatter_plate_{plate}.png`** & **`scatter_group_{celltype}.png`**: (from `marker_corr`)
    *   **What it shows**: Scatter plots of marker-marker correlations (e.g., Correlation of CD3 vs CD4) "Before" vs "After".
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
    *   **What it shows**: Integrated benchmarking using **scIB** metrics with **paired bootstrapping** (n=20) and **statistical SIGNIFICANCE**:
        *   **Batch Removal** (Goal: Minimal technical variance): 
            *   **ASW (Batch)**: Measures how well the batches are mixed. (Lower is Better - Expect **DECREASE**)
            *   **PCR comparison**: Principal Component Regression of batch variance. (Lower is Better - Expect **DECREASE**)
            *   **Graph Connectivity**: Measures the connectivity of cells from the same biological population across batches. (Higher is Better - Expect **INCREASE**)
        *   **Bio-conservation** (Goal: Preserve biological signal): 
            *   **ASW (Cell-type)**: Measures biological separation between different cell types. (Higher is Better - Expect **STABILITY/INCREASE**)
            *   **NMI**: Normalized Mutual Information of clustering vs. ground truth. (Higher is Better - Expect **STABILITY/INCREASE**)
            *   **ARI**: Adjusted Rand Index of clustering vs. ground truth. (Higher is Better - Expect **STABILITY/INCREASE**)
    *   **Statistical Annotations**: 
        *   **Stars (`*` to `****`)**: Indicate significance based on empirical P-values from bootstrapping with **FDR (Benjamini-Hochberg)** correction.
        *   **ns**: Non-significant changes.
    *   **Significance**: Quantitative proof of batch removal effectiveness vs. biological preservation. Success is characterized by significant improvements in batch removal metrics while maintaining or improving bio-conservation scores.

---

### 3. Cross-Panel Analysis (Global Evaluation)
These visualizations are located at the `results/{dataset}/` and appear only when both **Panel 1** and **Panel 2** are processed successfully.

#### **Folder: SharedMarker_Comparison/**
Evaluates how well markers that are common to both panels have been normalized.
*   **`umap_shared_by_plate.png`**: Shows shared markers colored by Plate. "Before," cells often cluster by plate (technical bias). "After," cells from different plates should intermix, forming clusters based on biology (cell type) rather than plate.
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
