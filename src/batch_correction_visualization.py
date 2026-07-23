from glob import glob
import os
import gc
from collections import defaultdict
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
import scanpy as sc
import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import umap
import numpy as np
import somoclu
import warnings
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from anndata import ImplicitModificationWarning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import chi2, wasserstein_distance, median_abs_deviation, pearsonr, wilcoxon, mannwhitneyu, ttest_rel
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import scib
from sklearn.neighbors import NearestNeighbors

from pypalettes import load_cmap as _load_cmap

from .utils import add_metadata, enforce_schema, read_cytof_file, get_indexed_files, harmonize_marker, apply_marker_schema
from .correction import arcsinh

_BEFORE_COLOR = "#F56455"
_AFTER_COLOR  = "#15134B"
_PANEL1_COLOR = "#A1B654"
_PANEL2_COLOR = "#B8321A"

_PLATE_COLORS = [
    "#2B9B81", "#F4C40F", "#FBE183", "#D8443C", "#9B3441",
    "#DE597C", "#1F6E9C", "#E87B89", "#E6A2A6", "#AA7AA1",
    "#9F5691", "#633372", "#92C051", "#FE9B00", "#FBE183",
]


def _savefig(save_path, fig=None, **kwargs):
    """Save figure as PNG and SVG."""
    _fig = fig if fig is not None else plt.gcf()
    _fig.savefig(save_path, **kwargs)
    svg_path = os.path.splitext(save_path)[0] + '.svg'
    svg_kw = {k: v for k, v in kwargs.items() if k != 'dpi'}
    svg_kw.setdefault('bbox_inches', 'tight')
    _fig.savefig(svg_path, **svg_kw)


def _build_celltype_palette(cell_types):
    """Return {celltype: rgba} using Alexandrite, sorted alphabetically."""
    sorted_types = sorted(str(ct) for ct in cell_types)
    n = len(sorted_types)
    if n == 0:
        return {}
    cmap = _load_cmap("Alexandrite")
    return {ct: cmap(i / max(n - 1, 1)) for i, ct in enumerate(sorted_types)}


def _build_plate_palette(plates):
    """Assign _PLATE_COLORS to plates in sorted order."""
    sorted_plates = sorted(str(p) for p in plates)
    return {p: _PLATE_COLORS[i % len(_PLATE_COLORS)] for i, p in enumerate(sorted_plates)}


# Visualize the distribution of marker intensities across plates for Anchor samples only
def visualize_anchor_across_plates(
    df,
    panel,
    markers,
    save_dir,
    anchor_name,
    lower_pct=1,
    upper_pct=99,
    subsample=50000,
    true_markers=None
):
    """
    Plots the intensity distribution for markers across different plates.
    
    This visual check ensures that the anchor sample shows consistent signal intensity after correction. It helps identify
    outlier plates or markers where batch correction might have failed.

    Args:
        df (pd.DataFrame): Data containing marker intensities and plate metadata.
        panel (str): Panel identifier.
        markers (list): List of markers to plot.
        save_dir (str): Directory to save the resulting plots.
        anchor_name (str): Name of the sample used as the normalization anchor.
        lower_pct/upper_pct (int): Percentiles for x-axis scaling to avoid outlier squashing.
        subsample (int): Max number of cells to plot per marker for performance.
        true_markers (list, optional): Subset of markers to focus on.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Filter markers if true_markers provided
    if true_markers:
        markers = [m for m in markers if m in true_markers]

    # Filter for the specified panel
    df_panel = df[df["panel"] == panel].copy()

    # Keep only markers that exist in dataframe
    df_anchor = df_panel[df_panel["sample_name"] == anchor_name].copy()

    markers = [m for m in markers if m in df_anchor.columns and pd.api.types.is_numeric_dtype(df_anchor[m])]
    if len(markers) == 0:
        print("No valid markers found.")
        return

    plates = df_anchor["plate"].unique()

    sns.set_theme(style="whitegrid")

    # Calculate global x-axis limits using percentile-based range
    marker_ranges = {}
    for marker in markers:

        all_vals = df_anchor[marker].dropna()
        if len(all_vals) == 0:
            continue

        lower = np.percentile(all_vals, lower_pct)
        upper = np.percentile(all_vals, upper_pct)

        # Add buffer around percentile range
        buffer = 0.2 * (upper - lower)

        xmin = lower - buffer
        xmax = upper + buffer

        # Ensure zero is visible if data are non-negative
        if xmin > 0:
            xmin = 0

        marker_ranges[marker] = (xmin, xmax)

    for marker in markers:

        if marker not in marker_ranges:
            continue

        global_min, global_max = marker_ranges[marker]

        n_plates = len(plates)
        n_cols = 4
        n_rows = int(np.ceil(n_plates / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5*n_cols, 4*n_rows)
        )

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, plate in enumerate(plates):

            ax = axes[i]
            df_plate = df_anchor[df_anchor["plate"] == plate]
            data = df_plate[marker].dropna()

            if len(data) == 0:
                ax.axis("off")
                continue
            
            # Subsample to speed up KDE calculation
            if len(data) > subsample:
                data = data.sample(n=subsample, random_state=42)

            # Use KDE with boundary smoothing to avoid curve sticking to zero
            sns.kdeplot(
                data,
                ax=ax,
                lw=2,
                clip=marker_ranges[marker]
            )

            # Apply identical x-axis limits for fair comparison
            ax.set_xlim(global_min, global_max)

            ax.set_title(f"{plate}", fontsize=13)
            ax.set_xlabel("Intensity", fontsize=12, fontweight='bold')
            ax.set_ylabel("Density", fontsize=12, fontweight='bold')

        # Remove unused axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"{panel} - Anchor across plates\nMarker: {marker}",
            fontsize=18, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{panel}_Anchor_{marker}.png")
            _savefig(save_path, fig=fig, dpi=300)
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
            plt.close(fig)

# Split by cell type within the anchor sample, to see if certain cell types are more affected by batch effects across plates
def visualize_anchor_celltype_across_plates(
    df,
    panel,
    markers,
    save_dir,
    anchor_name,
    celltype_col="celltype",
    lower_pct=1,
    upper_pct=99,
    subsample=50000,
    true_markers=None
):
    """
    Visualizes anchor distributions across plates, stratified by cell type.
    
    This helps determine if batch effects are cell-type specific. By comparing the same
    cell population  across plates, we can see if normalization
    effectively aligns all subpopulations.

    Outputs a grid of KDE plots (rows: cell types, columns: plates).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Filter markers if true_markers provided
    if true_markers:
        markers = [m for m in markers if m in true_markers]

    # Filter panel
    df_panel = df[df["panel"] == panel].copy()

    # Filter anchor
    df_anchor = df_panel[df_panel["sample_name"] == anchor_name].copy()

    if df_anchor.empty:
        print("No anchor data found.")
        return

    markers = [m for m in markers if m in df_anchor.columns and pd.api.types.is_numeric_dtype(df_anchor[m])]
    if len(markers) == 0:
        print("No valid markers found.")
        return

    plates = df_anchor["plate"].unique()
    celltypes = df_anchor[celltype_col].unique()

    sns.set_theme(style="whitegrid")

    for marker in markers:

        # Calculate global x-axis limits using percentile-based range across all plates for this marker
        all_vals = df_anchor[marker].dropna()
        if len(all_vals) == 0:
            continue

        lower = np.percentile(all_vals, lower_pct)
        upper = np.percentile(all_vals, upper_pct)
        buffer = 0.3 * (upper - lower)

        xmin = lower - buffer
        xmax = upper + buffer
        if xmin > 0:
            xmin = 0

        # Set up grid of subplots: rows = cell types, columns = plates
        n_rows = len(celltypes)
        n_cols = len(plates)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4*n_cols, 3*n_rows),
            sharex=True,
            sharey=False
        )

        # Handle case when there's only one row or one column to ensure axes is always 2D array
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, celltype in enumerate(celltypes):
            for j, plate in enumerate(plates):

                ax = axes[i, j]

                df_sub = df_anchor[
                    (df_anchor["plate"] == plate) &
                    (df_anchor[celltype_col] == celltype)
                ]

                data = df_sub[marker].dropna()

                if len(data) == 0:
                    ax.axis("off")
                    continue
                
                # Subsample to speed up KDE calculation in grid
                if len(data) > subsample:
                    data = data.sample(n=subsample, random_state=42)

                sns.kdeplot(
                    data,
                    ax=ax,
                    lw=1.8,
                    clip=(xmin, xmax)
                )

                ax.set_xlim(xmin, xmax)

                if i == 0:
                    ax.set_title(f"{plate}", fontsize=13)

                if j == 0:
                    ax.set_ylabel(f"{celltype}\nDensity", fontsize=12, fontweight='bold')
                else:
                    ax.set_ylabel("")

                ax.set_xlabel("", fontsize=12, fontweight='bold')

        plt.suptitle(
            f"{panel} - Anchor across plates (by celltype)\nMarker: {marker}",
            fontsize=17, fontweight='bold'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{panel}_Anchor_Celltype_{marker}.png")
            _savefig(save_path, fig=fig, dpi=300)
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
            plt.close(fig)


def _balanced_paired_anchor_subset(
    df_before,
    df_after,
    anchor_name,
    sample_col="sample_name",
    group_cols=("plate",),
    max_cells=50000,
    random_state=42,
):
    """Return the same anchor-cell rows before/after, balanced across groups."""
    required = {sample_col, *group_cols}
    missing_before = required - set(df_before.columns)
    missing_after = required - set(df_after.columns)
    if missing_before or missing_after:
        raise ValueError(
            "Anchor UMAP requires matching metadata columns. "
            f"Missing before={sorted(missing_before)}, after={sorted(missing_after)}"
        )

    def _metadata_groups(df):
        return {
            tuple(str(value) for value in row)
            for row in df[list(group_cols)].drop_duplicates().itertuples(
                index=False, name=None
            )
        }

    expected_before = _metadata_groups(df_before)
    expected_after = _metadata_groups(df_after)
    if expected_before != expected_after:
        raise ValueError(
            "Before/after data do not contain the same plate/panel groups."
        )

    before = df_before[df_before[sample_col].astype(str) == str(anchor_name)].copy()
    after = df_after[df_after[sample_col].astype(str) == str(anchor_name)].copy()
    anchor_groups = _metadata_groups(before) & _metadata_groups(after)
    missing_anchor_groups = sorted(expected_before - anchor_groups)
    if missing_anchor_groups:
        print(
            f"[UMAP][WARN] Anchor sample {anchor_name!r} is unavailable in "
            f"plate/panel groups {missing_anchor_groups}; those groups are excluded."
        )
    # CytoNorm exports can reorder files/groups, so the global DataFrame index is
    # not a stable cell identifier. Pair rows by their position *within* each
    # metadata group; conversion/correction preserves the event order per file.
    before_groups = before.groupby(list(group_cols), observed=True, sort=True)
    after_groups = after.groupby(list(group_cols), observed=True, sort=True)
    group_keys = sorted(set(before_groups.groups) & set(after_groups.groups), key=str)
    if not group_keys:
        raise ValueError(f"No plate groups found for anchor sample {anchor_name!r}.")

    rng = np.random.default_rng(random_state)
    paired_sizes = [
        min(len(before_groups.get_group(key)), len(after_groups.get_group(key)))
        for key in group_keys
    ]
    n_per_group = int(min(paired_sizes))
    if max_cells is not None:
        n_per_group = min(n_per_group, max(1, int(max_cells) // len(group_keys)))
    before_selected = []
    after_selected = []
    for key in group_keys:
        before_group = before_groups.get_group(key).reset_index(drop=True)
        after_group = after_groups.get_group(key).reset_index(drop=True)
        paired_size = min(len(before_group), len(after_group))
        chosen_positions = rng.choice(paired_size, size=n_per_group, replace=False)
        before_selected.append(before_group.iloc[chosen_positions])
        after_selected.append(after_group.iloc[chosen_positions])

    before = pd.concat(before_selected, ignore_index=True)
    after = pd.concat(after_selected, ignore_index=True)
    print(
        f"[UMAP] Anchor-only: {anchor_name!r} | groups={len(group_keys)} | "
        f"cells/group={n_per_group} | cells/state={len(before)}"
    )
    return before, after


# UMAP visualization focused on plate effects, showing how samples cluster by plate before and after correction
def umap_plate_effect(
    df_before,
    df_after,
    markers,
    plate_col="plate",
    celltype_col="celltypes",
    anchor_name=None,
    sample_col="sample_name",
    n_pcs=20,
    n_neighbors=30,
    min_dist=0.3,
    subsample=50000,
    size=5,
    alpha=0.6,
    random_state=42,
    save_path=None
):
    """
    Standardizes 'Before' and 'After' data to plot them in the same UMAP space focused on batches.

    When anchor_name is provided, only paired cells from that shared anchor sample
    are used, with equal maximum sampling per plate.
    
    If 'Before' correction cells cluster strongly by 'Plate', it indicates a high batch effect.
    The 'After' plot should show better mixing of colors (plates) if correction was successful.
    """
    warnings.filterwarnings("ignore", category=UserWarning, message=".*No data for colormapping provided.*")
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

    rng = np.random.default_rng(random_state)
    markers = [m for m in markers if m in df_before.columns and m in df_after.columns and pd.api.types.is_numeric_dtype(df_before[m]) and pd.api.types.is_numeric_dtype(df_after[m])]

    if anchor_name is not None:
        df_before, df_after = _balanced_paired_anchor_subset(
            df_before,
            df_after,
            anchor_name=anchor_name,
            sample_col=sample_col,
            group_cols=(plate_col,),
            max_cells=subsample,
            random_state=random_state,
        )
    # Subsample for speed when no anchor was requested.
    elif df_before.shape[0] > subsample:
        idx_before = rng.choice(df_before.shape[0], subsample, replace=False)
        df_before = df_before.iloc[idx_before].reset_index(drop=True)
    if anchor_name is None and df_after.shape[0] > subsample:
        idx_after = rng.choice(df_after.shape[0], subsample, replace=False)
        df_after = df_after.iloc[idx_after].reset_index(drop=True)

    # Marker matrices. UMAP markers must be genuinely observed in the selected
    # panel/anchor subset; silently replacing a panel-mismatched all-NaN marker
    # with zero would hide an upstream marker-selection error.
    X_before = df_before[markers].to_numpy(dtype=float)
    X_after = df_after[markers].to_numpy(dtype=float)
    invalid_before = ~np.isfinite(X_before)
    invalid_after = ~np.isfinite(X_after)
    if invalid_before.any() or invalid_after.any():
        bad_markers = [
            marker for i, marker in enumerate(markers)
            if invalid_before[:, i].any() or invalid_after[:, i].any()
        ]
        raise ValueError(
            "UMAP markers contain missing or non-finite values in the selected "
            f"panel/anchor subset: {bad_markers}"
        )

    # AnnData
    adata_before = ad.AnnData(X_before)
    adata_before.obs[plate_col] = df_before[plate_col].astype(str).values
    adata_before.obs["status"] = "Before"

    adata_after = ad.AnnData(X_after)
    adata_after.obs[plate_col] = df_after[plate_col].astype(str).values
    adata_after.obs["status"] = "After"

    do_celltype_plot = (
        celltype_col in df_before.columns
        and celltype_col in df_after.columns
        and not df_before[celltype_col].isna().all()
        and not df_after[celltype_col].isna().all()
    )
    if do_celltype_plot:
        adata_before.obs[celltype_col] = df_before[celltype_col].fillna("unknown").astype(str).values
        adata_after.obs[celltype_col] = df_after[celltype_col].fillna("unknown").astype(str).values

    # Preprocessing + PCA on BEFORE (to get PCs)
    sc.pp.scale(adata_before, max_value=10)
    sc.tl.pca(adata_before, n_comps=min(n_pcs, len(markers)-1), svd_solver="arpack")
    
    # Preprocessing + PCA project AFTER using BEFORE's scale and PCs
    # Manually scale after using before's mean/std
    before_values = df_before[markers].astype(float)
    after_values = df_after[markers].astype(float)
    mean_b = before_values.mean()
    std_b = before_values.std().replace([np.inf, -np.inf, 0], 1.0).fillna(1.0)
    X_after_scaled = (after_values - mean_b) / std_b
    X_after_scaled = np.nan_to_num(
        X_after_scaled.to_numpy(dtype=float), nan=0.0, posinf=10.0, neginf=-10.0
    )
    X_after_scaled = np.clip(X_after_scaled, -10, 10) # Match sc.pp.scale(max_value=10)
    
    # Project AFTER onto BEFORE's PCA space
    adata_after.obsm["X_pca"] = X_after_scaled @ adata_before.varm["PCs"]

    # Compute UMAP based on BEFORE's neighbors
    sc.pp.neighbors(adata_before, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_before, min_dist=min_dist, random_state=random_state)
    
    # To project AFTER into the same UMAP space, we need the fitted UMAP model
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    adata_before.obsm["X_umap"] = reducer.fit_transform(adata_before.obsm["X_pca"])
    adata_after.obsm["X_umap"] = reducer.transform(adata_after.obsm["X_pca"])

    # Plot
    # Set plate colors
    all_plates = sorted(set(adata_before.obs[plate_col].unique()) | set(adata_after.obs[plate_col].unique()))
    plate_palette = _build_plate_palette(all_plates)
    for _adata in (adata_before, adata_after):
        _plates = sorted(str(p) for p in _adata.obs[plate_col].unique())
        _adata.uns[f"{plate_col}_colors"] = [plate_palette[p] for p in _plates]

    if do_celltype_plot:
        all_celltypes = sorted(
            set(adata_before.obs[celltype_col].astype(str).unique())
            | set(adata_after.obs[celltype_col].astype(str).unique())
        )
        celltype_palette = _build_celltype_palette(all_celltypes)
        for _adata in (adata_before, adata_after):
            _celltypes = sorted(str(c) for c in _adata.obs[celltype_col].unique())
            _adata.uns[f"{celltype_col}_colors"] = [
                celltype_palette.get(c, "#808080") for c in _celltypes
            ]

    n_cols = 2 if do_celltype_plot else 1
    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 12), squeeze=False)
    sc.pl.umap(adata_before, color=plate_col, ax=axes[0, 0], show=False, size=size, alpha=alpha)
    axes[0, 0].set_title("Before - Plate", fontsize=16, fontweight='bold')
    sc.pl.umap(adata_after, color=plate_col, ax=axes[1, 0], show=False, size=size, alpha=alpha)
    axes[1, 0].set_title("After - Plate", fontsize=16, fontweight='bold')
    if do_celltype_plot:
        sc.pl.umap(adata_before, color=celltype_col, ax=axes[0, 1], show=False, size=size, alpha=alpha)
        axes[0, 1].set_title("Before - Cell type", fontsize=16, fontweight='bold')
        sc.pl.umap(adata_after, color=celltype_col, ax=axes[1, 1], show=False, size=size, alpha=alpha)
        axes[1, 1].set_title("After - Cell type", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return adata_before, adata_after

# UMAP visualization showing joint projection of raw and corrected data, colored by plate, cell type, and age group
def umap_batch_evaluation(
    df_raw,
    df_corrected,
    markers,
    plate_col="plate",
    celltype_col="celltypes",
    subsample=50000,
    random_state=42,
    save_path=None,
    true_markers=None
):
    """
    Joint UMAP evaluation to compare Raw vs Corrected states side-by-side.
    
    Generates a 2x2 grid:
    1. Raw data colored by Plate.
    2. Raw data colored by Cell Type.
    3. Corrected data colored by Plate.
    4. Corrected data colored by Cell Type.
    """
    warnings.filterwarnings("ignore", category=UserWarning, message=".*No data for colormapping provided.*")
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

    # Keep only markers that are in true_markers if provided
    if true_markers:
        markers = [m for m in markers if m in true_markers]

    markers = [m for m in markers if m in df_raw.columns and m in df_corrected.columns and pd.api.types.is_numeric_dtype(df_raw[m]) and pd.api.types.is_numeric_dtype(df_corrected[m])]

    # same cells
    n_cells = min(subsample, df_raw.shape[0], df_corrected.shape[0])
    # Avoid errors if indices don't match or n_cells is 0
    if n_cells == 0:
        print("[WARN] No common cells or insufficient data for UMAP subsampling.")
        return None

    idx_raw = np.random.RandomState(random_state).choice(df_raw.index, n_cells, replace=False)
    idx_corr = np.random.RandomState(random_state).choice(df_corrected.index, n_cells, replace=False)

    df_raw_sub = df_raw.loc[idx_raw].copy()
    df_corr_sub = df_corrected.loc[idx_corr].copy()

    df_raw_sub["status"] = "raw"
    df_corr_sub["status"] = "corrected"

    df_all = pd.concat([df_raw_sub, df_corr_sub], ignore_index=True)

    X = df_all[markers].fillna(0).values.astype(np.float32)

    # SAFE Cell Type Processing for UMAP
    obs_df = df_all[[plate_col, "status"]].copy()
    
    # Identify if a valid cell type column exists
    actual_ct_col = celltype_col if celltype_col in df_all.columns else ('celltype' if 'celltype' in df_all.columns else None)
    
    do_celltype_plot = False
    if actual_ct_col and not df_all[actual_ct_col].isnull().all():
        obs_df[celltype_col] = df_all[actual_ct_col].fillna("unknown").astype("category")
        do_celltype_plot = True
    else:
        # Avoid KeyError in AnnData creation
        obs_df[celltype_col] = "N/A"

    adata = ad.AnnData(
        X,
        obs=obs_df
    )

    # PCA + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, random_state=random_state)

    # Harmonized cell-type colors propagated to slices
    if do_celltype_plot and celltype_col in adata.obs.columns:
        unique_types = sorted(str(ct) for ct in adata.obs[celltype_col].unique())
        ct_palette = _build_celltype_palette(unique_types)
        adata.uns[f"{celltype_col}_colors"] = [ct_palette.get(ct, '#808080') for ct in unique_types]

    # Set plate colors
    all_plates = sorted(str(p) for p in adata.obs[plate_col].unique())
    plate_palette = _build_plate_palette(all_plates)
    adata.uns[f"{plate_col}_colors"] = [plate_palette[p] for p in all_plates]

    # split raw / corrected
    adata_raw = adata[adata.obs["status"] == "raw"].copy()
    adata_corr = adata[adata.obs["status"] == "corrected"].copy()

    # Propagate plate colors to sub-adatas
    for sub in (adata_raw, adata_corr):
        _plates = sorted(str(p) for p in sub.obs[plate_col].unique())
        sub.uns[f"{plate_col}_colors"] = [plate_palette[p] for p in _plates]

    # Apply consistent colors to each sub-adata
    if do_celltype_plot and celltype_col in adata.uns:
        ct_palette = _build_celltype_palette(
            sorted(str(ct) for ct in adata.obs[celltype_col].unique()))
        for sub in (adata_raw, adata_corr):
            cats = sorted(str(ct) for ct in sub.obs[celltype_col].unique())
            sub.uns[f"{celltype_col}_colors"] = [ct_palette.get(ct, '#808080') for ct in cats]

    # Width: 1 row per Batch (Raw/Corr), column per Plate view and Celltype view
    n_cols = 2 if do_celltype_plot else 1
    
    # If n_cols is 1, subplots(2, 1) returns a 1D array of 2 axes.
    fig, axes = plt.subplots(2, n_cols, figsize=(7.5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    sc.pl.umap(adata_raw, color=plate_col, ax=axes[0,0], show=False)
    axes[0,0].set_title("Before - Plate", fontsize=15, fontweight='bold')

    if do_celltype_plot:
        sc.pl.umap(adata_raw, color=celltype_col, ax=axes[0,1], show=False)
        axes[0,1].set_title("Before - Cell type", fontsize=15, fontweight='bold')

    sc.pl.umap(adata_corr, color=plate_col, ax=axes[1,0], show=False)
    axes[1,0].set_title("After - Plate", fontsize=15, fontweight='bold')

    if do_celltype_plot:
        sc.pl.umap(adata_corr, color=celltype_col, ax=axes[1,1], show=False)
        axes[1,1].set_title("After - Cell type", fontsize=15, fontweight='bold')

    plt.tight_layout()
    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return adata


# UMAP visualization showing raw and corrected data side by side for each plate, colored by cell type
def umap_celltype_by_plate(
    df_raw,
    df_corrected,
    markers,
    plate_col="plate",
    celltype_col="celltypes",
    subsample=50000,
    random_state=42,
    save_path=None
):
    """
    Detailed plate-by-plate UMAP breakdown.
    
    Each column represents a Plate. Each row is either 'Raw' or 'Corrected'.
    Spotting if plates have mixed after correction.
    """
    warnings.filterwarnings("ignore", category=UserWarning, message=".*No data for colormapping provided.*")
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

    # SKIP if cell_type is missing or empty
    actual_celltype_col = celltype_col if celltype_col in df_raw.columns else ('celltypes' if 'celltypes' in df_raw.columns else None)
    if actual_celltype_col is None or df_raw[actual_celltype_col].isnull().all():
        print(f"[SKIP] umap_celltype_by_plate: '{celltype_col}' column is missing or empty.")
        return None

    # keep shared markers
    markers = [m for m in markers if m in df_raw.columns and m in df_corrected.columns and pd.api.types.is_numeric_dtype(df_raw[m]) and pd.api.types.is_numeric_dtype(df_corrected[m])]

    # Sample each DataFrame independently; after panel filtering they may have
    # disjoint index values so we cannot use the same idx on both.
    n_cells = min(subsample, df_raw.shape[0], df_corrected.shape[0])
    df_raw_sub = df_raw.sample(n=n_cells, random_state=random_state).copy()
    df_corr_sub = df_corrected.sample(n=n_cells, random_state=random_state).copy()

    df_raw_sub["status"] = "raw"
    df_corr_sub["status"] = "corrected"

    df_all = pd.concat([df_raw_sub, df_corr_sub], ignore_index=True)

    X = df_all[markers].fillna(0).values.astype(np.float32)

    adata = ad.AnnData(
        X,
        obs=df_all[[plate_col, celltype_col, "status"]].copy()
    )

    # PCA + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, random_state=random_state)

    # Harmonized cell-type colors: build once from all categories
    unique_types = sorted(str(ct) for ct in adata.obs[celltype_col].unique())
    ct_palette = _build_celltype_palette(unique_types)
    adata.uns[f"{celltype_col}_colors"] = [ct_palette.get(ct, '#808080') for ct in unique_types]

    plates = sorted(adata.obs[plate_col].unique())

    # Plate colors
    plate_palette = _build_plate_palette(plates)
    adata.uns[f"{plate_col}_colors"] = [plate_palette[str(p)] for p in plates]

    adata_raw = adata[adata.obs["status"] == "raw"].copy()
    adata_corr = adata[adata.obs["status"] == "corrected"].copy()

    # Derive save directory and stem for per-plate saves
    if save_path:
        save_dir_plates = os.path.dirname(save_path)
        save_stem = os.path.splitext(os.path.basename(save_path))[0]
    else:
        save_dir_plates = None
        save_stem = None

    fig, axes = plt.subplots(2, len(plates), figsize=(5*len(plates), 10))
    if len(plates) == 1:
        axes = axes.reshape(2, 1)

    for i, p in enumerate(plates):
        # Sub-adatas for this plate; set consistent colors explicitly
        sub_raw = adata_raw[adata_raw.obs[plate_col] == p].copy()
        sub_corr = adata_corr[adata_corr.obs[plate_col] == p].copy()
        for sub in (sub_raw, sub_corr):
            cats = sorted(str(ct) for ct in sub.obs[celltype_col].unique())
            sub.uns[f"{celltype_col}_colors"] = [ct_palette.get(ct, '#808080') for ct in cats]

        # Before
        sc.pl.umap(sub_raw, color=celltype_col, ax=axes[0, i], show=False)
        axes[0, i].set_title(f"Before - {p}", fontsize=15, fontweight='bold')
        axes[0, i].set_xlabel("UMAP1", fontsize=12, fontweight='bold')
        axes[0, i].set_ylabel("UMAP2", fontsize=12, fontweight='bold')

        # After
        sc.pl.umap(sub_corr, color=celltype_col, ax=axes[1, i], show=False)
        axes[1, i].set_title(f"After - {p}", fontsize=15, fontweight='bold')
        axes[1, i].set_xlabel("UMAP1", fontsize=12, fontweight='bold')
        axes[1, i].set_ylabel("UMAP2", fontsize=12, fontweight='bold')

        # Save individual per-plate figure
        if save_dir_plates:
            fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 6))
            for sub, ax_p, label in [(sub_raw, axes_p[0], "Before"), (sub_corr, axes_p[1], "After")]:
                sc.pl.umap(sub, color=celltype_col, ax=ax_p, show=False)
                ax_p.set_title(f"{label} - {p}", fontsize=15, fontweight='bold')
                ax_p.set_xlabel("UMAP1", fontsize=12, fontweight='bold')
                ax_p.set_ylabel("UMAP2", fontsize=12, fontweight='bold')
            fig_p.suptitle(f"Cell Types - {p}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            safe_p = str(p).replace("/", "_").replace(" ", "_")
            _savefig(os.path.join(save_dir_plates, f"{save_stem}_{safe_p}.png"), fig=fig_p, dpi=300, bbox_inches='tight')
            plt.close(fig_p)

    plt.tight_layout()
    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return adata


# Compute EMD between raw and normalized distributions for each marker, plate, and optionally cell type
def compute_emd_mad(
    df_raw,
    df_corr,
    markers,
    ref_plate,
    plate_col="plate",
    cluster_col=None,
    anchor=None,
    grid_rows=8,
    grid_cols=8,
    n_sample=30000,
    random_state=42,
    save_dir=None
):
    """
    Computes Earth Mover's Distance (EMD) and Median Absolute Deviation (MAD) to quantify batch effects.
    Uses SOM (Self-Organizing Map) for population-stratified comparison between plates.

    Args:
        df_raw (pd.DataFrame): Raw data before correction.
        df_corr (pd.DataFrame): Data after batch correction.
        markers (list): List of biological markers to evaluate.
        ref_plate: Plate used as the correction reference. Every other plate is
            compared directly with this plate; non-reference plates are not
            compared with one another.
        plate_col (str): Column name for technical plate/batch identifier.
        cluster_col (str, optional): Pre-existing cluster column. If None, SOM will be trained.
        anchor (str, optional): If provided, calculations only use cells from this specific sample.
        grid_rows/cols (int): SOM grid dimensions for population stratification.
        n_sample (int): Max number of cells to sample per cluster-marker pair.
        save_dir (str, optional): Path to save raw EMD results as CSV.

    Returns:
        tuple: (emd_df, mad_df, mad_score) containing detailed and summarized metrics.
    """
    markers = [m for m in markers if m in df_raw.columns and m in df_corr.columns and pd.api.types.is_numeric_dtype(df_raw[m]) and pd.api.types.is_numeric_dtype(df_corr[m])]
    rng = np.random.default_rng(random_state)

    # Optional: filter anchor sample
    if anchor is not None:
        df_raw = df_raw[df_raw["sample_name"] == anchor]
        df_corr = df_corr[df_corr["sample_name"] == anchor]

    df_raw_nodes = df_raw.copy()
    df_corr_nodes = df_corr.copy()

    # Determine clustering method: External labels (cell_type) or SOM
    if cluster_col is not None and cluster_col in df_raw.columns:
        print(f"  [INFO] Using provided cluster column '{cluster_col}' for EMD calculation.")
        df_raw_nodes["EMD_Cluster"] = df_raw[cluster_col]
        df_corr_nodes["EMD_Cluster"] = df_corr[cluster_col]
        clusters = sorted(df_raw_nodes["EMD_Cluster"].dropna().unique())
    else:
        print(f"  [INFO] No cluster column provided or found. Training SOM ({grid_rows}x{grid_cols}) for population stratification.")
        X_corr = df_corr[markers].fillna(0).values
        X_raw  = df_raw[markers].fillna(0).values
        scaler = StandardScaler()
        X_corr_scaled = scaler.fit_transform(X_corr)
        X_raw_scaled  = scaler.transform(X_raw)

        try:
            som = somoclu.Somoclu(
                n_rows=grid_rows,
                n_columns=grid_cols,
                maptype="planar",
                initialization="pca"
            )
            som.train(X_corr_scaled)
            bmus_corr = som.bmus
            node_corr = bmus_corr[:,0] * grid_cols + bmus_corr[:,1]
            bmus_raw  = som.get_bmus(X_raw_scaled)
            node_raw  = bmus_raw[:,0] * grid_cols + bmus_raw[:,1]
        except Exception as som_err:
            print(f"  [WARN] SOM failed ({som_err}). Falling back to KMeans clustering.")
            from sklearn.cluster import KMeans
            n_clusters = grid_rows * grid_cols
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            node_corr = km.fit_predict(X_corr_scaled)
            node_raw  = km.predict(X_raw_scaled)

        df_corr_nodes["EMD_Cluster"] = node_corr
        df_raw_nodes["EMD_Cluster"]  = node_raw
        clusters = sorted(df_corr_nodes["EMD_Cluster"].unique())

    plates = list(df_raw_nodes[plate_col].dropna().unique())
    corr_plates = set(df_corr_nodes[plate_col].dropna().unique())
    if ref_plate not in plates or ref_plate not in corr_plates:
        raise ValueError(
            f"Reference plate {ref_plate!r} must be present in both raw and corrected "
            f"data after anchor filtering. Raw plates: {plates}; corrected plates: {sorted(corr_plates, key=str)}"
        )
    comparison_plates = [plate for plate in plates if plate != ref_plate and plate in corr_plates]
    if not comparison_plates:
        raise ValueError(f"No non-reference plates are available to compare with {ref_plate!r}.")

    # Reference-based EMD: compare every target plate directly with ref_plate.
    emd_results = []
    for plate2 in comparison_plates:
        plate1 = ref_plate
        raw1 = df_raw_nodes[df_raw_nodes[plate_col] == plate1]
        raw2 = df_raw_nodes[df_raw_nodes[plate_col] == plate2]
        corr1 = df_corr_nodes[df_corr_nodes[plate_col] == plate1]
        corr2 = df_corr_nodes[df_corr_nodes[plate_col] == plate2]

        for cluster in clusters:
            raw_node1 = raw1[raw1["EMD_Cluster"] == cluster]
            raw_node2 = raw2[raw2["EMD_Cluster"] == cluster]
            corr_node1 = corr1[corr1["EMD_Cluster"] == cluster]
            corr_node2 = corr2[corr2["EMD_Cluster"] == cluster]

            if raw_node1.shape[0] < 10 or raw_node2.shape[0] < 10:
                continue
            if corr_node1.shape[0] < 10 or corr_node2.shape[0] < 10:
                continue

            for marker in markers:
                x_raw = rng.choice(raw_node1[marker].dropna().values,
                                   min(len(raw_node1), n_sample), replace=False)
                y_raw = rng.choice(raw_node2[marker].dropna().values,
                                   min(len(raw_node2), n_sample), replace=False)
                x_corr = rng.choice(corr_node1[marker].dropna().values,
                                    min(len(corr_node1), n_sample), replace=False)
                y_corr = rng.choice(corr_node2[marker].dropna().values,
                                    min(len(corr_node2), n_sample), replace=False)

                emd_raw = wasserstein_distance(x_raw, y_raw)
                emd_corr = wasserstein_distance(x_corr, y_corr)
                emd_reduction = emd_raw - emd_corr

                emd_results.append({
                    "plate1": plate1,
                    "plate2": plate2,
                    "SOM_node": cluster,
                    "marker": marker,
                    "EMD_raw": emd_raw,
                    "EMD_corr": emd_corr,
                    "EMD_reduction": emd_reduction
                })

    emd_df = pd.DataFrame(emd_results)

    # Save EMD results to CSV for cross-method comparison
    if save_dir is not None:
        emd_csv_path = os.path.join(save_dir, "emd_results.csv")
        emd_df.to_csv(emd_csv_path, index=False)
        print(f"[SAVE] EMD results saved to: {emd_csv_path}")

    # MAD per batch
    mad_results = []
    for plate in plates:
        df_raw_plate = df_raw_nodes[df_raw_nodes[plate_col] == plate]
        df_corr_plate = df_corr_nodes[df_corr_nodes[plate_col] == plate]

        for cluster in clusters:
            raw_node = df_raw_plate[df_raw_plate["EMD_Cluster"] == cluster]
            corr_node = df_corr_plate[df_corr_plate["EMD_Cluster"] == cluster]

            if raw_node.shape[0] == 0 or corr_node.shape[0] == 0:
                continue

            for marker in markers:
                x = raw_node[marker].dropna().values
                y = corr_node[marker].dropna().values
                if len(x) == 0 or len(y) == 0:
                    continue
                x = rng.choice(x, min(len(x), n_sample), replace=False)
                y = rng.choice(y, min(len(y), n_sample), replace=False)

                mad_raw = median_abs_deviation(x)
                mad_corr = median_abs_deviation(y)
                mad_diff_abs = abs(mad_corr - mad_raw)

                mad_results.append({
                    "plate": plate,
                    "SOM_node": cluster,
                    "marker": marker,
                    "MAD_raw": mad_raw,
                    "MAD_corr": mad_corr,
                    "MAD_diff_abs": mad_diff_abs
                })

    mad_df = pd.DataFrame(mad_results)

    # Compute MAD score per batch (median of absolute differences)
    mad_score = mad_df.groupby("plate")["MAD_diff_abs"].median().reset_index()
    mad_score = mad_score.rename(columns={"MAD_diff_abs": "MAD_score"})

    return emd_df, mad_df, mad_score


# Plot density/violin style EMD per SOM node per marker in a paper-style format
def plot_emd(emd_df, mad_score, save_path=None):
    """
    Main EMD visualization suite.
    
    1. Global KDE Plot: Shows the overall shift in distribution (Raw vs Corrected).
    2. Boxplots per target plate: EMD to the reference plate for all markers.
    3. Per-Plate KDE: Detailed look at individual batch reduction efficiency.
    """

    # Create directory if needed
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    raw_color = _BEFORE_COLOR
    corr_color = _AFTER_COLOR

    # GLOBAL KDE PLOT
    emd_raw_mean = emd_df["EMD_raw"].mean()
    emd_corr_mean = emd_df["EMD_corr"].mean()
    emd_reduction_mean = emd_df["EMD_reduction"].mean()
    mad_median = mad_score["MAD_score"].median() if not mad_score.empty else np.nan

    # Calculate p-value (Wilcoxon Signed-Rank Test) - Paired comparison    
    stat, p_val = wilcoxon(emd_df["EMD_raw"], emd_df["EMD_corr"], alternative='greater')
    
    # Save p-value to a text file for separate reporting
    if save_path is not None:
        p_val_path = os.path.join(save_path, "emd_p_value.txt")
        with open(p_val_path, "w") as f:
            f.write(f"Wilcoxon Signed-Rank Test (EMD Before vs After)\n")
            f.write(f"Alternative hypothesis: Before > After\n")
            f.write(f"Statistic: {stat:.4f}\n")
            f.write(f"P-Value: {p_val:.2e}\n")
        print(f"P-value saved to: {p_val_path}")

    plt.figure(figsize=(7,5))
    sns.kdeplot(emd_df["EMD_raw"], fill=True, alpha=0.4, linewidth=2, label="Before", color=raw_color)
    sns.kdeplot(emd_df["EMD_corr"], fill=True, alpha=0.4, linewidth=2, label="After", color=corr_color)

    textstr = (
        f"Mean EMD reduction = {emd_reduction_mean:.3f}\n"
        f"Median MAD reduction = {mad_median:.3f}"
    )
    plt.text(0.98, 0.05, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.xlabel("Earth Mover's Distance (EMD)", fontsize=13, fontweight='bold')
    plt.ylabel("Density", fontsize=13, fontweight='bold')
    plt.title("Global Batch Effect Reduction (KDE)", fontsize=16, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11, prop={'weight':'bold'})
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        _savefig(os.path.join(save_path, "EMD_global_KDE.png"), dpi=300)
    else:
        plt.show()
    plt.close()

    # PER-TARGET-PLATE BOXPLOT. Each EMD row is ref_plate -> target plate,
    # so assign it only to plate2 instead of duplicating it under both plates.
    emd_long = []
    for _, row in emd_df.iterrows():
        emd_long.append({'Plate': row['plate2'], 'EMD': row['EMD_raw'], 'Condition': 'Before', 'Marker': row['marker']})
        emd_long.append({'Plate': row['plate2'], 'EMD': row['EMD_corr'], 'Condition': 'After', 'Marker': row['marker']})
    
    plot_data = pd.DataFrame(emd_long)
    plates_sorted = sorted(plot_data['Plate'].unique(), key=lambda x: str(x))
    
    raw_color_v = _BEFORE_COLOR
    corr_color_v = _AFTER_COLOR
    single_target = len(plates_sorted) == 1
    figure_size = (6.8, 6.2) if single_target else (max(10, len(plates_sorted) * 2.5), 7)
    fig, ax = plt.subplots(figsize=figure_size)
    sns.set_theme(style="white", font_scale=1.3)

    sns.boxplot(
        data=plot_data,
        x='Plate',
        y='EMD',
        hue='Condition',
        palette={'Before': raw_color_v, 'After': corr_color_v},
        width=0.30,
        linewidth=1.2,
        showfliers=False,
        order=plates_sorted,
        hue_order=['Before', 'After'],
        boxprops=dict(edgecolor="#4D5656"),
        whiskerprops=dict(color="#4D5656"),
        capprops=dict(color="#4D5656"),
        medianprops=dict(color="#2C3E50", linewidth=1.5),
        ax=ax,
    )

    max_points_per_box = 500
    strip_parts = []
    for plate in plot_data['Plate'].unique():
        for condition in plot_data['Condition'].unique():
            subset = plot_data[(plot_data['Plate'] == plate) & (plot_data['Condition'] == condition)]
            if len(subset) > max_points_per_box:
                subset = subset.sample(n=max_points_per_box, random_state=42)
            strip_parts.append(subset)
    df_stripplot = pd.concat(strip_parts, ignore_index=True)

    # Match point clouds to the narrowed hue boxes. Seaborn's stripplot uses
    # its own fixed dodge span, which would leave points outside width=0.30.
    rng = np.random.default_rng(42)
    condition_offsets = {'Before': -0.075, 'After': 0.075}
    condition_colors = {'Before': raw_color_v, 'After': corr_color_v}
    for plate_idx, plate in enumerate(plates_sorted):
        for condition in ['Before', 'After']:
            values = df_stripplot.loc[
                (df_stripplot['Plate'] == plate)
                & (df_stripplot['Condition'] == condition),
                'EMD',
            ].to_numpy()
            x = (
                plate_idx
                + condition_offsets[condition]
                + rng.uniform(-0.055, 0.055, size=len(values))
            )
            ax.scatter(
                x, values, s=12, alpha=0.4,
                facecolor=condition_colors[condition],
                edgecolor='black', linewidth=0.35, zorder=3,
            )

    # Statistical annotation for each plate using FDR (BH) correction
    y_max_total = plot_data['EMD'].max()
    p_values = []
    comparison_indices = []
    for i, plate in enumerate(plates_sorted):
        raw_vals = plot_data[(plot_data['Plate'] == plate) & (plot_data['Condition'] == 'Before')]['EMD'].values
        corr_vals = plot_data[(plot_data['Plate'] == plate) & (plot_data['Condition'] == 'After')]['EMD'].values
        
        if len(raw_vals) > 0 and len(corr_vals) > 0:
            try:
                if len(raw_vals) == len(corr_vals):
                    _, p = wilcoxon(raw_vals, corr_vals)
                else:
                    _, p = mannwhitneyu(raw_vals, corr_vals)
                p_values.append(p)
                comparison_indices.append(i)
            except Exception:
                pass

    if p_values:
        _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
        for idx, i in enumerate(comparison_indices):
            p_val = p_adj[idx]
            plate = plates_sorted[i]
            
            stars = "ns"
            if p_val < 0.0001: stars = "****"
            elif p_val < 0.001: stars = "***"
            elif p_val < 0.01: stars = "**"
            elif p_val < 0.05: stars = "*"
            
            local_max = plot_data[plot_data['Plate'] == plate]['EMD'].max()
            y_pos = local_max + (y_max_total * 0.05)
            h = y_max_total * 0.02
            
            ax.plot([i - 0.15, i - 0.15, i + 0.15, i + 0.15], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.2, c='#2C3E50')
            ax.text(i, y_pos + h, stars, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    reference_plate = emd_df['plate1'].iloc[0]
    if single_target:
        ax.set_title(
            f"EMD Distribution: {plates_sorted[0]} vs {reference_plate}",
            fontsize=17, fontweight='bold', pad=18,
        )
    else:
        ax.set_title(
            f"EMD to Reference Plate ({reference_plate}) per Target Plate",
            fontsize=18, fontweight='bold', pad=20,
        )
    ax.tick_params(axis='x', labelrotation=0 if single_target else 45, labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.set_ylabel("Earth Mover's Distance (EMD)", fontsize=16, fontweight='bold', labelpad=10)
    ax.set_xlabel("Target Plate", fontsize=16, fontweight='bold', labelpad=10)
    sns.despine(ax=ax)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='grey')
    ax.legend(title="Condition", frameon=True, facecolor='white',
              bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=14, title_fontsize=15, prop={'weight':'bold'})
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_title(), fontweight='bold')
    fig.tight_layout()
    if save_path is not None:
        _savefig(os.path.join(save_path, "EMD_per_plate_boxplot.png"), fig=fig, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    # PER-PLATE KDE PLOTS
    unique_plates = emd_df["plate2"].unique()
    for plate in unique_plates:
        df_plate = emd_df[emd_df["plate2"] == plate]
        if df_plate.empty: continue

        emd_raw_mean = df_plate["EMD_raw"].mean()
        emd_corr_mean = df_plate["EMD_corr"].mean()
        emd_reduction_mean = df_plate["EMD_reduction"].mean()
        mad_plate = mad_score[mad_score["plate"] == plate]
        mad_plate_median = mad_plate["MAD_score"].median() if not mad_plate.empty else np.nan

        plt.figure(figsize=(7,5))
        sns.kdeplot(df_plate["EMD_raw"], fill=True, alpha=0.4, linewidth=2, label="Before", color=raw_color)
        sns.kdeplot(df_plate["EMD_corr"], fill=True, alpha=0.4, linewidth=2, label="After", color=corr_color)

        textstr = (
            f"Plate = {plate}\n"
            f"Mean EMD reduction = {emd_reduction_mean:.3f}\n"
            f"Median MAD reduction = {mad_plate_median:.3f}"
        )
        plt.text(0.98, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment="bottom", horizontalalignment="right",
                    fontweight='bold',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.xlabel("Earth Mover's Distance (EMD)", fontsize=13, fontweight='bold')
        plt.ylabel("Density", fontsize=13, fontweight='bold')
        reference_plate = df_plate["plate1"].iloc[0]
        plt.title(f"Batch Effect Reduction - {plate} vs {reference_plate}", fontsize=16, fontweight='bold')
        plt.legend(loc="upper right", fontsize=11, prop={'weight':'bold'})
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if save_path is not None:
            _savefig(os.path.join(save_path, f"EMD_KDE_{plate}.png"), dpi=300)
        else:
            plt.show()
        plt.close()


def plot_emd_per_marker(emd_df, save_path=None):
    """
    Plots reference-based EMD per marker for each target plate.
    """
    if emd_df.empty:
        print("[WARN] No EMD data to plot.")
        return

    # Create directory if needed
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    raw_color = _BEFORE_COLOR
    corr_color = _AFTER_COLOR

    # plate1 is always the reference; group results only by the target (plate2).
    emd_combined = emd_df[['plate2', 'marker', 'EMD_raw', 'EMD_corr']].rename(columns={'plate2': 'Plate'})

    # Melt for seaborn: Plate, marker, Condition (Before/After), EMD
    emd_melted = emd_combined.melt(id_vars=['Plate', 'marker'], 
                                    value_vars=['EMD_raw', 'EMD_corr'],
                                    var_name='Condition', value_name='EMD')
    
    # Map Condition names for cleaner labels
    emd_melted['Condition'] = emd_melted['Condition'].map({'EMD_raw': 'Before', 'EMD_corr': 'After'})

    # Getunique plates and ensure they are sorted for consistent order (numerical if possible)
    try:
        unique_plates = sorted(emd_melted['Plate'].unique(), key=lambda x: (int(str(x).replace('Plate', '').strip()) if any(char.isdigit() for char in str(x)) else str(x)))
    except:
        unique_plates = sorted(emd_melted['Plate'].unique(), key=lambda x: str(x))

    for plate in unique_plates:
        plate_data = emd_melted[emd_melted['Plate'] == plate]
        
        plt.figure(figsize=(8, 6))
        
        # Create boxplot: x is Condition (Before/After), y is EMD value
        # Each point in the boxplot will represent a marker
        sns.boxplot(
            data=plate_data, 
            x='Condition', 
            y='EMD', 
            hue='Condition',
            palette={'Before': raw_color, 'After': corr_color},
            width=0.4,
            showfliers=False,
            legend=False
        )

        # Add points (markers) on top of the boxes
        # To avoid over-plotting when markers are many, subsample points
        max_markers_to_plot = 500
        if len(plate_data) > max_markers_to_plot:
            # Sample from each condition to maintain balance
            plate_data_sampled = pd.concat([
                plate_data[plate_data['Condition'] == 'Before'].sample(n=min(len(plate_data)//2, max_markers_to_plot//2), random_state=42),
                plate_data[plate_data['Condition'] == 'After'].sample(n=min(len(plate_data)//2, max_markers_to_plot//2), random_state=42)
            ])
        else:
            plate_data_sampled = plate_data

        sns.stripplot(
            data=plate_data_sampled,
            x='Condition',
            y='EMD',
            color='black',
            alpha=0.4,
            jitter=0.1,
            size=4
        )

        # Statistical annotation (Wilcoxon Signed-Rank Test for Before vs After across markers)
        before_vals = plate_data[plate_data['Condition'] == 'Before']['EMD'].values
        after_vals = plate_data[plate_data['Condition'] == 'After']['EMD'].values

        if len(before_vals) > 0 and len(after_vals) > 0 and len(before_vals) == len(after_vals):
            try:
                _, p = wilcoxon(before_vals, after_vals)
                
                stars = "ns"
                if p < 0.0001: stars = "****"
                elif p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                elif p < 0.05: stars = "*"

                y_max = plate_data['EMD'].max()
                h = y_max * 0.05
                plt.plot([0, 0, 1, 1], [y_max + h, y_max + 2*h, y_max + 2*h, y_max + h], lw=1.5, c='black')
                plt.text(0.5, y_max + 2*h, stars, ha='center', va='bottom', color='black', fontweight='bold')
            except Exception as e:
                print(f"[WARN] Wilcoxon failed for boxplot plate {plate}: {e}")

        reference_plate = emd_df["plate1"].iloc[0]
        plt.title(
            f"EMD Distribution across Markers - {plate} vs {reference_plate}",
            fontsize=16,
            fontweight='bold'
        )
        plt.xlabel("Condition", fontsize=14, fontweight='bold')
        plt.ylabel("Earth Mover's Distance", fontsize=14, fontweight='bold')
        plt.xticks(fontsize=13, fontweight='bold')
        plt.yticks(fontsize=13, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            safe_plate_name = str(plate).replace("/", "_").replace("\\", "_")
            _savefig(os.path.join(save_path, f"EMD_marker_boxplot_{safe_plate_name}.png"), dpi=300)
        else:
            plt.show()
        plt.close()


def variation_marker(df, markers, group_col, info_cols, panel=None, subsample=100000):
    """
    Calculates variance explained by metadata factors for EACH marker separately.
    Returns a long-form DataFrame with columns: [group, marker, factor, variance_explained].
    """
    
    # Subset to selected panel
    df_panel = df[df['panel'] == panel].dropna(axis=1, how="all").copy()

    # Keep only markers that exist in dataframe
    markers = [m for m in markers if m in df_panel.columns and pd.api.types.is_numeric_dtype(df_panel[m])]

    all_res = []

    # Loop over each condition separately
    if group_col is not None and group_col in df_panel.columns:
        grouped = df_panel.groupby(group_col)
    else:
        grouped = [("all", df_panel)]

    for group, df_group in grouped:
        if len(df_group) > subsample:
            df_group = df_group.sample(n=subsample, random_state=42)
            print(f"  [INFO] Subsampling {group} to {subsample} cells for marker variance estimation.")

        # Ensure numeric
        df_group[markers] = df_group[markers].fillna(0).astype(float)
        
        # Identify columns with variation and no NaNs to avoid ANOVA failure
        active_cols = []
        for col in info_cols:
            if col in df_group.columns:
                # Drop rows with NaN in the current factor for ANOVA stability
                if df_group[col].notna().any() and df_group[col].nunique() > 1:
                    # Specific check for celltypes: skip if only "unknown" or "Unassigned"
                    if col == 'celltypes' or col == 'celltype':
                        unique_vals = [v for v in df_group[col].dropna().unique() if str(v).lower() not in ['unknown', 'unassigned', 'n/a']]
                        if len(unique_vals) < 2:
                            continue
                    active_cols.append(col)
        
        if not active_cols:
            print(f"  [WARN] No valid metadata factors found for group {group}. Skipping marker-level variance.")
            continue

        # Re-filter df_group to rows where all active_cols are non-NaN
        df_clean = df_group.dropna(subset=active_cols)
        if df_clean.empty:
            continue

        n_cells = len(df_clean)
        for marker in markers:
            # Build regression formula: Marker ~ factors
            formula = f'Q("{marker}") ~ ' + " + ".join([f'Q("{col}")' for col in active_cols])

            try:
                # Fit linear model
                model = ols(formula, data=df_clean).fit()
                # Type II ANOVA (more robust for unbalanced designs)
                anova_res = sm.stats.anova_lm(model, typ=2)

                total_ss = anova_res["sum_sq"].sum()
                # Compute proportion of variance explained
                var_exp_per_factor = anova_res["sum_sq"] / total_ss

                for factor_raw, val in var_exp_per_factor.items():
                    # Clean the factor name from Q("...")
                    factor = factor_raw
                    if factor_raw.startswith('Q("') and factor_raw.endswith('")'):
                        factor = factor_raw[3:-2]

                    if factor == 'Residual':
                        continue

                    abs_ss = anova_res.loc[factor_raw, "sum_sq"] / n_cells  # SS per cell, comparable across runs
                    all_res.append({
                        'group': group,
                        'marker': marker,
                        'factor': factor,
                        'variance_explained': val,       # proportion (R²)
                        'sum_sq_per_cell': abs_ss,       # absolute variance, comparable before/after
                        'n_cells': n_cells,
                    })
            except Exception as e:
                pass

    return pd.DataFrame(all_res)


# Plot the distribution of variance explained by metadata factors across markers
def plot_variance_marker_boxplot(var_marker_raw, var_marker_corr, plate_col='plate', save_path=None, use_log=False):
    """
    Plots the distribution of variance explained by metadata factors across markers.
    Compares Before (Raw) and After (Corrected) for each factor.
    """
    if var_marker_raw is None or var_marker_corr is None or (var_marker_raw.empty and var_marker_corr.empty):
        print("[WARN] No marker variance data to plot.")
        return

    # Style setup
    sns.set_theme(style="white", font_scale=1.1)
    raw_color = _BEFORE_COLOR
    corr_color = _AFTER_COLOR

    # Add Stage column
    var_marker_raw = var_marker_raw.copy()
    var_marker_corr = var_marker_corr.copy()
    var_marker_raw['Stage'] = 'Before'
    var_marker_corr['Stage'] = 'After'

    # Combine
    plot_data = pd.concat([var_marker_raw, var_marker_corr], ignore_index=True)

    if plot_data.empty:
        print("[WARN] Combined plot_data is empty. Skipping plot.")
        return

    # Use log-scale for variance if requested
    y_col = 'variance_explained'
    if use_log:
        # Use log10(x + eps) to avoid issues with zero/small values
        eps = 1e-6
        # Ensure variance_explained is numeric before applying log
        plot_data['variance_explained'] = pd.to_numeric(plot_data['variance_explained'], errors='coerce')
        plot_data['neg_log_variance'] = -np.log10(plot_data['variance_explained'].fillna(0) + eps)
        y_col = 'neg_log_variance'

    # Dynamically get factors available in the data
    available_factors = plot_data['factor'].unique()
    
    # Priority order for factors
    priority_order = ['plate', 'sample_name', 'celltypes', 'celltype', 'Time', 'condition', 'dataset']
    factors = sorted(available_factors, key=lambda x: priority_order.index(x) if x in priority_order else 99)
    
    # Setup Figure (Dynamic width based on factor count)
    plt.figure(figsize=(max(10, len(factors)*3.0), 8))
    
    # Boxplot Comparing Before and After per factor
    ax = sns.boxplot(
        data=plot_data,
        x='factor',
        y=y_col,
        hue='Stage',
        palette={'Before': raw_color, 'After': corr_color},
        width=0.6,
        linewidth=1.5,
        showfliers=False,
        order=factors,
        boxprops=dict(edgecolor="#4D5656"),
        whiskerprops=dict(color="#4D5656"),
        capprops=dict(color="#4D5656"),
        medianprops=dict(color="#2C3E50", linewidth=2.0)
    )

    # Add stripplot (markers) with jitter and transparency
    sns.stripplot(
        data=plot_data,
        x='factor',
        y=y_col,
        hue='Stage',
        dodge=True,
        jitter=0.1,
        size=4.5,
        alpha=0.5,
        linewidth=0.6,
        edgecolor='black',
        order=factors,
        palette={'Before': _BEFORE_COLOR, 'After': _AFTER_COLOR},
        legend=False
    )

    if use_log:
        ax.set_ylabel('-Log10(Variance Explained)', fontsize=15, fontweight='bold')
    else:
        ax.set_ylabel('Proportion of Variance Explained', fontsize=15, fontweight='bold')
    
    ax.set_title('Variance Explained by Metadata Factors across Markers', fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('Biological / Batch Factors', fontsize=16, fontweight='bold', labelpad=15)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    y_max_total = plot_data[y_col].max()
    p_values = []
    comparison_indices = []
    for i, factor in enumerate(factors):
        # Stats & Stars Logic
        mask_before = (plot_data['factor'] == factor) & (plot_data['Stage'] == 'Before')
        mask_after = (plot_data['factor'] == factor) & (plot_data['Stage'] == 'After')
        
        # Match values by marker for paired testing (much more sensitive)
        data_b = plot_data[mask_before].set_index('marker')
        data_a = plot_data[mask_after].set_index('marker')
        common_markers = data_b.index.intersection(data_a.index)
        
        b_vals = pd.to_numeric(data_b.loc[common_markers, 'variance_explained'], errors='coerce').values
        a_vals = pd.to_numeric(data_a.loc[common_markers, 'variance_explained'], errors='coerce').values
        
        if len(b_vals) > 5:
            try:
                # Wilcoxon Signed-Rank Test handles paired Before/After data much better than Mann-Whitney
                _, p = wilcoxon(b_vals, a_vals, alternative='two-sided')
                p_values.append(p)
                comparison_indices.append(i)
                print(f"[DEBUG STATS] Factor: {factor} | Wilcoxon p={p:.2e}")
            except Exception:
                # Fallback to Mann-Whitney if Wilcoxon is not applicable
                _, p = mannwhitneyu(b_vals, a_vals, alternative='two-sided')
                p_values.append(p)
                comparison_indices.append(i)
                print(f"[DEBUG STATS] Factor: {factor} | Fallback MWU p={p:.2e}")
        elif len(b_vals) > 1:
            _, p = mannwhitneyu(b_vals, a_vals, alternative='two-sided')
            p_values.append(p)
            comparison_indices.append(i)
            print(f"[DEBUG STATS] Factor: {factor} | Small N MWU p={p:.2e}")

    if p_values:
        _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
        for idx, i in enumerate(comparison_indices):
            p_val = p_adj[idx]
            factor = factors[i]
            
            stars = "ns"
            if p_val < 0.0001: stars = "****"
            elif p_val < 0.001: stars = "***"
            elif p_val < 0.01: stars = "**"
            elif p_val < 0.05: stars = "*"

            # Draw brackets and stars
            local_max = plot_data[plot_data['factor'] == factor][y_col].max()
            if use_log:
                y_pos = local_max + 0.5
                h = 0.2
            else:
                y_pos = local_max + (y_max_total * 0.08)
                h = y_max_total * 0.02
            
            plt.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2], [y_pos, y_pos + h, y_pos + h, y_pos], lw=1.5, c='#566573')
            plt.text(i, y_pos + h, stars, ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='grey')

    # Customize legend
    leg = plt.legend(title="Stage", frameon=True, facecolor='white', loc='upper left',
                     bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=13)
    leg.set_title("Stage", prop={'size': 14, 'weight': 'bold'})
    for text in leg.get_texts():
        text.set_fontweight('bold')

    # Add footnote explaining the proportional-vs-absolute issue for Time
    ax.annotate(
        "* Proportional R² for Time may increase after batch correction (plate variance\n"
        "  removed → smaller denominator). See 'Absolute Time Variance' panel below.",
        xy=(0, -0.18), xycoords='axes fraction',
        fontsize=9, color='#7F8C8D', style='italic'
    )

    try:
        plt.tight_layout(rect=[0, 0.05, 0.9, 1])
    except Exception:
        pass

    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marker-level variance boxplot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

    # ── Second panel: Absolute Time variance (SS/cell) ─────────────────────────
    # This is the ground truth for whether time correction actually worked.
    # If SS/cell for Time decreases, correction is effective regardless of proportion change.
    has_abs = 'sum_sq_per_cell' in plot_data.columns
    time_data = plot_data[plot_data['factor'] == 'Time'] if has_abs and 'Time' in plot_data['factor'].values else None
    if time_data is not None and not time_data.empty and has_abs:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        sns.boxplot(
            data=time_data, x='Stage', y='sum_sq_per_cell',
            palette={'Before': raw_color, 'After': corr_color},
            order=['Before', 'After'], width=0.5,
            linewidth=1.5, showfliers=False,
            boxprops=dict(edgecolor="#4D5656"),
            whiskerprops=dict(color="#4D5656"),
            capprops=dict(color="#4D5656"),
            medianprops=dict(color="#2C3E50", linewidth=2.0),
            ax=ax2
        )
        sns.stripplot(
            data=time_data, x='Stage', y='sum_sq_per_cell',
            palette={'Before': raw_color, 'After': corr_color},
            order=['Before', 'After'], dodge=False, jitter=0.15,
            size=4, alpha=0.5, linewidth=0.5, edgecolor='black',
            legend=False, ax=ax2
        )
        b_vals = time_data[time_data['Stage'] == 'Before']['sum_sq_per_cell'].dropna()
        a_vals = time_data[time_data['Stage'] == 'After']['sum_sq_per_cell'].dropna()
        if len(b_vals) > 1 and len(a_vals) > 1:
            try:
                _, p = wilcoxon(b_vals.values, a_vals.values, alternative='two-sided')
            except Exception:
                _, p = mannwhitneyu(b_vals.values, a_vals.values, alternative='two-sided')
            stars = "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            y_top = max(b_vals.max(), a_vals.max())
            h = y_top * 0.05
            ax2.plot([0, 0, 1, 1], [y_top + h, y_top + 2*h, y_top + 2*h, y_top + h], lw=1.5, c='#566573')
            ax2.text(0.5, y_top + 2*h, stars, ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Absolute Time Variance (SS / n_cells)', fontsize=12, fontweight='bold')
        ax2.set_title('Absolute Time Variance Before vs After\n(↓ means time correction worked)', fontsize=13, fontweight='bold')
        ax2.set_xlabel('')
        ax2.tick_params(labelsize=12)
        sns.despine(ax=ax2)
        ax2.grid(axis='y', linestyle='--', alpha=0.3, color='grey')
        plt.tight_layout()
        if save_path:
            abs_path = os.path.splitext(save_path)[0] + '_absolute_time.png'
            _savefig(abs_path, dpi=300, bbox_inches='tight')
            print(f"Absolute time variance plot saved to: {abs_path}")
        else:
            plt.show()
        plt.close()


# Correlation of marker correlations before and after correction, to assess preservation of relationships between markers
def marker_corr(
    df_before,
    df_after,
    markers,
    plate_col='plate',
    celltype_col=None,
    group_col=None,
    save_dir=None,
):
    """
    Evaluates biological relationship preservation by comparing marker-marker correlations 
    before and after correction.

    Args:
        df_before, df_after (pd.DataFrame): Data before and after correction.
        markers (list): Markers to use for correlation matrix.
        plate_col (str): Metadata column for grouping by batch.
        celltype_col (str, optional): Metadata column for grouping by cell type.
        save_dir (str, optional): Directory to save scatter plots and summary CSV.

    Returns:
        pd.DataFrame: Summary of Pearson R values of the correlation coefficients.
    """
    markers = [m for m in markers if m in df_before.columns and m in df_after.columns and pd.api.types.is_numeric_dtype(df_before[m]) and pd.api.types.is_numeric_dtype(df_after[m])]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    summary_records = []

    # Per-plate correlation preservation by default. Passing plate_col=None
    # requests one overall marker-correlation comparison across all cells.
    if plate_col is None:
        correlation_groups = [("All cells", df_before[markers], df_after[markers], "overall")]
    else:
        correlation_groups = [
            (
                str(plate),
                df_before[df_before[plate_col] == plate][markers],
                df_after[df_after[plate_col] == plate][markers],
                "plate",
            )
            for plate in df_before[plate_col].unique()
        ]

    for plate, df_b, df_a, level in correlation_groups:

        corr_b = df_b.corr()
        corr_a = df_a.corr()

        triu_idx = np.triu_indices(len(markers), k=1)
        corr_b_vals = corr_b.values[triu_idx]
        corr_a_vals = corr_a.values[triu_idx]

        R, p_val = pearsonr(corr_b_vals, corr_a_vals)
        summary_records.append({
            'level': level,
            'plate': plate if level == 'plate' else None,
            'celltype': None,
            'group': None,
            'R': R,
            'p_value': p_val,
        })

        # scatter plot
        plt.figure(figsize=(5,5))
        plt.scatter(corr_b_vals, corr_a_vals, alpha=0.5, s=15, color=_AFTER_COLOR, label='All pairs')
        plt.plot([-1,1], [-1,1], 'r--', lw=1)
        plt.xlabel('Correlation Before', fontsize=12, fontweight='bold')
        plt.ylabel('Correlation After', fontsize=12, fontweight='bold')
        title_prefix = f'Plate {plate}' if level == 'plate' else 'Overall marker'
        plt.title(f'{title_prefix} correlation preservation\nR={R:.2f}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        if save_dir:
            save_name = f"scatter_plate_{plate}.png" if level == 'plate' else "scatter_overall.png"
            _savefig(os.path.join(save_dir, save_name), dpi=300)
        else:
            plt.show()
        plt.close()

    # per group correlation preservation
    split_levels = []
    if celltype_col:
        split_levels.append(celltype_col)
    if group_col:
        split_levels.append(group_col)
    
    if split_levels:
        groups = df_before.groupby(split_levels)
        for grp_keys, df_b_grp in groups:
            # subset df_after to the same group
            df_a_grp = df_after
            for col, key in zip(split_levels, grp_keys if isinstance(grp_keys, tuple) else [grp_keys]):
                df_a_grp = df_a_grp[df_a_grp[col]==key]
            
            if df_b_grp.empty or df_a_grp.empty:
                continue

            corr_b = df_b_grp[markers].corr()
            corr_a = df_a_grp[markers].corr()
            triu_idx = np.triu_indices(len(markers), k=1)
            corr_b_vals = corr_b.values[triu_idx]
            corr_a_vals = corr_a.values[triu_idx]

            R, p_val = pearsonr(corr_b_vals, corr_a_vals)

            # save summary
            record = {'level':'group','plate':None}
            
            # Extract grp_keys correctly: handle both single value and tuple cases
            current_keys = grp_keys if isinstance(grp_keys, (tuple, list)) else [grp_keys]
            
            if celltype_col:
                record['celltype'] = current_keys[0]
            else:
                record['celltype'] = None
            if group_col:
                record['group'] = current_keys[-1]
            else:
                record['group'] = None
            record['R'] = R
            record['p_value'] = p_val
            summary_records.append(record)

            # scatter plot
            plt.figure(figsize=(5,5))
            plt.scatter(corr_b_vals, corr_a_vals, alpha=0.5, s=15, color=_AFTER_COLOR, label='All pairs')
            plt.plot([-1,1], [-1,1],'r--', lw=1)
            plt.xlabel('Correlation Before', fontsize=12, fontweight='bold')
            plt.ylabel('Correlation After', fontsize=12, fontweight='bold')
            title = "Marker correlation preservation"
            if celltype_col:
                title += f'\nCellType: {record["celltype"]}'
            if group_col:
                if celltype_col: title += " | "
                else: title += "\n"
                
                curr_group = str(record["group"])
                if len(curr_group) <= 2:
                    title += f'Group ({group_col}): {curr_group}'
                else:
                    title += f'Group: {curr_group}'

            title += f'\nR={R:.2f}'
            plt.title(title)
            plt.legend(loc='lower right')
            plt.tight_layout()
            if save_dir:
                # Sanitize filename for Windows/Filesystem safety
                safe_celltype = str(record['celltype']).replace("/", "_").replace("\\", "_").replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace(",", "")
                safe_group = str(record['group']).replace("/", "_").replace("\\", "_").replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace(",", "")
                
                # Dynamic filename based on available levels
                if celltype_col and group_col:
                    save_name = f"scatter_group_{safe_celltype}_{safe_group}.png"
                elif celltype_col:
                    save_name = f"scatter_celltype_{safe_celltype}.png"
                elif group_col:
                    save_name = f"scatter_group_{safe_group}.png"
                else:
                    save_name = f"scatter_all_combined.png"
                
                _savefig(os.path.join(save_dir, save_name), dpi=300)
            else:
                plt.show()
            plt.close()

    summary_df = pd.DataFrame(summary_records)

    # FDR correction (Benjamini-Hochberg) across all plate + group tests
    if not summary_df.empty and "p_value" in summary_df.columns:
        valid = summary_df["p_value"].notna()
        p_adj = np.full(len(summary_df), np.nan)
        if valid.any():
            _, p_adj_vals, _, _ = multipletests(summary_df.loc[valid, "p_value"].values, method="fdr_bh")
            p_adj[valid.values] = p_adj_vals
        summary_df["p_value_adj"] = p_adj

    # Save marker correlation results to CSV
    if save_dir:
        csv_path = os.path.join(save_dir, "marker_correlation_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"[SAVE] Marker correlation summary saved to: {csv_path}")

    return summary_df


# UMAP visualization of shared markers
def visualize_shared_marker_correction(
    df_raw_P1, df_corrected_P1,
    df_raw_P2=None, df_corrected_P2=None,
    markers_P1=None, markers_P2=None,
    plate_col="plate",
    celltype_col="celltypes",
    panel_col="panel",
    anchor_name=None,
    sample_col="sample_name",
    subsample=50000,
    random_state=42,
    save_dir=None
):
    """
    Visualizes the effectiveness of batch correction on shared markers across multiple panels.
    Generates UMAPs colored by Plate, Cell Type, and Panel to look for mixing and biological separation.

    Args:
        df_raw_P1, df_corrected_P1 (pd.DataFrame): Data from Panel 1.
        df_raw_P2, df_corrected_P2 (pd.DataFrame, optional): Data from Panel 2.
        markers_P1, markers_P2 (list): Marker lists for each panel to find intersection.
        subsample (int): Max cells to include in the global UMAP.
        save_dir (str, optional): Directory to save generated UMAP plots.
    """
    # Filter the specific annoying warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*No data for colormapping provided.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs value 1 overridden.*")
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

    has_p2 = (df_raw_P2 is not None and df_corrected_P2 is not None)
    
    # Keep only markers present in available panels and numeric
    if has_p2:
        markers = [m for m in (set(markers_P1) & set(markers_P2)) 
                if m in df_raw_P1.columns and pd.api.types.is_numeric_dtype(df_raw_P1[m])
                and m in df_raw_P2.columns and pd.api.types.is_numeric_dtype(df_raw_P2[m])]
    else:
        markers = [m for m in markers_P1 if m in df_raw_P1.columns and pd.api.types.is_numeric_dtype(df_raw_P1[m])]

    if len(markers) == 0:
        print("[WARN] No shared markers found for visualization.")
        return

    print("Markers for visualization:", markers)

    # Tag rows before concatenating panels so the same anchor cells can be selected
    # in Before and After data.
    def _tag_panel_rows(df, panel_name):
        tagged = df.copy().reset_index(drop=True)
        tagged[panel_col] = panel_name
        tagged.index = pd.Index(
            [f"{panel_name}:{i}" for i in range(len(tagged))],
            name="__paired_cell_id",
        )
        return tagged

    # Combine raw and corrected data
    dfs_before = [_tag_panel_rows(df_raw_P1, "Panel1")]
    dfs_after = [_tag_panel_rows(df_corrected_P1, "Panel1")]
    
    if has_p2:
        dfs_before.append(_tag_panel_rows(df_raw_P2, "Panel2"))
        dfs_after.append(_tag_panel_rows(df_corrected_P2, "Panel2"))

    # Downsample BEFORE concatenation to save memory
    # Total cells across all panels
    n_before_total = sum(len(d) for d in dfs_before)
    n_after_total = sum(len(d) for d in dfs_after)
    target_cells = min(subsample, n_before_total, n_after_total)
    
    # Stratified sampling from each DataFrame to maintain panel balance
    def subsample_list(df_list, total_target):
        total_len = sum(len(d) for d in df_list)
        subsampled_dfs = []
        for d in df_list:
            # Proportionally sample from each panel
            n_sub = int((len(d) / total_len) * total_target)
            if n_sub > 0:
                subsampled_dfs.append(d.sample(n=n_sub, random_state=random_state).copy())
        return pd.concat(subsampled_dfs, ignore_index=True)

    df_before_sub = subsample_list(dfs_before, target_cells)
    df_after_sub = subsample_list(dfs_after, target_cells)
    
    # Cast markers to float32 to further save memory
    for m in markers:
        if m in df_before_sub.columns:
            df_before_sub[m] = df_before_sub[m].astype('float32')
        if m in df_after_sub.columns:
            df_after_sub[m] = df_after_sub[m].astype('float32')

    df_before_sub["status"] = "Before"
    df_after_sub["status"] = "After"

    df_all = pd.concat([df_before_sub, df_after_sub], ignore_index=True)
    df_all = df_all.reset_index(drop=True)
    X = df_all[markers].fillna(0).values.astype(np.float32)
    
    # DYNAMIC COLUMN SELECTION
    obs_cols = [plate_col, "panel", "status"]
    if celltype_col in df_all.columns:
        obs_cols.append(celltype_col)
    
    adata = ad.AnnData(X, obs=df_all[obs_cols].copy())

    # PCA + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, random_state=random_state)

    # Set panel colors
    unique_panels = sorted(adata.obs[panel_col].unique())
    _panel_color_map = {"Panel1": _PANEL1_COLOR, "Panel2": _PANEL2_COLOR}
    adata.uns[f"{panel_col}_colors"] = [_panel_color_map.get(p, '#808080') for p in unique_panels]

    # Set plate colors
    all_plates = sorted(str(p) for p in adata.obs[plate_col].unique())
    plate_palette = _build_plate_palette(all_plates)
    adata.uns[f"{plate_col}_colors"] = [plate_palette[p] for p in all_plates]

    # For the plate-colored diagnostic, use only the shared anchor sample and
    # balance cells across panel/plate groups. The all-sample embedding above is
    # retained for cell-type and panel-colored biological-preservation plots.
    adata_plate = adata
    if anchor_name is not None:
        anchor_before_full = pd.concat(dfs_before, axis=0)
        anchor_after_full = pd.concat(dfs_after, axis=0)
        anchor_before, anchor_after = _balanced_paired_anchor_subset(
            anchor_before_full,
            anchor_after_full,
            anchor_name=anchor_name,
            sample_col=sample_col,
            group_cols=(panel_col, plate_col),
            max_cells=subsample,
            random_state=random_state,
        )
        anchor_before["status"] = "Before"
        anchor_after["status"] = "After"
        anchor_all = pd.concat([anchor_before, anchor_after], ignore_index=True)
        anchor_obs_cols = [plate_col, panel_col, "status"]
        adata_plate = ad.AnnData(
            anchor_all[markers].fillna(0).to_numpy(dtype=np.float32),
            obs=anchor_all[anchor_obs_cols].copy(),
        )
        sc.pp.scale(adata_plate, max_value=10)
        sc.tl.pca(adata_plate, svd_solver="arpack")
        sc.pp.neighbors(adata_plate)
        sc.tl.umap(adata_plate, random_state=random_state)
        anchor_plates = sorted(str(p) for p in adata_plate.obs[plate_col].unique())
        anchor_plate_palette = _build_plate_palette(anchor_plates)
        adata_plate.uns[f"{plate_col}_colors"] = [
            anchor_plate_palette[p] for p in anchor_plates
        ]

    # Layout per panel, colored by Plate
    panels_to_plot = ["Panel1"]
    if has_p2: panels_to_plot.append("Panel2")
    
    fig, axes = plt.subplots(2, len(panels_to_plot), figsize=(6 * len(panels_to_plot), 10), squeeze=False)
    for i, status in enumerate(["Before", "After"]):
        for j, panel in enumerate(panels_to_plot):
            ax = axes[i, j]
            mask = (adata_plate.obs["status"] == status) & (adata_plate.obs[panel_col] == panel)
            sub = adata_plate[mask]
            if sub.n_obs == 0:
                continue
            sc.pl.umap(sub, color=plate_col, ax=ax, show=False, size=6, frameon=False)
            ax.set_title(f"{status} - {panel} (Plate)", fontsize=15, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            # Adjust tick labels if they appear but sc.pl.umap frameon=False usually hides them
            ax.xaxis.label.set_size(12)
            ax.xaxis.label.set_weight('bold')
            ax.yaxis.label.set_size(12)
            ax.yaxis.label.set_weight('bold')

    plt.tight_layout()
    plt.suptitle(
        "UMAP of Marker Correction - Plate Coloring",
        fontsize=18,
        fontweight='bold',
        y=1.02,
    )
    if save_dir:
        _savefig(os.path.join(save_dir, "umap_shared_by_plate.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # layout per panel, colored by Cell Type
    if celltype_col in adata.obs.columns:
        # Apply Alexandrite palette to cell types
        unique_types = sorted(str(ct) for ct in adata.obs[celltype_col].dropna().unique())
        ct_palette = _build_celltype_palette(unique_types)
        adata.uns[f"{celltype_col}_colors"] = [ct_palette.get(ct, '#808080') for ct in unique_types]

        fig, axes = plt.subplots(2, len(panels_to_plot), figsize=(6 * len(panels_to_plot), 10), squeeze=False)
        for i, status in enumerate(["Before", "After"]):
            for j, panel in enumerate(panels_to_plot):
                ax = axes[i, j]
                mask = (adata.obs["status"] == status) & (adata.obs["panel"] == panel)
                sub = adata[mask]
                if sub.n_obs == 0:
                    continue
                sc.pl.umap(sub, color=celltype_col, ax=ax, show=False, size=6, frameon=False)
                ax.set_title(f"{status} - {panel} (Cell Type)", fontsize=15, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.xaxis.label.set_size(12)
                ax.xaxis.label.set_weight('bold')
                ax.yaxis.label.set_size(12)
                ax.yaxis.label.set_weight('bold')
        plt.tight_layout()
        plt.suptitle("UMAP of Marker Correction - Cell Type Coloring", fontsize=18, fontweight='bold', y=1.02)
        if save_dir:
            _savefig(os.path.join(save_dir, "umap_shared_by_celltype.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Combined Panels, separate UMAPs for Before and After, color by panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, status in enumerate(["Before", "After"]):
        ax = axes[i]
        mask = adata.obs["status"] == status
        sub = adata[mask]
        if sub.n_obs == 0:
            continue
            
        sc.pl.umap(sub, color=panel_col, ax=ax, show=False, size=6, frameon=False)
        ax.set_title(f"{status} - Combined Panels", fontsize=15, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.label.set_size(12)
        ax.xaxis.label.set_weight('bold')
        ax.yaxis.label.set_size(12)
        ax.yaxis.label.set_weight('bold')
    plt.tight_layout()
    plt.suptitle("UMAP of Shared Marker Correction - Panel Coloring", fontsize=18, fontweight='bold', y=1.02)
    if save_dir:
        _savefig(os.path.join(save_dir, "umap_shared_by_panel.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return adata



# Compute variance of each marker within each cell type before and after correction
def compute_celltype_variance(df_raw, df_corrected, batch_col='plate', ctx=None):
    # Detect available cell type column
    celltype_col = 'celltypes' if 'celltypes' in df_raw.columns else ('celltype' if 'celltype' in df_raw.columns else None)
    
    # SKIP if celltypes is missing or entirely empty
    if celltype_col is None:
        print(f"[SKIP] compute_celltype_variance: No cell type column found.")
        return pd.DataFrame(), pd.DataFrame()
        
    if df_raw[celltype_col].isnull().all() or (df_raw[celltype_col].nunique() <= 1 and df_raw[celltype_col].iloc[0] == 'Unassigned'):
        print(f"[SKIP] compute_celltype_variance: '{celltype_col}' column is empty or all 'Unassigned'.")
        return pd.DataFrame(), pd.DataFrame()

    # Exclude non-marker columns
    if ctx is not None:
        exclude_cols = ctx.meta_cols + ["original_filename", celltype_col]
    else:
        exclude_cols = ['plate','sample_name','dataset','Event_length','Time','Anchor','condition',celltype_col]
    
    marker_cols_raw = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and c not in exclude_cols]
    marker_cols_corr = [c for c in df_corrected.columns if pd.api.types.is_numeric_dtype(df_corrected[c]) and c not in exclude_cols]

    # Raw: compute mean per celltype per batch
    # Filter only if multiple batches exist
    if df_raw[batch_col].nunique() < 2:
            return pd.DataFrame(index=[0], columns=marker_cols_raw).fillna(0), pd.DataFrame(index=[0], columns=marker_cols_corr).fillna(0)

    grouped_raw = df_raw.groupby([celltype_col, batch_col])[marker_cols_raw].mean().reset_index()
    # Pivot: index=cell_type, columns=batch, values=marker signal
    var_marker_raw = pd.DataFrame(index=df_raw[celltype_col].unique(), columns=marker_cols_raw)
    for ct in df_raw[celltype_col].unique():
        ct_data = grouped_raw[grouped_raw[celltype_col]==ct]
        # compute variance across batches
        if ct_data.shape[0] < 2:
            var_marker_raw.loc[ct] = 0
        else:
            var_marker_raw.loc[ct] = ct_data[marker_cols_raw].var(ddof=1)

    # Corrected
    grouped_corr = df_corrected.groupby([celltype_col, batch_col])[marker_cols_corr].mean().reset_index()
    var_marker_corr = pd.DataFrame(index=df_corrected[celltype_col].unique(), columns=marker_cols_corr)
    for ct in df_corrected[celltype_col].unique():
        ct_data = grouped_corr[grouped_corr[celltype_col]==ct]
        # Check if enough data points to compute variance
        if ct_data.shape[0] < 2:
            var_marker_corr.loc[ct] = 0
        else:
            var_marker_corr.loc[ct] = ct_data[marker_cols_corr].var(ddof=1)

    if var_marker_raw.sum().sum() == 0 and var_marker_corr.sum().sum() == 0:
        return pd.DataFrame(), pd.DataFrame()

    return var_marker_raw.astype(float), var_marker_corr.astype(float)


def compute_metrics(df_after, markers, df_before, batch_key='plate', label_key='celltypes', subsample=10000, n_iterations=5, random_state=42):
    """
    Compute batch correction metrics using a PAIRED bootstrapping approach.
    For each iteration, the SAME cell indices are sampled from before and after data.
    """

    # 1. Fallback for label key
    actual_label_key = label_key if label_key in df_after.columns else ('celltypes' if 'celltypes' in df_after.columns else None)
    
    # Check if we have valid labels
    has_labels = False
    if actual_label_key:
        valid_labels = df_after[actual_label_key].dropna()
        unique_vals = valid_labels.unique()
        if len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] != 'Unassigned'):
            has_labels = True

    # Ensure df_before and df_after have the same index for paired sampling
    if len(df_before) != len(df_after):
        min_len = min(len(df_before), len(df_after))
        df_before = df_before.iloc[:min_len].copy()
        df_after = df_after.iloc[:min_len].copy()
    
    # Limit to numeric markers
    markers = [m for m in markers if m in df_after.columns and pd.api.types.is_numeric_dtype(df_after[m])]
    
    results_before = []
    results_after = []
    
    # Default values for bio-conservation
    results_before = []
    results_after = []
    
    # Sample the same INDICES for both dataframes to ensure PAIRED testing
    common_indices = np.arange(len(df_after))
    
    print(f"  [METRICS] Running {n_iterations} paired bootstrap iterations (n={subsample})...")

    for i in range(n_iterations):
        seed = random_state + i
        np.random.seed(seed)
        
        # PAIRED SAMPLING: Select same row indices for both
        idx = np.random.choice(common_indices, size=min(len(common_indices), subsample), replace=False)
        
        # Extract data for this iteration
        df_sub_b = df_before.iloc[idx].copy()
        df_sub_a = df_after.iloc[idx].copy()
        
        X_b = df_sub_b[markers].fillna(0).values.astype('float32')
        X_a = df_sub_a[markers].fillna(0).values.astype('float32')
        
        # Setup AnnData objects
        adata_b = ad.AnnData(X_b)
        adata_a = ad.AnnData(X_a)
        
        for ad_obj, df_sub in zip([adata_b, adata_a], [df_sub_b, df_sub_a]):
            ad_obj.obs[batch_key] = df_sub[batch_key].astype('category').values
            if has_labels:
                ad_obj.obs[actual_label_key] = df_sub[actual_label_key].astype('category').values
            
            sc.pp.scale(ad_obj)
            sc.tl.pca(ad_obj, n_comps=min(50, X_b.shape[1]-1), svd_solver='arpack', random_state=seed)
            sc.pp.neighbors(ad_obj, use_rep='X_pca', n_neighbors=15)

        # --- 1. Batch Removal Metrics ---
        # ASW Batch: Raw silhouette score (higher means more batch separation)
        asw_b_before = abs(scib.me.silhouette(adata_b, label_key=batch_key, embed='X_pca'))
        asw_b_after = abs(scib.me.silhouette(adata_a, label_key=batch_key, embed='X_pca'))

        # PCR: scib.me.pcr score (higher means batch explains more variance)
        pcr_before = scib.me.pcr(adata_b, covariate=batch_key, embed='X_pca', recompute_pca=False)
        pcr_after = scib.me.pcr(adata_a, covariate=batch_key, embed='X_pca', recompute_pca=False)
        
        # --- 2. Bio-conservation Metrics ---
        bio_metrics_b = [0, 0, 0, 0] # gc, asw_l, nmi, ari
        bio_metrics_a = [0, 0, 0, 0]
        
        if has_labels:
            for j, ad_obj in enumerate([adata_b, adata_a]):
                gc = scib.me.graph_connectivity(ad_obj, label_key=actual_label_key)
                asw_l = scib.me.silhouette(ad_obj, label_key=actual_label_key, embed='X_pca')
                
                # Resolution optimization: find max NMI/ARI across resolutions
                best_nmi, best_ari = 0.0, 0.0
                # Scan from 0.2 to 2.0 with a step of 0.2 for high precision
                for res in np.arange(0.2, 2, 0.2):
                    try:
                        sc.tl.leiden(ad_obj, resolution=res, key_added='leiden_temp', random_state=seed)
                        curr_nmi = scib.me.nmi(ad_obj, cluster_key='leiden_temp', label_key=actual_label_key)
                        curr_ari = scib.me.ari(ad_obj, cluster_key='leiden_temp', label_key=actual_label_key)
                        
                        if curr_nmi > best_nmi:
                            best_nmi = curr_nmi
                            best_ari = curr_ari
                    except:
                        continue
                
                if j == 0: bio_metrics_b = [gc, asw_l, best_nmi, best_ari]
                else: bio_metrics_a = [gc, asw_l, best_nmi, best_ari]

        results_before.append([asw_b_before, pcr_before] + bio_metrics_b)
        results_after.append([asw_b_after, pcr_after] + bio_metrics_a)

    def finalize_metric(res_list):
        arr = np.array(res_list)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
        return {
            'asw_batch': (means[0], stds[0], arr[:, 0].tolist()),
            'pcr': (means[1], stds[1], arr[:, 1].tolist()),
            'graph_connectivity': (means[2], stds[2], arr[:, 2].tolist()),
            'asw_label': (means[3], stds[3], arr[:, 3].tolist()),
            'nmi': (means[4], stds[4], arr[:, 4].tolist()),
            'ari': (means[5], stds[5], arr[:, 5].tolist())
        }

    return finalize_metric(results_before), finalize_metric(results_after)

def plot_metrics_comparison(metrics_before, metrics_after, panel, save_dir):
    """
    Plots a comparison of batch correction metrics before and after correction for a given panel.
    Uses bootstrapping-based metrics (ASW, PCR, Graph Connectivity, NMI, ARI).

    Args:
        metrics_before (dict): Metrics calculated from raw data.
        metrics_after (dict): Metrics calculated from corrected data.
        panel (str): Name of the antibody panel.
        save_dir (str): Path to save the summary plot and CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine which metrics are available
    has_labels = metrics_before['asw_label'][0] is not None
    
    # Style setup
    sns.set_theme(style="white", font_scale=1.1)
    raw_color = _BEFORE_COLOR
    corr_color = _AFTER_COLOR

    all_labels = [
        'ASW (Batch) ↓',        # lower is better — batch silhouette should shrink
        'PCR ↓',                 # lower is better — plate variance in PCA should shrink
        'Graph Connectivity ↑',  # higher is better — cells should mix across batches
        'ASW (Cell-type) ↑',    # higher is better — cell types should stay separated
        'NMI (Conservation) ↑', # higher is better — cluster labels should be preserved
        'ARI (Conservation) ↑'  # higher is better — cluster labels should be preserved
    ]
    
    metric_keys = ['asw_batch', 'pcr', 'graph_connectivity', 'asw_label', 'nmi', 'ari']
    
    # Filter available metrics
    # Only keep Batch Removal metrics (0, 1) by default
    indices = [0, 1] 
    
    # Bio-conservation metrics (2, 3, 4, 5) only if labels exist and values are valid
    if has_labels:
        for i in [2, 3, 4, 5]:
            key = metric_keys[i]
            val = metrics_before.get(key)
            if val is not None and val[0] is not None and val[0] > 1e-6:
                indices.append(i)
    
    # Define plot labels based on indices
    plot_labels = [all_labels[i] for i in indices]
    
    before_means = []
    before_stds = []
    after_means = []
    after_stds = []
    raw_p_values = []
    
    for i in indices:
        key = metric_keys[i]
        b_val_tuple = metrics_before.get(key)
        a_val_tuple = metrics_after.get(key)
        
        # Ensure we have valid data
        if b_val_tuple is None or a_val_tuple is None or b_val_tuple[0] is None:
            before_means.append(0)
            before_stds.append(0)
            after_means.append(0)
            after_stds.append(0)
            raw_p_values.append(np.nan)
            continue

        b_mean, b_std, b_vals = b_val_tuple
        a_mean, a_std, a_vals = a_val_tuple
        
        before_means.append(b_mean if b_mean is not None else 0)
        before_stds.append(b_std if b_std is not None else 0)
        after_means.append(a_mean if a_mean is not None else 0)
        after_stds.append(a_std if a_std is not None else 0)
        
        # Calculate P-value using Bootstrap Hypothesis Testing (Empirical P-value)
        if b_vals is not None and a_vals is not None and len(b_vals) > 1 and len(a_vals) > 1:
            valid_mask = ~np.isnan(b_vals) & ~np.isnan(a_vals)
            b_clean = np.array(b_vals)[valid_mask]
            a_clean = np.array(a_vals)[valid_mask]
            
            if len(b_clean) > 0:
                diffs = a_clean - b_clean
                
                # DIRECTION SENSITIVE P-VALUE:
                # 1. Batch Removal (Low is better): ASW-Batch, PCR
                if key in ['asw_batch', 'pcr']:
                    p = np.mean(diffs >= 0)  # Prob that it failed to decrease
                # 2. Batch Removal (High is better): Graph Connectivity
                # 3. Bio-conservation (High is better): ASW-Label, NMI, ARI
                else:
                    p = np.mean(diffs <= 0)  # Prob that it failed to increase
                
                raw_p_values.append(p)
            else:
                raw_p_values.append(np.nan)
        else:
            raw_p_values.append(np.nan)

    # Apply FDR correction (Benjamini-Hochberg) to raw p-values
    p_values = np.array([np.nan] * len(raw_p_values))
    mask = ~np.isnan(raw_p_values)
    if mask.any():
        _, corrected_p, _, _ = multipletests(np.array(raw_p_values)[mask], method='fdr_bh')
        p_values[mask] = corrected_p

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(plot_labels)), 7))
    x = np.arange(len(plot_labels))
    width = 0.35

    # Plot bars
    ax.bar(x - width/2, before_means, width, yerr=before_stds, label='Before', 
            color=raw_color, edgecolor='black', linewidth=1.2, capsize=5)
    ax.bar(x + width/2, after_means, width, yerr=after_stds, label='After', 
            color=corr_color, edgecolor='black', linewidth=1.2, capsize=5)

    # Add explanatory category labels below the X-axis
    y_group_line = -0.15  
    y_group_text = -0.16
    
    # Identify the last index for Batch Removal
    # 0: ASW Batch, 1: PCR, 2: GC (indices 0, 1, 2)
    group1_indices = [idx_in_plot for plot_pos, idx_in_plot in enumerate(indices) if idx_in_plot <= 2]
    if group1_indices:
        group1_start_x = 0
        group1_end_x = len(group1_indices) - 1
        ax.plot([x[group1_start_x]-width, x[group1_end_x]+width], [y_group_line, y_group_line], 
                color="#2E4053", lw=2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x[group1_start_x]+x[group1_end_x])/2, y_group_text, 'Batch Removal', 
                ha='center', va='top', fontsize=11, fontweight='bold', color='#2E4053',
                transform=ax.get_xaxis_transform())
    
    # 2. Bio-conservation Group (ASW Label, NMI, ARI) -> Indices 3, 4, 5
    group2_indices_in_plot = [plot_pos for plot_pos, idx_in_plot in enumerate(indices) if idx_in_plot >= 3]
    if group2_indices_in_plot:
        group2_start_x = group2_indices_in_plot[0]
        group2_end_x = group2_indices_in_plot[-1]
        ax.plot([x[group2_start_x]-width, x[group2_end_x]+width], [y_group_line, y_group_line], 
                color="#1E8449", lw=2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x[group2_start_x]+x[group2_end_x])/2, y_group_text, 'Bio-conservation', 
                ha='center', va='top', fontsize=11, fontweight='bold', color='#1E8449',
                transform=ax.get_xaxis_transform())
    
    # Adjust X-axis limits to only show columns we have data for
    ax.set_xlim(-0.5, len(indices) - 0.5)

    # Calculate global max for Y-axis optimization
    max_y_val = 0
    for i in range(len(indices)):
        val = max(before_means[i] + (before_stds[i] or 0), after_means[i] + (after_stds[i] or 0))
        if val > max_y_val: max_y_val = val
    
    # Set Y-axis limit to accommodate stars (add headroom consistent with scale)
    headroom = 0.2 if max_y_val > 0.1 else 0.05
    ax.set_ylim(0, max_y_val + headroom)

    # Add value labels        
    for i in range(len(x)):
        p_val = p_values[i]
        if not np.isnan(p_val):
            if p_val < 0.05:
                stars = "*"
                if p_val < 0.0001: stars = "****"
                elif p_val < 0.001: stars = "***"
                elif p_val < 0.01: stars = "**"
                text_color = "black"
            else:
                stars = "ns"
                text_color = "grey"
            
            y_max_bar = max(before_means[i] + (before_stds[i] or 0), after_means[i] + (after_stds[i] or 0))
            # Main significance line and star
            ax.plot([i - width/2, i - width/2, i + width/2, i + width/2], 
                    [y_max_bar + 0.05, y_max_bar + 0.07, y_max_bar + 0.07, y_max_bar + 0.05], 
                    lw=1.5, c='black')
            ax.text(i, y_max_bar + 0.07, stars, ha='center', va='bottom', 
                    fontsize=14 if stars != "ns" else 12, 
                    fontweight='bold', color=text_color)

    ax.set_ylabel('Score (Mean ± SD)', fontsize=15, fontweight='bold')
    ax.set_title(f'Batch Correction Metrics (Bootstrapping) - {panel}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    ax.set_xticklabels(plot_labels, fontsize=13, fontweight='bold')
    ax.set_xlabel("") # Removed the "Factor" or other xlabel

    ax.legend(title="Stage", frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=12, title_fontsize=13, prop={'weight':'bold'})
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()

    # Apply bold to legend title
    legend = ax.get_legend()
    if legend:
        plt.setp(legend.get_title(), fontweight='bold')

    # Adjust layout to make room for bottom annotations and side legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(save_dir, f"{panel}_metrics_summary_plot.png")
    _savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    summary_data = {
        'Metric': [metric_keys[i] for i in indices],
        'Before_Mean': before_means,
        'Before_Std': before_stds,
        'After_Mean': after_means,
        'After_Std': after_stds
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, f"{panel}_metrics_summary.csv"), index=False)
    print(f"[METRIC] Saved Bootstrapping metrics and plots to {save_dir}")


# Plot variance of each marker within each cell type before and after correction, side by side for both panels
def plot_variance_bar_celltype(var_P1_before, var_P1_after, var_P2_before=None, var_P2_after=None, save_path=None):
    # Filter the legend coloring warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Setting the 'color' property will override.*")
    
    # If initial return from compute_celltype_variance was empty, don't plot
    if var_P1_before.empty or var_P1_after.empty:
        print("[SKIP] plot_variance_bar_celltype: Data is empty. Skipping plot.")
        return
        
    has_p2 = (var_P2_before is not None and var_P2_after is not None and not var_P2_before.empty)
    
    panels = ['Panel1']
    dfs_before = [var_P1_before]
    dfs_after = [var_P1_after]
    
    if has_p2:
        panels.append('Panel2')
        dfs_before.append(var_P2_before)
        dfs_after.append(var_P2_after)
    
    # Identify unique cell types across available panels
    all_cell_lists = [df.index.tolist() for df in dfs_before]
    flat_list = [item for sublist in all_cell_lists for item in sublist]
    all_celltypes = sorted(list(set(flat_list)))

    n_celltypes = len(all_celltypes)

    # Calculate figure width based on number of cell types and panels
    n_axes = len(panels)
    fig, axes = plt.subplots(1, n_axes, figsize=(max(8 if not has_p2 else 12, n_celltypes*1.0), 6), sharey=True, squeeze=False)
    
    for idx, (panel, df_before, df_after) in enumerate(zip(panels, dfs_before, dfs_after)):
        ax = axes[0, idx]
        # Use the global sorted celltypes for this panel (if missing, they will be empty bars)
        x = np.arange(n_celltypes)
        width = 0.35
        
        # Compute mean variance across markers for each cell type
        mean_before = df_before.mean(axis=1).reindex(all_celltypes, fill_value=0)
        mean_after = df_after.mean(axis=1).reindex(all_celltypes, fill_value=0)
        
        # Bar plot: Before/After use consistent global colors across all figures
        ax.bar(x - width/2, mean_before, width=width, color=_BEFORE_COLOR, label='Before', edgecolor='black')
        ax.bar(x + width/2, mean_after, width=width, color=_AFTER_COLOR, label='After', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_celltypes, rotation=45, ha='right', fontsize=11)
        ax.set_title(panel, fontsize=15, fontweight='bold')
        ax.set_ylabel("Mean Marker Variance", fontsize=13, fontweight='bold')
        ax.axhline(0, color='grey', linestyle='--')
    
    # Create a single legend for both stages
    legend_handles = [
        plt.Rectangle((0,0),1,1, color=_BEFORE_COLOR, label='Before', edgecolor='black'),
        plt.Rectangle((0,0),1,1, color=_AFTER_COLOR, label='After', edgecolor='black')
    ]
    axes[0, n_axes-1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Stage', fontsize=11, title_fontsize=12)

    plt.suptitle("Variance of Shared Markers by Cell Type", fontsize=17, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1.15, 1])

    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# UMAP visualization of marker expression by cell type, showing how samples cluster by cell type before and after correction, for the shared markers
def umap_marker_expression(df_before, df_after, markers, panel_name, subsample=50000, random_state=42, save_path=None):
    warnings.filterwarnings("ignore", category=UserWarning, message=".*No data for colormapping provided.*")
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

    # Keep only markers present in both dataframes and ensure they are numeric
    markers_before = [m for m in markers if m in df_before.columns and pd.api.types.is_numeric_dtype(df_before[m])]
    markers_after  = [m for m in markers if m in df_after.columns and pd.api.types.is_numeric_dtype(df_after[m])]

    # Use shared numeric markers
    shared_markers = [m for m in markers_before if m in markers_after]
    
    if not shared_markers:
        print(f"[WARN] No shared numeric markers for UMAP expression in {panel_name}")
        return

    # Downsample
    n_before = min(subsample, df_before.shape[0])
    n_after  = min(subsample, df_after.shape[0])
    df_before_sub = df_before.sample(n_before, random_state=random_state)
    df_after_sub  = df_after.sample(n_after, random_state=random_state)

    # Combine data specifically for the numeric markers
    X_before = df_before_sub[shared_markers].fillna(0).values.astype(float)
    X_after = df_after_sub[shared_markers].fillna(0).values.astype(float)
    X_all = np.vstack([X_before, X_after])

    # AnnData
    adata = ad.AnnData(X_all)
    adata.var_names = shared_markers
    adata.obs["Stage"] = ["Before"] * n_before + ["After"] * n_after

    # PCA + neighbors + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, random_state=random_state)

    n_markers = len(shared_markers)
    n_cols = 5
    n_rows = int(np.ceil(n_markers / n_cols)) * 2

    # Increase figsize for better visibility
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.ravel()  # flatten

    for idx, marker in enumerate(shared_markers):
        # Before
        row_before = idx // n_cols * 2 
        col = idx % n_cols
        ax_before = axes[row_before * n_cols + col]
        sub_before = adata[adata.obs["Stage"] == "Before"]
        sc.pl.umap(sub_before, color=marker, ax=ax_before, show=False, frameon=False, size=20, cmap="jet")
        ax_before.set_title(f"{marker} - Before", fontsize=15, fontweight='bold')

        # After
        row_after = row_before + 1
        ax_after = axes[row_after * n_cols + col]
        sub_after = adata[adata.obs["Stage"] == "After"]
        sc.pl.umap(sub_after, color=marker, ax=ax_after, show=False, frameon=False, size=20, cmap="jet")
        ax_after.set_title(f"{marker} - After", fontsize=15, fontweight='bold')

    # Turn off any unused axes
    for ax in axes:
        if not ax.has_data():
            ax.axis("off")

    plt.suptitle(f"{panel_name} - Marker UMAP", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.98])
    if save_path:
        _savefig(save_path, fig=fig, bbox_inches="tight")
        print(f"UMAP marker expression plot saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)

# Correlation of marker expression before and after correction, to assess preservation of marker expression levels
def marker_expression_correlation_plot(df_P1_before, df_P1_after,
                                        df_P2_before=None, df_P2_after=None,
                                        save_dir=None, ctx=None):
    
    # Columns to exclude (metadata)
    if ctx is not None:
        exclude_cols = ctx.meta_cols + ["original_filename"]
    else:
        exclude_cols = ["panel","plate","sample_name","dataset",
                        "Event_length","Time","Anchor",
                        "condition","celltypes", "original_filename"]
    
    # Numeric markers only
    markers_P1 = [c for c in df_P1_before.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_P1_before[c])]
    # Filter out empty or whitespace markers from the lists
    markers_P1 = [m for m in markers_P1 if m and str(m).strip()]
    
    p1_before = df_P1_before[markers_P1].median()
    p1_after = df_P1_after[markers_P1].median()
    
    df_P1 = pd.DataFrame({
        "marker": markers_P1,
        "before": p1_before.values,
        "after": p1_after.values,
        "panel": "Panel1"
    })
    
    dfs_to_concat = [df_P1]

    # Panel2 markers
    has_p2 = (df_P2_before is not None and df_P2_after is not None)
    if has_p2:
        markers_P2 = [c for c in df_P2_before.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_P2_before[c]) and c and str(c).strip()]
        # Preserve every marker that is measured in Panel2. Merged matrices may
        # also contain all-NaN columns belonging only to Panel1; the finite-value
        # filter below removes those per panel. Do not restrict both panels to
        # their marker intersection, because panel-specific markers are valid
        # observations for expression-preservation analysis.
        p2_before = df_P2_before[markers_P2].median()
        p2_after = df_P2_after[markers_P2].median()
        df_P2 = pd.DataFrame({
            "marker": markers_P2,
            "before": p2_before.values,
            "after": p2_after.values,
            "panel": "Panel2"
        })
        dfs_to_concat.append(df_P2)

    # Combine
    df = pd.concat(dfs_to_concat, ignore_index=True)

    # A merged matrix can retain marker columns from another panel as all-NaN
    # columns. Their medians are non-finite and scipy.stats.pearsonr rejects
    # them, so keep only markers with usable before/after summary values.
    finite_mask = np.isfinite(df["before"]) & np.isfinite(df["after"])
    excluded = df.loc[~finite_mask, ["panel", "marker", "before", "after"]]
    if not excluded.empty:
        excluded_labels = ", ".join(
            f"{row.panel}:{row.marker}" for row in excluded.itertuples(index=False)
        )
        print(
            "[WARN] Excluding markers with non-finite before/after medians: "
            f"{excluded_labels}"
        )
    df = df.loc[finite_mask].copy()

    if len(df) < 2:
        raise ValueError(
            "Marker-expression correlation requires at least two markers with "
            "finite before/after medians."
        )
    
    # Save marker expression correlation to CSV
    if save_dir:
        csv_path = os.path.join(save_dir, "marker_expression_correlation_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[SAVE] Marker expression correlation data saved to: {csv_path}")

    # Pearson correlation
    r, p = pearsonr(df["before"], df["after"])
    
    # Save p-value and r to a text file
    if save_dir:
        p_val_path = os.path.join(save_dir, "marker_expression_correlation_p_value.txt")
        with open(p_val_path, "w") as f:
            f.write(f"Pearson Correlation (Marker Expression Preservation)\n")
            f.write(f"Correlation Coefficient (r): {r:.4f}\n")
            f.write(f"P-Value: {p:.2e}\n")
        print(f"Correlation p-value saved to: {p_val_path}")

    p_text = f"p = {p:.2e}" if p < 0.01 else f"p = {p:.4f}"
    
    # Plot
    plt.figure(figsize=(7,7))
    colors = {"Panel1": _PANEL1_COLOR, "Panel2": _PANEL2_COLOR}
    
    panels_to_plot = ["Panel1"]
    if has_p2: panels_to_plot.append("Panel2")

    for panel in panels_to_plot:
        sub = df[df["panel"] == panel]
        plt.scatter(
            sub["before"],
            sub["after"],
            label=panel,
            s=80,
            color=colors[panel],
            alpha=0.8
        )
    # Identity line
    min_val = min(df["before"].min(), df["after"].min())
    max_val = max(df["before"].max(), df["after"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="grey")

    # Annotate outliers — markers furthest from identity line
    df["residual"] = (df["after"] - df["before"]).abs()
    n_label = min(5, len(df))
    outliers = df.nlargest(n_label, "residual")
    ax_curr = plt.gca()
    for _, row in outliers.iterrows():
        ax_curr.annotate(
            row["marker"],
            xy=(row["before"], row["after"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, fontweight='bold', color='#2C3E50',
            arrowprops=dict(arrowstyle="-", color='#95A5A6', lw=0.8)
        )

    plt.xlabel("Median expression before correction", fontsize=12, fontweight='bold', fontname='Arial', labelpad=12)
    plt.ylabel("Median expression after correction", fontsize=12, fontweight='bold', fontname='Arial', labelpad=12)
    plt.title(f"Marker expression preservation\nr = {r:.3f}", fontsize=14, fontweight='bold', fontname='Arial', pad=20)

    plt.legend(fontsize=11, prop={'family': 'Arial', 'weight': 'bold'}, frameon=False)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.xticks(fontsize=10, fontweight='bold', fontname='Arial')
    plt.yticks(fontsize=10, fontweight='bold', fontname='Arial')
    plt.tight_layout()
    if save_dir:
        _savefig(os.path.join(save_dir, "marker_expression_correlation.png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def load_and_preprocess_panel_batch(panel_name, dataset_to_process, ctx, verbose=False):
    """
    Load data for batch-correction visualization:
    1. Time-Corrected Data (Before Batch Correction) -> Apply arcsinh
    2. Batch-Corrected Data (After Batch Correction)
    """
    # Load metadata.xlsx
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(project_root, "metadata", "metadata.xlsx")
    metadata_df = pd.read_excel(metadata_path) if os.path.exists(metadata_path) else None

    anchor_name = ctx.anchors.get(dataset_to_process)
    
    # Scan Time-Corrected directory to find valid plates (those containing the anchor)
    time_dir = os.path.join(ctx.output_dir, "time_corrected")
    time_index = get_indexed_files(
        time_dir,
        dataset_to_process,
        panel_name,
        metadata_df,
        recursive=True,
    )

    # Filter to only keep plates that contain the anchor sample
    valid_plates_index = {}
    for plate, files in time_index.items():
        is_anchor_found = False
        for f in files:
            file_name_orig = os.path.basename(f)
            
            # Use metadata to find the sample name for this file
            # Robust matching: compare base names without extension
            f_base = os.path.splitext(file_name_orig)[0].lower()
            
            def get_base(x):
                return os.path.splitext(str(x))[0].lower()

            match = metadata_df[metadata_df["file_name"].apply(get_base) == f_base]
            
            if not match.empty:
                current_sample = str(match.iloc[0].get("sample_name", ""))
                if current_sample == anchor_name:
                    is_anchor_found = True
                    break
        
        if is_anchor_found:
            valid_plates_index[plate] = files
    
    if not valid_plates_index:
        if verbose: print(f"[WARN] No anchor samples found for {dataset_to_process} {panel_name} in {time_dir}")
        return None, None, []

    # Flatten and load before-batch data
    all_before_files = []
    # Sort plates to ensure consistent order
    for plate_name in sorted(valid_plates_index.keys()):
        all_before_files.extend(valid_plates_index[plate_name])
        
    dfs_before = []
    markers = []
    
    # AGGRESSIVE MEMORY FIX: Use float32 and Category dtypes
    print(f"  [MEMORY] Loading raw files ({len(all_before_files)})...")
    
    for f in all_before_files:
        try:
            df = read_cytof_file(f, low_memory=False)
            
            # Standardize column names (metals to markers) first
            df.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in df.columns]
            df = apply_marker_schema(df, ctx)
            
            if not markers:
                _, markers = enforce_schema(df, ctx)
                
            # 3. Convert signal columns to float32
            signal_cols = [c for c in df.columns if any(char.isdigit() for char in c) or c == "Time"]
            
            for c in df.columns:
                if c in signal_cols:
                    try:
                        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
                    except Exception:
                        pass
            
            df = add_metadata(df, f, metadata_df=metadata_df)
            
            # 4. Convert metadata to category
            for col in ['plate', 'sample_name', 'condition', 'dataset', 'panel']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    
            dfs_before.append(df)
        except Exception as e:
            if verbose: print(f"  [WARN] Failed to load {os.path.basename(f)}: {e}")
            continue
    
    if not dfs_before:
        return None, None, []
        
    df_b = pd.concat(dfs_before, ignore_index=True)
    del dfs_before
    gc.collect()

    df_b = arcsinh(df_b, markers) # Apply arcsinh to Time-Corrected data

    # Scan Batch-Corrected directory (Results root) for the corresponding plates
    corrected_dir = os.path.join(ctx.output_dir, "corrected")
    recursive_outputs = os.path.isdir(corrected_dir)
    if not recursive_outputs:
        # Backward compatibility with legacy flat output directories.
        corrected_dir = ctx.output_dir
    corr_index = get_indexed_files(
        corrected_dir,
        dataset_to_process,
        panel_name,
        metadata_df,
        exclude_uncorrected=True,
        recursive=recursive_outputs,
    )
    all_after_files = []
    for plate in sorted(valid_plates_index.keys()):
        if plate in corr_index:
            all_after_files.extend(corr_index[plate])
    
    if not all_after_files:
        if verbose: print(f"[WARN] No batch-corrected files found for {dataset_to_process} {panel_name}")
        return df_b, pd.DataFrame(), markers

    dfs_after = []
    print(f"  [MEMORY] Loading corrected files ({len(all_after_files)}) ...")
    for f in all_after_files:
        try:
            df = read_cytof_file(f, low_memory=False)
            df = add_metadata(df, f, metadata_df=metadata_df)
            
            # 2. Convert metadata to category
            for col in ['plate', 'sample_name', 'condition', 'dataset', 'panel']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            dfs_after.append(df)
        except Exception as e:
            if verbose: print(f"  [WARN] Failed to load corrected {os.path.basename(f)}: {e}")
            continue
    
    if not dfs_after:
        return df_b, pd.DataFrame(), markers

    df_a = pd.concat(dfs_after, ignore_index=True)
    del dfs_after
    gc.collect()

    # Filter out non-marker columns from the markers list
    if markers:
        markers = [m for m in markers if m not in ctx.meta_cols and m != "original_filename"]

    return df_b, df_a, markers
    

