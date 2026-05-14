from glob import glob
import os
import re
from collections import defaultdict
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
import scanpy as sc
import seaborn as sns
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, entropy, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests
from glob import glob


from .utils import harmonize_marker, apply_marker_schema, add_metadata, assign_cell_type, enforce_schema, read_cytof_file, get_indexed_files

# Umap visualization colored by cell type
def umap_by_celltype(df_before, df_after, celltype_col="cell_type",
                                panel=None, markers=None, subsample=50000,
                                n_pcs=30, n_neighbors=15, min_dist=0.3,
                                random_state=42, size=8):
    """
    Generates side-by-side UMAP plots for 'Before' and 'After' time correction.
    
    The goal is to verify that time-drift correction has not distorted the biological
    clustering. Biological populations (cell types) should remain grouped similarly
    in both plots, but with tighter or more consistent distributions in the 'After' plot.

    Args:
        df_before (pd.DataFrame): Data before time correction.
        df_after (pd.DataFrame): Data after time correction.
        celltype_col (str): Column name containing population labels.
        panel (str): Antibody panel to visualize.
        markers (list): Biological markers to use for dimensionality reduction.
        subsample (int): Max cells to include in UMAP for speed.
        n_pcs, n_neighbors, min_dist: standard UMAP/PCA parameters.
    """
    # Subset panel
    df_b = df_before[df_before['panel']==panel].dropna(axis=1, how="all").copy()
    df_a = df_after[df_after['panel']==panel].dropna(axis=1, how="all").copy()
    
    # Subsample if needed
    if subsample is not None:
        if len(df_b) > subsample:
            df_b = df_b.sample(n=subsample, random_state=random_state)
        if len(df_a) > subsample:
            df_a = df_a.sample(n=subsample, random_state=random_state)
    
    # Determine markers if not provided
    markers = [m for m in markers if m in df_b.columns and pd.api.types.is_numeric_dtype(df_b[m])]
    print(f"Using {len(markers)} markers for UMAP: {markers}")


    # Add a column to indicate Before/After
    df_b["Correction"] = "Before"
    df_a["Correction"] = "After"
    
    # Combine datasets
    df_combined = pd.concat([df_b, df_a], ignore_index=True)
    
    # Create AnnData object
    X = df_combined[markers].fillna(0).to_numpy(dtype=np.float32)
    adata = ad.AnnData(X)
    if celltype_col in df_combined.columns and not df_combined[celltype_col].isna().all():
        adata.obs[celltype_col] = df_combined[celltype_col].astype("category").values
    else:
        adata.obs[celltype_col] = "unknown"
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")

    adata.obs["Correction"] = df_combined["Correction"].astype("category").values
    
    # Preprocessing + PCA + neighbors + UMAP
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    
    # Force consistent colors for cell types
    if celltype_col in adata.obs:
        # Get all unique cell types across Before and After
        unique_types = sorted(adata.obs[celltype_col].unique())
        # Use a standard Scanpy palette
        adata.uns[f"{celltype_col}_colors"] = sc.pl.palettes.default_20[:len(unique_types)]

    # Split UMAP coordinates for Before and After
    adata_before = adata[adata.obs["Correction"]=="Before"].copy()
    adata_after = adata[adata.obs["Correction"]=="After"].copy()
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(14,7))
    
    sc.pl.umap(adata_before, color=celltype_col, ax=axes[0], show=False, size=size, legend_loc='right margin')
    axes[0].set_title("Before Correction", fontsize=16, fontweight='bold')
    
    sc.pl.umap(adata_after, color=celltype_col, ax=axes[1], show=False, size=size, legend_loc='right margin')
    axes[1].set_title("After Correction", fontsize=16, fontweight='bold')
    
    plt.tight_layout()

    return adata

# Millions formatter for x-axis labels
def millions_formatter(x, pos):
    return f"{int(x/1e6)}M"

# Function to plot marker trends over time for each sample in a given plate and panel
def marker_trends(df, plate, panel, markers, save_dir, window=500):
    """
    Plots the rolling median of marker intensities over the acquisition time.
    
    This is the primary diagnostic for time-drift. It displays signal stability
    across each sample run. In 'Before' plots, signal often fluctuates; 
    in 'After' plots, the stable marker/DNA and biological marker 
    trends should be flattened.

    Args:
        df (pd.DataFrame): Data containing 'Time' and marker values.
        plate (str): Plate ID to visualize.
        panel (str): Panel ID.
        markers (list): Markers to track.
        save_dir (str): Directory for output PNGs.
        window (int): Rolling window size for smoothing.
    """
    os.makedirs(save_dir, exist_ok=True)  # create folder if not exist
    
    df_panel = df[df['panel'] == panel].dropna(axis=1, how='all')
    df_plate = df_panel[df_panel["plate"] == plate].copy()
    samples = df_plate['sample_name'].unique()

    markers = [m for m in markers if m in df_plate.columns and pd.api.types.is_numeric_dtype(df_plate[m])]

    for marker in markers:
        n_samples = len(samples)
        n_cols = 4
        n_rows = int(np.ceil(n_samples / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        axes = axes.flatten()

        for i, sample in enumerate(samples):
            df_sample = df_plate[df_plate['sample_name'] == sample].sort_values("Time")
            
            # Speed up: Downsample for plotting if sample is too large
            if len(df_sample) > 20000:
                # Use iloc[::step] to maintain time sequence distribution
                step = int(len(df_sample) / 10000)
                df_plot = df_sample.iloc[::step].copy()
            else:
                df_plot = df_sample.copy()

            # Compute rolling median and percentiles on the (potentially downsampled) data
            # Adjust window size if data was downsampled
            eff_window = max(10, int(window / (len(df_sample)/len(df_plot)))) if len(df_sample) > 20000 else window
            
            smoothed = df_plot[marker].rolling(window=eff_window, center=True, min_periods=1).median()
            low = df_plot[marker].rolling(window=eff_window, center=True, min_periods=1).quantile(0.05)
            high = df_plot[marker].rolling(window=eff_window, center=True, min_periods=1).quantile(0.95)

            # Plot
            ax = axes[i]
            ax.plot(df_plot["Time"], smoothed, color="blue", lw=2, label="Median trend")
            ax.fill_between(df_plot["Time"], low, high, color="blue", alpha=0.2, label="5%-95% percentile")
            ax.set_title(f"Sample: {sample}", fontsize=13)
            ax.set_xlabel("Acquisition Time", fontsize=15, fontweight='bold')
            ax.set_ylabel("Intensity", fontsize=15, fontweight='bold')
            ax.xaxis.set_major_formatter(millions_formatter)
            ax.tick_params(axis='both', labelsize=12)

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(f"{marker} trends over time - Plate: {plate}", fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0,0,1,0.95])

        # Save figure as high-resolution PNG
        save_path = os.path.join(save_dir, f"{panel}_{plate}_{marker}_Timetrend.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

# Visualize the time effect on marker intensities before and after correction
# Millions formatter
def millions_formatter(x, pos):
    return f"{int(x/1e6)}M"

def plot_time_effect_density(
    before_df,
    after_df,
    panel,
    markers,
    save_dir,
    window_ylim=(0,10),
    subsample=50000
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Filter panel
    before_df = before_df[before_df['panel'] == panel].dropna(axis=1, how="all")
    after_df = after_df[after_df['panel'] == panel].dropna(axis=1, how="all")

    markers = [m for m in markers if m in before_df.columns and pd.api.types.is_numeric_dtype(before_df[m])]

    plates = before_df['plate'].unique()

    if len(plates) == 0:
        print(f"No plates found for panel={panel}. Skipping.")
        return

    ncols = 4
    nrows = max(1, int(np.ceil(len(plates) / ncols)))

    for marker in markers:
        # Check if marker exists in both dataframes
        if marker not in before_df.columns:
            print(f"{marker} not found in dataframe. Skipping.")
            continue

        # BEFORE CORRECTION
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).flatten()

        for ax, plate in zip(axes, plates):
            df_plate = before_df[before_df['plate'] == plate].dropna(subset=[marker])

            if df_plate.empty:
                ax.axis("off")
                continue
            
            # Subsample per plate
            if subsample is not None and len(df_plate) > subsample:
                df_plate = df_plate.sample(n=subsample, random_state=42)

            x = df_plate['Time']
            y = df_plate[marker]
            y = np.arcsinh(y / 5)

            hb = ax.hexbin(
                x, y,
                gridsize=200,
                cmap='jet',
                bins='log'
            )

            ax.set_title(f"{marker} - {plate} (Before)", fontsize=15, fontweight='bold')
            ax.set_xlabel("Time", fontsize=15, fontweight='bold')
            ax.set_ylabel("Arcsinh(Intensity)", fontsize=15, fontweight='bold')
            ax.set_ylim(window_ylim)
            ax.xaxis.set_major_formatter(millions_formatter)
            ax.tick_params(axis='both', labelsize=12)

        for ax in axes[len(plates):]:
            ax.axis("off")

        plt.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(
                save_dir,
                f"{panel}_{marker}_BeforeCorrection_density.png"
            )
            fig.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

        # AFTER CORRECTION
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = np.array(axes).flatten()

        for ax, plate in zip(axes, plates):
            df_plate = after_df[after_df['plate'] == plate].dropna(subset=[marker])

            if df_plate.empty:
                ax.axis("off")
                continue
            
            # Subsample per plate
            if subsample is not None and len(df_plate) > subsample:
                df_plate = df_plate.sample(n=subsample, random_state=42)

            x = df_plate['Time']
            y = df_plate[marker]
            y = np.arcsinh(y / 5)

            hb = ax.hexbin(
                x, y,
                gridsize=200,
                cmap='jet',
                bins='log'
            )

            ax.set_title(f"{marker} - {plate} (After)", fontsize=15, fontweight='bold')
            ax.set_xlabel("Time", fontsize=15, fontweight='bold')
            ax.set_ylabel("Arcsinh(Intensity)", fontsize=15, fontweight='bold')
            ax.set_ylim(window_ylim)
            ax.xaxis.set_major_formatter(millions_formatter)
            ax.tick_params(axis='both', labelsize=12)

        for ax in axes[len(plates):]:
            ax.axis("off")

        plt.tight_layout()

        if save_dir is not None:
            save_path = os.path.join(
                save_dir,
                f"{panel}_{marker}_AfterCorrection_density.png"
            )
            fig.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

def plot_time_effect_density_merged(
    before_df,
    after_df,
    panel,
    markers,
    save_dir,
    window_ylim=(0,11),
    subsample=100000
):
    """
    Plots the time-drift effect for all plates arranged sequentially on the X-axis.
    Each plate occupies its own segment of the X-axis, ensuring data is not over-plotted.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # MEMORY OPTIMIZATION: Use view/query and avoid .copy() if possible to prevent 5GB allocations
    mask_b = (before_df['panel'].astype(str) == str(panel))
    mask_a = (after_df['panel'].astype(str) == str(panel))
    
    # Check if either dataframe needs filtering. If the whole DF is already for this panel, avoid copying.
    if mask_b.all():
        df_b = before_df
    else:
        df_b = before_df[mask_b].copy() 

    if mask_a.all():
        df_a = after_df
    else:
        df_a = after_df[mask_a].copy()

    # Visualization only needs ~10k points per plate to look accurate
    max_total_points = 1000000
    if len(df_b) > max_total_points:
        df_b = df_b.sample(n=max_total_points, random_state=42)
    if len(df_a) > max_total_points:
        df_a = df_a.sample(n=max_total_points, random_state=42)

    markers = [m for m in markers if m in df_b.columns and pd.api.types.is_numeric_dtype(df_b[m])]

    for marker in markers:
        plates_b = df_b['plate'].astype(str).unique() if 'plate' in df_b.columns else []
        plates_a = df_a['plate'].astype(str).unique() if 'plate' in df_a.columns else []
        union_plates = list(set(plates_b) & set(plates_a))

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

        plates = sorted(union_plates, key=natural_sort_key)
        
        if not plates:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        y_all = pd.concat([df_b[marker], df_a[marker]]).dropna()
        if not y_all.empty:
            global_ymin, global_ymax = y_all.min(), y_all.max()
            global_ymin -= 0.5
            global_ymax += 0.5
        else:
            global_ymin, global_ymax = window_ylim

        for idx, (df_plot, title_suffix) in enumerate(zip([df_b, df_a], ["Before", "After"])):
            ax = axes[idx]
            
            # Check if Time column exists. 
            if 'Time' not in df_plot.columns:
                print(f"  [SKIP] Skipping alignment plot for {marker} because 'Time' column is missing in {title_suffix} data.")
                plt.close(fig) # Close the figure we just created
                return # Skip this marker entirely
            
            plot_cols = [marker, 'plate', 'Time']
            data = df_plot[plot_cols].dropna().copy()
            if data.empty:
                plt.close(fig)
                return
                
            # --- PLATE SEQUENCING LOGIC ---
            # 1. Normalize time within each plate to [0, 1] and then shift by plate index
            data['Global_Time'] = 0.0
            plate_centers = []
            
            for p_idx, p in enumerate(plates):
                p_mask = data['plate'].astype(str) == p
                if not p_mask.any():
                    continue
                
                # Internal time for this plate
                p_time = data.loc[p_mask, 'Time']
                t_min, t_max = p_time.min(), p_time.max()
                
                # Scale to [0, 0.9] to leave a small gap between plates
                if t_max > t_min:
                    normalized = (p_time - t_min) / (t_max - t_min) * 0.9
                else:
                    normalized = 0.45
                
                # Shift by plate index
                data.loc[p_mask, 'Global_Time'] = normalized + p_idx
                plate_centers.append(p_idx + 0.45) # Midpoint for label

            # Subsample globally if needed
            if subsample and len(data) > subsample:
                data = data.sample(n=subsample, random_state=42)
            
            x = data['Global_Time']
            y = data[marker]
            
            hb = ax.hexbin(
                x, y,
                gridsize=200,
                cmap='jet',
                bins='log'
            )
            
            # Add Plate markers (vertical lines/labels) to the X-axis
            n_plates = len(plates)
            label_step = max(1, n_plates // 10) 
            
            for p_idx, (p, p_mid) in enumerate(zip(plates, plate_centers)):
                # Draw vertical separator
                ax.axvline(p_idx, color='grey', linestyle='-', alpha=0.3, lw=1)
                if p_idx == len(plates) - 1:
                    ax.axvline(p_idx + 0.9, color='grey', linestyle='-', alpha=0.3, lw=1)
                
                # Plate Label - only show some and rotate if crowded
                if p_idx % label_step == 0:
                    rotation = 45 if n_plates > 5 else 0
                    display_label = str(p) if str(p).lower().startswith('plate') else f"P{p}"
                    ax.text(p_mid, -0.05, display_label, transform=ax.get_xaxis_transform(), 
                            ha='right' if rotation else 'center', 
                            va='top', fontsize=10 if n_plates > 10 else 12, 
                            fontweight='bold', color='#2E4053',
                            rotation=rotation)

            ax.set_title(f"{marker} - Multi-Plate Alignment ({title_suffix})", fontsize=18, fontweight='bold')
            ax.set_xlabel("Plates (Sequential)", fontsize=15, fontweight='bold')
            ax.set_ylabel("Arcsinh Intensity", fontsize=15, fontweight='bold')
            ax.set_ylim(0, 10)
            
            # Remove default X-axis ticks as we have custom plate labels
            ax.set_xticks([])
            ax.tick_params(labelsize=13)
            
            # Add a colorbar
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Log10(Count)', fontsize=13, fontweight='bold')

        plt.tight_layout()
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{panel}_{marker}_PlateAlignment_TimeDrift.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [SAVED] Plate alignment drift plot -> {save_path}")
        else:
            plt.show()
        plt.close(fig)

        plt.tight_layout()

# Plot signal distributions for each marker across plates before and after correction (before and after in the same figure)
def plot_signal_distributions_joint(df_before, df_after, panel, markers, save_dir,
                                        lower_pct=1, upper_pct=99, colors=("C0","C1"), subsample=50000, 
                                        kd_df=None):
    """
    Overlays 'Before' and 'After' distribution curves for every marker.
    
    Includes statistical annotation per plot:
    - Adjusted P-value (FDR corrected Mann-Whitney U): Significance of the shift.
    - KL Divergence (KD): Magnitude of signal profile change.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    df_before = df_before[df_before['panel'] == panel].dropna(axis=1, how="all")
    df_after = df_after[df_after['panel'] == panel].dropna(axis=1, how="all")

    markers = [m for m in markers if m in df_before.columns and pd.api.types.is_numeric_dtype(df_before[m])]

    plates = [p for p in df_before['plate'].unique() if p in df_after['plate'].unique()]
    if len(plates) == 0:
        print(f"No plates found for panel={panel}.")
        return

    n_cols = 5
    for plate in plates:
        df_b_plate = df_before[df_before['plate']==plate]
        df_a_plate = df_after[df_after['plate']==plate]

        # Calculate marker ranges using a subset for speed
        marker_ranges = {}
        df_b_sample = df_b_plate.sample(n=min(len(df_b_plate), subsample), random_state=42)
        for marker in markers:
            if marker not in df_b_sample.columns:
                continue
            all_vals = df_b_sample[marker].dropna()
            if len(all_vals) == 0:
                continue
            lower = np.percentile(all_vals, lower_pct)
            upper = np.percentile(all_vals, upper_pct)
            buffer = 0.2 * (upper - lower) if upper > lower else 0.5
            marker_ranges[marker] = (lower - buffer, upper + buffer)

        n_rows = int(np.ceil(len(markers)/n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

        # 1. First, collect all raw p-values and KD values for this plate
        p_raw_list = []
        kd_list = []
        valid_markers = []

        for marker in markers:
            if marker in df_b_plate.columns and marker in df_a_plate.columns:
                data_b = df_b_plate[marker].dropna()
                data_a = df_a_plate[marker].dropna()
                
                if len(data_b) > 10 and len(data_a) > 10:
                    try:
                        # Downsample for P-value calculation to avoid over-sensitivity
                        p_sample_size = 10000
                        db_p = data_b.sample(n=min(len(data_b), p_sample_size), random_state=42)
                        da_p = data_a.sample(n=min(len(data_a), p_sample_size), random_state=42)
                        
                        _, p = mannwhitneyu(db_p, da_p, alternative='two-sided')
                        
                        if kd_df is not None and marker in kd_df['marker'].values:
                            kd_val = kd_df.loc[kd_df['marker'] == marker, 'KD'].values[0]
                        else:
                            min_val, max_val = min(data_b.min(), data_a.min()), max(data_b.max(), data_a.max())
                            bins = np.linspace(min_val, max_val, 101)
                            pb, _ = np.histogram(data_b, bins=bins, density=True)
                            pa, _ = np.histogram(data_a, bins=bins, density=True)
                            kd_val = entropy(pb + 1e-10, pa + 1e-10)
                        
                        p_raw_list.append(p)
                        kd_list.append(kd_val)
                        valid_markers.append(marker)
                    except:
                        pass

        # 2. Perform FDR correction across all markers in this plate
        p_adj_list = []
        if p_raw_list:
            _, p_adj_list, _, _ = multipletests(p_raw_list, method='fdr_bh')

        marker_to_stats = {m: {'p_adj': p_adj, 'kd': kd} for m, p_adj, kd in zip(valid_markers, p_adj_list, kd_list)}

        for m_idx, marker in enumerate(markers):
            row = m_idx // n_cols
            col = m_idx % n_cols
            ax = axes[row, col]

            if marker not in marker_ranges:
                ax.axis("off")
                continue

            # Plot Before Correction
            if marker in df_b_plate.columns:
                # Speed up: Subsample for KDE
                data_b = df_b_plate[marker].dropna()
                if len(data_b) > subsample:
                    data_b = data_b.sample(n=subsample, random_state=42)
                
                if len(data_b) > 0:
                    sns.kdeplot(data_b, ax=ax, color=colors[0],
                                linewidth=2, label="Before Correction", clip=marker_ranges[marker])
            
            # Plot After Correction
            if marker in df_a_plate.columns:
                # Speed up: Subsample for KDE
                data_a = df_a_plate[marker].dropna()
                if len(data_a) > subsample:
                    data_a = data_a.sample(n=subsample, random_state=42)
                
                if len(data_a) > 0:
                    sns.kdeplot(data_a, ax=ax, color=colors[1],
                                linewidth=2, label="After Correction", clip=marker_ranges[marker])

            # Annotate with KL Divergence (KD) and Stats (Using FDR adjusted P)
            if marker in marker_to_stats:
                stats = marker_to_stats[marker]
                p_adj = stats['p_adj']
                kd_val = stats['kd']
                
                def get_stars(p):
                    if p < 0.0001: return "****"
                    if p < 0.001: return "***"
                    if p < 0.01: return "**"
                    if p < 0.05: return "*"
                    return "ns"

                stars = get_stars(p_adj)
                stat_text = f"{stars}\np={p_adj:.2e}\nKD={kd_val:.4f}"
                
                ax.text(0.95, 0.95, stat_text, transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

            ax.set_xlim(marker_ranges[marker])
            ax.set_title(marker, fontsize=12, fontweight='bold')
            ax.set_xlabel("Signal", fontsize=12, fontweight='bold')
            ax.set_ylabel("Density", fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', labelsize=10)

        # Turn off unused axes
        for empty_idx in range(len(markers), n_rows*n_cols):
            row = empty_idx // n_cols
            col = empty_idx % n_cols
            axes[row, col].axis("off")

        # Add global legend
        fig.legend(handles=[Line2D([0],[0], color=colors[0], lw=3, label="Before Correction"),
                            Line2D([0],[0], color=colors[1], lw=3, label="After Correction")],
                    loc='upper right', prop={'size': 14, 'weight': 'bold'})

        plt.tight_layout(rect=[0,0,0.85,0.95])
        plt.suptitle(f"Plate {plate} - Panel {panel}", fontsize=20, fontweight='bold', y=0.98)
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{panel}_{plate}_distribution.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

# Calculate Mann-Whitney U test p-values before and after correction for each marker
def calculate_mwu(df_before, df_after, panel, exclude_cols=None, subsample=10000, ctx=None):
    if exclude_cols is None:
        exclude_cols = list(ctx.meta_cols) + ["original_filename", "cell_type", "CD45pos"]
        if getattr(ctx, "celltypes", None):
            exclude_cols.extend(ctx.celltypes)
        if getattr(ctx, "celltype_gating", None):
            exclude_cols.extend(ctx.celltype_gating)
        exclude_cols = list(dict.fromkeys(exclude_cols))
    
    # Subset panel
    df_before = df_before[df_before['panel'] == panel].dropna(axis=1, how="all").copy()
    df_after = df_after[df_after['panel'] == panel].dropna(axis=1, how="all").copy()
    
    # Select markers
    markers = [c for c in df_before.columns if c not in exclude_cols and c in df_after.columns]
    
    # Ensure markers are numeric
    markers = [m for m in markers if pd.api.types.is_numeric_dtype(df_before[m])]
    
    # Subsample to speed up MWU for millions of cells
    if len(df_before) > subsample:
        df_before = df_before.sample(n=subsample, random_state=42)
    if len(df_after) > subsample:
        df_after = df_after.sample(n=subsample, random_state=42)

    results = []

    for marker in markers:
        x_before = df_before[marker].dropna().values
        x_after = df_after[marker].dropna().values

        if len(x_before) == 0 or len(x_after) == 0:
            results.append({"marker": marker, "p_value": np.nan, "statistic": np.nan, "-log10(p)": 0})
            continue

        # Mann-Whitney U test
        stat, p = mannwhitneyu(x_before, x_after, alternative='two-sided')
        
        # Use -log10(p-value) for visualization
        log_p = -np.log10(p + 1e-300) 
        
        results.append({
            "marker": marker, 
            "p_value": p, 
            "statistic": stat,
            "-log10(p)": log_p
        })

    return pd.DataFrame(results).sort_values("-log10(p)", ascending=False)



# Plot the distribution of Mann-Whitney U test p-values for each marker
def plot_mwu_distribution(df_mwu, top_n=None, figsize=(10,6), save_path=None):
    """
    Plots the -log10(p-value) of Mann-Whitney U tests for each marker, 
    indicating how significantly the distributions changed before vs after correction.
    """
    # Use -log10(p) for better visualization
    df_plot = df_mwu.sort_values("-log10(p)", ascending=False)
    
    if top_n is not None:
        df_plot = df_plot.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x="-log10(p)", y="marker", data=df_plot, color="steelblue")

    plt.xlabel("-log10(p-value)", fontsize=13, fontweight='bold')
    plt.ylabel("Marker", fontsize=13, fontweight='bold')
    plt.title("Statistical Significance of Distribution Change Before vs After Correction (Mann-Whitney U)", fontsize=16, fontweight='bold')
    
    # Add significance line at p=0.05 (-log10(0.05) ~ 1.3)
    plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05 significance')
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [SAVE] MWU plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# Calculate the KL divergence before and after correction for each marker across plates
def calculate_kd(df_before, df_after, panel, exclude_cols=None, n_bins=100, ctx=None):
    if exclude_cols is None:
        exclude_cols = list(ctx.meta_cols) + ["original_filename"]
        if getattr(ctx, "celltypes", None):
            exclude_cols.extend(ctx.celltypes)
        if getattr(ctx, "celltype_gating", None):
            exclude_cols.extend(ctx.celltype_gating)
        exclude_cols = list(dict.fromkeys(exclude_cols))
    
    # Subset panel
    df_before = df_before[df_before['panel'] == panel].dropna(axis=1, how="all").copy()
    df_after = df_after[df_after['panel'] == panel].dropna(axis=1, how="all").copy()
    
    # Select markers
    markers = [c for c in df_before.columns if c not in exclude_cols and c in df_after.columns]
    
    # Ensure markers are numeric and drop non-numeric ones
    markers = [m for m in markers if pd.api.types.is_numeric_dtype(df_before[m])]
    
    results = []

    for marker in markers:
        x_before = df_before[marker].dropna().values
        x_after = df_after[marker].dropna().values

        if len(x_before) == 0 or len(x_after) == 0:
            results.append({"marker": marker, "KD": np.nan})
            continue

        # Common range for histogram
        min_val = min(x_before.min(), x_after.min())
        max_val = max(x_before.max(), x_after.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Histogram
        p_before, _ = np.histogram(x_before, bins=bins, density=True)
        p_after, _ = np.histogram(x_after, bins=bins, density=True)

        # Normalize to ensure they are valid probability distributions
        p_before = p_before / (p_before.sum() + 1e-10)
        p_after = p_after / (p_after.sum() + 1e-10)

        # KL divergence
        kl = entropy(p_before + 1e-10, p_after + 1e-10)
        results.append({"marker": marker, "KD": kl})

    return pd.DataFrame(results).sort_values("KD", ascending=False)


# Plot the distribution of KL divergence for each marker
def plot_kd_distribution(df_kd, top_n=None, figsize=None, save_path=None):
    df_plot = df_kd.sort_values("KD", ascending=False)
    
    if top_n is not None:
        df_plot = df_plot.head(top_n)
    
    # If we have many markers, increase the height to fill the space
    if figsize is None:
        height = max(8, len(df_plot) * 0.25) # Reduced height per marker to make it more compact
        figsize = (10, height) # Normal width
    
    plt.figure(figsize=figsize)
    # Norml width of bars
    sns.barplot(x="KD", y="marker", data=df_plot, color="steelblue", width=0.6)

    plt.xlabel("KL Divergence (KD)", fontsize=14, fontweight='bold')
    plt.ylabel("Marker", fontsize=14, fontweight='bold')
    plt.title("Distribution of KL Divergence Before vs After Time Correction", fontsize=18, fontweight='bold', pad=25)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Tighten the x-axis to the maximum data value instead of a fixed 0.05
    max_kd = df_plot["KD"].max()
    plt.xlim(0, max(0.015, max_kd * 1.1)) 
    
    # Reference line for "small" divergence - Ensure it is added and visible zorder=5
    threshold = 0.01
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold (KD={threshold})', linewidth=2.5, zorder=5)
    
    plt.legend(fontsize=13, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [SAVED] KD distribution plot -> {save_path}")
    else:
        plt.show()

    plt.close()


def preprocess_files(
    files,
    ctx,
    verbose=True
):
    """
    Unified preprocessing pipeline for CyTOF files (RAW or TIME).
    """
    dfs = []
    marker_set = set()

    for f in files:
        if verbose:
            print("\n===================================")
            print(f"[LOAD] {os.path.basename(f)}")

        # Load
        df = pd.read_csv(f)
        if verbose:
            print(f"[DEBUG] raw shape = {df.shape}")

        # Harmonize markers
        df.columns = [
            harmonize_marker(c) if not c.startswith("OmiqFilter")
            else c.replace("OmiqFilter", "")
            for c in df.columns
        ]

        if verbose:
            print(f"[DEBUG] harmonized cols = {len(df.columns)}")

        # Apply schema
        df = apply_marker_schema(df, ctx)

        # Metadata
        df = add_metadata(df, f)

        # Cell type (safe)
        df = assign_cell_type(df, ctx)

        if verbose:
            print(f"[DEBUG] after cell type = {df.shape}")

        # Enforce schema
        df, markers = enforce_schema(df, ctx)

        if verbose:
            print(f"[DEBUG] final shape = {df.shape}")
            print(f"[DEBUG] markers = {len(markers)}")

        marker_set.update(markers)
        dfs.append(df)

    # ================= SUMMARY =================
    if verbose:
        print("\n================ SUMMARY ================")
        print(f"[INFO] total files = {len(files)}")
        print(f"[INFO] processed dfs = {len(dfs)}")
        print(f"[INFO] unique markers = {len(marker_set)}")
        print(f"[INFO] sample markers = {list(marker_set)[:10]}")
        print("=========================================\n")

    return dfs, marker_set

def load_and_preprocess_panel_time(panel_name, dataset_to_process, ctx, verbose=False):
    """
    Load and preprocess files for a specific panel AND dataset.
    """
    # Load metadata.xlsx
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_path = os.path.join(project_root, "metadata", "metadata.xlsx")
    metadata_df = pd.read_excel(metadata_path) if os.path.exists(metadata_path) else None

    if metadata_df is not None:
        metadata_df = metadata_df.copy()
        metadata_df["_file_stem"] = metadata_df["file_name"].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # 1. Handle RAW data
    raw_files_dict = get_indexed_files(ctx.data_dir, dataset_to_process, panel_name, metadata_df)
    raw_files = []
    for paths in raw_files_dict.values():
        raw_files.extend(paths)
        
    if not raw_files:
        if verbose: print(f"[WARN] No raw files found for {dataset_to_process} {panel_name}")
        return None, None, []

    dfs_b = []
    # Use metadata_df when calling add_metadata
    for f in raw_files:
        df = read_cytof_file(f)
        
        # If cell_type already exists, assumed to be pre-processed; skip markers harmonization and schema
        if "cell_type" not in df.columns:
            df.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in df.columns]
            df = apply_marker_schema(df, ctx)
            df = assign_cell_type(df, ctx)
            df, markers = enforce_schema(df, ctx)
        else:
            # Metadata is still required for indexing and directory structure
            # Identify markers by excluding known meta columns
            markers = [c for c in df.columns if c not in ctx.meta_cols]
            
        df = add_metadata(df, f, metadata_df=metadata_df)
        dfs_b.append(df)
    
    df_b = pd.concat(dfs_b, ignore_index=True)
    
    # 2. Handle Corrected files
    out_dir = ctx.output_dir
    time_dir = os.path.join(out_dir, "time_corrected")
    
    corrected_files_dict = get_indexed_files(time_dir, dataset_to_process, panel_name, metadata_df)
    corrected_files = []
    for paths in corrected_files_dict.values():
        corrected_files.extend(paths)
        
    if not corrected_files:
        if verbose: print(f"[WARN] No time-corrected files found for {dataset_to_process} {panel_name} in {time_dir}")
        return df_b, pd.DataFrame(), markers
    
    dfs_a = []
    for f in corrected_files:
        df = read_cytof_file(f)
        # Always use add_metadata for consistent mapping
        df = add_metadata(df, f, metadata_df=metadata_df)
        
        # Ensure cell_type exists (as these are processed files)
        if "cell_type" not in df.columns:
            # Fallback if somehow not present
            df = assign_cell_type(df, ctx)
            
        dfs_a.append(df)
    
    df_a = pd.concat(dfs_a, ignore_index=True)
    return df_b, df_a, markers



