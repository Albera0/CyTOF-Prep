import numpy as np
import pandas as pd

def time_correction(df, panel, ctx, window=500, slope_clip=(0.3, 3)):
    """
    Applies cell-type-specific time-drift correction to CyTOF data.

    For each (plate, sample, cell-type) group listed in ctx.celltype2correct, a rolling
    median of the stable reference channel (e.g. DNA) is used to estimate the
    acquisition-time signal drift. A per-cell slope is computed as
    baseline / rolling_median and clipped to slope_clip, then multiplied into every
    biological marker signal (adaptive thresholding via clipping).

    Args:
        df (pd.DataFrame): Input data containing 'Time', 'plate', 'celltypes',
            biological markers, and metadata columns.
        panel (str): The antibody panel to process.
        ctx (CyTOFContext): Pipeline config with meta_cols, stable_marker,
            celltype2correct, and optionally true_markers / celltypes / celltype_gating.
        window (int): Rolling-median window size (cells). Clipped to [50, 0.8*n_cells].
        slope_clip (tuple): (min, max) bounds for the correction slope.

    Returns:
        pd.DataFrame: Copy of the panel-filtered input with corrected marker values.
    """
    info_cols = ctx.meta_cols
    stable_marker = ctx.stable_marker
    celltype2correct = ctx.celltype2correct

    if "celltypes" not in df.columns or not celltype2correct:
        print("[TIME] 'celltypes' column or celltype2correct missing — skipping time correction.")
        return df

    sample_group_col = "original_filename" if "original_filename" in df.columns else "sample_name"
    df_panel = df[df['panel'] == panel].dropna(axis=1, how="all").copy()
    df_corrected = df_panel.copy()

    stable_markers = stable_marker if isinstance(stable_marker, list) else [stable_marker]
    stable_markers = [c for c in df_panel.columns if c in stable_markers]

    # Build the exclusion set: metadata + stable channels + gate/flag columns.
    # Gate columns (OmiqFilter-derived binary flags, ctx.celltypes, ctx.celltype_gating)
    # must NOT be time-corrected — they are 0/1 labels, not continuous signal.
    gate_cols = set()
    if getattr(ctx, 'celltypes', None):
        gate_cols.update(ctx.celltypes)
    if getattr(ctx, 'celltype_gating', None):
        gate_cols.update(ctx.celltype_gating)

    exclude = set(info_cols + stable_markers + [sample_group_col, "celltypes"]) | gate_cols

    # Restrict to true_markers when provided (biological signal only), otherwise fall
    # back to all numeric non-excluded columns.
    if getattr(ctx, 'true_markers', None):
        markers_to_correct = [
            c for c in ctx.true_markers
            if c in df_panel.columns and c not in exclude
            and pd.api.types.is_numeric_dtype(df_panel[c])
        ]
    else:
        markers_to_correct = [
            c for c in df_panel.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df_panel[c])
        ]

    print(f"[TIME] Stable={stable_markers} | Targets={len(markers_to_correct)} markers")

    # Validate that cell type names match before starting the heavy loop
    actual_types = set(df_panel['celltypes'].dropna().unique())
    matched_types = actual_types & set(celltype2correct)
    unmatched = set(celltype2correct) - actual_types
    if not matched_types:
        print(f"[WARN-TIME] No cell types in celltype2correct match the data!")
        print(f"  celltype2correct : {sorted(celltype2correct)}")
        print(f"  actual celltypes : {sorted(actual_types)}")
        print(f"  → Time correction will be a no-op. Check your config.")
    elif unmatched:
        print(f"[TIME] {len(unmatched)} celltype2correct entries not found in data (will be skipped): {sorted(unmatched)}")

    n_corrected_cells = 0

    for plate, df_plate in df_panel.groupby('plate'):
        print(f"\n[TIME] Plate={plate} | n_cells={len(df_plate)}")
        for sample, df_sample in df_plate.groupby(sample_group_col):
            print(f"[TIME]   Sample={sample} | n={len(df_sample)}")
            for celltype, df_cell in df_sample.groupby('celltypes'):
                if celltype not in celltype2correct:
                    continue

                # Adaptive window: bounded by [50, 80% of group size], always odd
                cell_specific_window = min(window, int(len(df_cell) * 0.8))
                cell_specific_window = max(50, cell_specific_window)
                if cell_specific_window % 2 == 0:
                    cell_specific_window += 1

                print(f"[TIME]     CellType={celltype} | n={len(df_cell)} | window={cell_specific_window}")

                # 1. Sort by acquisition time
                df_cell_sorted = df_cell.sort_values('Time').astype("float", errors='ignore')

                # 2. Rolling median of the stable channel
                df_cell_sorted[stable_markers] = df_cell_sorted[stable_markers].apply(
                    pd.to_numeric, errors='coerce'
                )
                runmeds = df_cell_sorted[stable_markers].rolling(
                    window=cell_specific_window, center=True, min_periods=1
                ).median()
                baseline = df_cell_sorted[stable_markers].mean().values

                # 3. Per-cell correction slope = baseline / rolling_median (clipped)
                slopes_df = pd.DataFrame(index=df_cell_sorted.index)
                for i, m in enumerate(stable_markers):
                    smoothed = runmeds[m].values
                    smoothed_safe = np.where(smoothed < 1e-6, np.nan, smoothed)
                    s = baseline[i] / smoothed_safe
                    s = np.where(np.isnan(s), 1.0, s)
                    s = np.clip(s, slope_clip[0], slope_clip[1])
                    slopes_df[m] = s

                # Average slope across stable markers (handles multi-channel references)
                slope_mean = slopes_df.mean(axis=1).astype(float).values

                # 4. Apply slope to biological markers and stable channel
                for m in markers_to_correct + stable_markers:
                    col_values = pd.to_numeric(df_cell_sorted[m], errors='coerce').values
                    df_corrected.loc[df_cell_sorted.index, m] = col_values * slope_mean

                n_corrected_cells += len(df_cell)

    frac = n_corrected_cells / max(len(df_panel), 1) * 100
    print(f"\n[TIME] Corrected {n_corrected_cells}/{len(df_panel)} cells ({frac:.1f}%)")
    if n_corrected_cells == 0 and matched_types:
        print("[WARN-TIME] Correction produced no output — check that 'celltypes' values are not all NaN.")

    return df_corrected

def arcsinh(df, markers, cofactor=5):
    """
    Apply arcsinh transformation to marker signals, commonly used for normalization in flow/mass cytometry data.
    """
    df = df.copy()
    # Filter only existing markers and ensure they are numeric
    valid_markers = [m for m in markers if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
    df[valid_markers] = np.arcsinh(df[valid_markers].astype(float) / cofactor)
    return df

def build_reference_stats(df, markers, anchor_sample, dataset, panel, ctx):
    """
    Builds reference marker distribution statistics for batch effect correction.
    
    This function analyzes the 'Anchor' sample in the 'Reference Plate' to establish 
    baseline intensity ranges (quantiles) for each marker. These statistics serve 
    as standard to which all other plates' anchor samples will be aligned.

    Args:
        df (pd.DataFrame): Data containing Arcsinh-transformed values from the reference plate.
        markers (list): List of biological markers to characterize.
        anchor_sample (str): Name of the sample used as the alignment anchor.
        dataset (str): Experimental dataset Identifier (e.g., 'D0').
        panel (str): Panel name.
        ctx (CyTOFContext): Global configuration context.

    Returns:
        dict: A dictionary where keys are markers and values contain 'qlo', 'qhi', and 'ref_lohi' ranges.
    """
    # Select data for the specified dataset, panel, reference plate, and anchor sample
    df_ref = df[
        (df["dataset"] == dataset) &
        (df["panel"] == panel) &
        (df["plate"] == ctx.ref_plate) &
        (df["sample_name"] == anchor_sample)
    ].copy()
    print(f"\n[REF] Building reference: dataset={dataset} panel={panel} anchor={anchor_sample}")
    print(f"[REF] n_cells={len(df_ref)}")
    if df_ref.empty:
        print(f"[WARN-REF] No cells found for ref_plate={ctx.ref_plate!r} anchor={anchor_sample!r}. Returning empty ref_stats.")
        return {}
    ref_stats = {}
    for m in markers:
        if m not in df_ref.columns:
            print(f"[WARN-REF] marker '{m}' not in reference data columns. Skipping.")
            continue
        # Calculate the proportion of zeros for this marker
        zero_prop = np.mean(df_ref[m].values == 0)
        # Lower quantile, avoid zero inflation
        qlo = min(zero_prop + 0.1, 0.8)
        # Use qlo~95th percentile as reference range
        ref_lohi = np.percentile(df_ref[m], [qlo * 100, 95])
        print(f"[REF] {m} | zero={zero_prop:.3f} qlo={qlo:.3f} refLo={ref_lohi[0]:.3f} refHi={ref_lohi[1]:.3f}")
        ref_stats[m] = {
            "qlo": qlo,
            "qhi": 0.95,
            "ref_lohi": ref_lohi
        }
    return ref_stats

def batch_correct_plate(df_plate, markers, ref_stats, anchor_sample):
    """
    Normalizes a single plate by aligning its anchor sample distribution to the reference.
    
    Using the statistics generated by 'build_reference_stats', this function applies 
    a linear transformation (scale and shift) to each marker in the plate. The goal 
    is to make the anchor sample in this plate look like the reference anchor sample, 
    thereby removing technical batch effects.

    Args:
        df_plate (pd.DataFrame): Data for the specific plate to be corrected.
        markers (list): List of biological markers to normalize.
        ref_stats (dict): Reference distribution statistics.
        anchor_sample (str): Biological sample name used as the anchor in this plate.

    Returns:
        pd.DataFrame: A copy of the input plate data with batch effects removed.
    """
    df = df_plate.copy()
    # Find the anchor sample for the current plate
    df_anchor = df_plate[df_plate["sample_name"] == anchor_sample]
    print(f"\n[BATCH] Start plate correction | anchor={anchor_sample}")
    print(f"[BATCH] n_cells={len(df_plate)} | n_anchor={len(df_anchor)}")
    
    for m in markers:
        # Skip markers not in reference
        if m not in ref_stats:
            print(f"[WARN-BATCH] missing in ref_stats: {m}")
            continue
        # Skip markers not in data
        if m not in df.columns:
            print(f"[WARN-REF] missing marker in reference: {m}")
            continue
            
        # Ensure the column is numeric and drop NaNs for quantile calculation
        anchor_values = pd.to_numeric(df_anchor[m], errors='coerce').dropna()
        
        # Check if the marker has valid signal in the anchor sample
        # If the anchor for this specific plate has no signal (ghost column), skip correction
        if len(anchor_values) == 0 or (anchor_values == 0).all():
            print(f"[BATCH] {m} | No valid signal in anchor sample for this plate. Keeping Arcsinh values.")
            # Convert to numeric to ensure consistency, but don't transform
            df[m] = pd.to_numeric(df[m], errors='coerce')
            continue

        qlo = ref_stats[m]["qlo"]
        qhi = ref_stats[m]["qhi"]
        refLo, refHi = ref_stats[m]["ref_lohi"]
        
        # Calculate anchor sample quantiles
        batLo, batHi = np.percentile(anchor_values, [qlo * 100, qhi * 100])
        
        # If any quantile calculation returned NaN or if quantiles are equal, do not scale
        if np.isnan(batLo) or np.isnan(batHi) or batHi == batLo:
            print(f"[BATCH] {m} | Identical or NaN quantiles in anchor. Keeping Arcsinh values.")
            scale, shift = 1.0, 0.0
        else:
            scale = (refHi - refLo) / (batHi - batLo)
            # Clip scale to a reasonable range (0.3-3x) to prevent inflation of noise/faint signals
            scale = np.clip(scale, 0.3, 3.0)
            shift = refLo - batLo * scale
            
        print(f"[BATCH] {m} | batLo={batLo:.3f} batHi={batHi:.3f} scale={scale:.3f} shift={shift:.3f}")
        # Apply linear transformation to all cells. Ensure numeric and preserve data.
        target_values = pd.to_numeric(df[m], errors='coerce')
        df[m] = np.maximum(target_values * scale + shift, 0)
    return df

