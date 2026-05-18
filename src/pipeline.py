import os
import sys
import json
import matplotlib
# Force non-interactive backend only when running as a plain script, not in Jupyter
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import fcswrite
import gc
import traceback
from glob import glob
from collections import defaultdict
from .utils import *
from .correction import *
from .time_correction_visualization import *
from .batch_correction_visualization import *


def _load_pheno_markers(project_root, dataset, panel, available_cols):
    """
    Look for pheno_markers in study config JSONs saved under results/.
    Searches for any JSON under results/ whose cytof_config.pheno_markers (or top-level
    pheno_markers) is set and whose filename references the given dataset and panel.
    Falls back to None if nothing is found.
    """
    results_dir = os.path.join(project_root, "results")
    if not os.path.isdir(results_dir):
        return None

    # Collect candidate JSONs: prefer ones whose name contains the dataset (and panel)
    candidates = glob(os.path.join(results_dir, "**", "study_config_*.json"), recursive=True)
    # Sort so dataset+panel-specific files come before more generic ones
    def _score(p):
        name = os.path.basename(p).lower()
        return (dataset.lower() not in name, panel.lower() not in name)
    candidates = sorted(candidates, key=_score)

    for cfg_path in candidates:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            pheno = (cfg.get("cytof_config", {}).get("pheno_markers")
                     or cfg.get("pheno_markers"))
            if pheno:
                filtered = [m for m in pheno if m in available_cols]
                if filtered:
                    print(f"  [UMAP] pheno_markers loaded from {os.path.basename(cfg_path)}: {filtered}")
                    return filtered
        except Exception:
            continue
    return None

import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*dtype argument will be deprecated.*")
warnings.filterwarnings("ignore", message="No data for colormapping provided via 'c'")
warnings.filterwarnings("ignore", message="Tight layout not applied")

def load_data_from_omiq(data_dir, metadata_df, ctx, dataset=None, panel=None, celltype_col=None):
    """
    Load CyTOF files exported from OMIQ and return an annotated cell matrix.

    Reads all CSV/FCS files in data_dir, matches each file against metadata_df to
    determine sample_name, plate, dataset, panel, and any other ctx.meta_cols columns.
    Harmonizes marker names, applies the marker schema, assigns cell types, and
    concatenates everything into a single DataFrame with one row per cell.

    Args:
        data_dir (str): Directory containing CSV or FCS files exported from OMIQ.
        metadata_df (pd.DataFrame): Metadata table with at minimum: file_name,
            sample_name, plate, dataset, panel. Any extra columns listed in
            ctx.meta_cols are also propagated to the output matrix.
        ctx (CyTOFContext): Pipeline configuration context.
        dataset (str, optional): If provided, only files whose metadata row has
            this dataset value are loaded (e.g. 'D0').
        panel (str, optional): If provided, only files whose metadata row has
            this panel value are loaded.
        celltype_col (str, optional): Name of the column in the input CSV that
            contains cell type labels (e.g. 'Population' or 'celltypes'). When
            provided, that column is renamed to 'celltypes' and used directly,
            skipping the OmiqFilter-based gating assignment. If None, cell types
            are derived from ctx.celltype_gating binary flag columns.

    Returns:
        tuple[pd.DataFrame, list[str]]: (matrix, markers) where matrix is the
            full annotated cell DataFrame and markers is the list of biological
            marker column names used for batch correction.
    """
    dfs = []
    markers = None
    n_skipped = 0

    for f in glob(os.path.join(data_dir, "*")):
        if not f.lower().endswith(('.csv', '.fcs')):
            continue
        file_name = os.path.basename(f)

        match = metadata_df[metadata_df["file_name"] == file_name]
        if match.empty:
            name_no_ext = os.path.splitext(file_name)[0]
            match = metadata_df[metadata_df["file_name"] == name_no_ext]

        if match.empty:
            print(f"[WARN] Skipping file not in metadata: {file_name}")
            n_skipped += 1
            continue

        row = match.iloc[0]
        file_dataset = str(row.get("dataset", ""))
        file_panel = str(row.get("panel", ""))

        if dataset is not None and file_dataset != dataset:
            continue
        if panel is not None and file_panel != panel:
            continue

        df = read_cytof_file(f)

        # --- Cell type annotation (BEFORE harmonization) ---
        # Must happen first because harmonization strips "OmiqFilter" from column names.
        cell_type_series = None
        # Case 1: user provided the exact column name (e.g. a single "OmiqFilter" string column)
        if celltype_col is not None and celltype_col in df.columns:
            cell_type_series = df[celltype_col].reset_index(drop=True)
            print(f"  [CT] column '{celltype_col}' → values: {cell_type_series.dropna().unique()[:5]}")
            df["celltypes"] = cell_type_series.values
        # Case 2: auto-detect multiple OmiqFilter<gate> binary flag columns.
        # Restrict to gates listed in ctx.celltype_gating so broad hierarchical gates
        # (e.g. OmiqFilteralive, OmiqFilterbulk_leukocytes) are never assigned instead
        # of the specific cell-type gates the user cares about.
        elif any(c.startswith("OmiqFilter") for c in df.columns):
            _allowed = set(ctx.celltype_gating) if ctx.celltype_gating else None
            gate_cols = {
                c.replace("OmiqFilter", ""): c
                for c in df.columns
                if c.startswith("OmiqFilter")
                and (_allowed is None or c.replace("OmiqFilter", "") in _allowed)
            }
            print(f"  [CT] auto-detected {len(gate_cols)} OmiqFilter gate columns: {list(gate_cols.keys())[:5]}")
            def _is_positive(val):
                if val is None:
                    return False
                if isinstance(val, bool):
                    return val
                try:
                    return float(val) > 0
                except (ValueError, TypeError):
                    return str(val).strip().lower() == "true"
            def _get_type(row):
                for name, col in gate_cols.items():
                    if _is_positive(row.get(col)):
                        return name
                return None
            cell_type_series = df.apply(_get_type, axis=1).reset_index(drop=True)
            n_assigned = cell_type_series.notna().sum()
            print(f"  [CT] {n_assigned}/{len(df)} cells assigned a cell type")
        elif celltype_col is not None:
            avail = [c for c in df.columns if "omiq" in c.lower() or "gate" in c.lower() or "filter" in c.lower()]
            print(f"  [WARN] celltype_col='{celltype_col}' not found. Possible gate columns: {avail[:10]}")

        # --- Harmonize marker names (strips OmiqFilter prefix from remaining cols) ---
        df.columns = [
            harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter", "")
            for c in df.columns
        ]
        df = df.reset_index(drop=True)
        df = apply_marker_schema(df, ctx)

        # Attach cell_type — never filters or drops any rows
        if cell_type_series is not None:
            df["celltypes"] = cell_type_series.values

        df, file_markers = enforce_schema(df, ctx)
        df = add_metadata(df, f, metadata_df=metadata_df, ctx=ctx)

        # Accumulate markers across files so multi-panel loads capture all markers.
        if markers is None:
            markers = list(file_markers)
        else:
            for _m in file_markers:
                if _m not in markers:
                    markers.append(_m)

        dfs.append(df)
        print(f"[LOAD] {file_name} | n_cells={len(df)}")

    if not dfs:
        raise ValueError(
            f"No files loaded from {data_dir!r}. Check data_dir and metadata_df."
        )

    matrix = pd.concat(dfs, ignore_index=True)
    print(f"\n[LOAD] Total: {len(matrix)} cells from {len(dfs)} files ({n_skipped} skipped)")
    return matrix, markers or []


def apply_correction(df, markers, anchor_name, ctx):
    """
    Apply time drift correction and batch correction to an annotated CyTOF matrix.

    Stages (identical to run_cytof_correction):
    1. Cell-type-specific time drift correction (stable-marker rolling-median slope).
    2. Arcsinh transformation (cofactor = 5) on biological markers only.
    3. Quantile-based batch correction (anchor-sample alignment to reference plate).

    Args:
        df (pd.DataFrame): Annotated cell matrix from load_data_from_omiq.
            Must contain 'panel', 'plate', 'sample_name', 'dataset', 'celltypes'.
        markers (list[str]): Biological marker column names (from load_data_from_omiq).
            Only these are used for batch correction (gate/flag columns excluded).
        anchor_name (str): 'sample_name' value of the inter-plate anchor sample.
        ctx (CyTOFContext): Pipeline config (ref_plate, meta_cols, celltype2correct…).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            (intermediate, corrected)
            - intermediate : time-corrected + arcsinh, BEFORE batch correction.
                             Use as "After" in time-correction visualization, and
                             as the saved pre-batch baseline for diagnostics.
            - corrected    : fully corrected (time + arcsinh + batch).
    """
    corrected_panels = []
    intermediate_panels = []

    # Columns that are OmiqFilter-derived binary flags or gate labels — must not be
    # arcsinh-transformed (they are 0/1 integers, not continuous signal).
    gate_cols = set()
    if getattr(ctx, 'celltypes', None):
        gate_cols.update(ctx.celltypes)
    if getattr(ctx, 'celltype_gating', None):
        gate_cols.update(ctx.celltype_gating)
    _non_signal = set(ctx.meta_cols) | {"celltypes"} | gate_cols

    # Determine unique (dataset, panel) groups
    dataset_col_present = "dataset" in df.columns
    group_keys_list = (
        df[["dataset", "panel"]].drop_duplicates().values.tolist()
        if dataset_col_present
        else [[None, p] for p in df["panel"].unique()]
    )

    for dataset, panel in group_keys_list:
        if dataset is not None:
            df_group = df[(df["dataset"] == dataset) & (df["panel"] == panel)].copy()
        else:
            df_group = df[df["panel"] == panel].copy()
            dataset = ""

        # Restrict batch-correction markers to those present in this panel's data.
        # Biological markers only — gate columns are excluded via the markers list
        # (which comes from load_data_from_omiq / enforce_schema).
        valid_markers = [m for m in markers if m in df_group.columns]
        if len(valid_markers) < len(markers):
            missing = set(markers) - set(valid_markers)
            print(f"[WARN] {len(missing)} marker(s) absent from panel '{panel}' and skipped: "
                  f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

        # ── Step 1: Cell-type-specific time drift correction ─────────────────
        # Gate/flag columns are NOT corrected (handled inside time_correction).
        df_group = time_correction(df_group, panel, ctx)

        # ── Step 2: Arcsinh transformation ───────────────────────────────────
        # Applied only to continuous signal columns; gate columns are excluded.
        all_data_cols = [c for c in df_group.columns if c not in _non_signal]
        df_group = arcsinh(df_group, all_data_cols)

        # Capture time-corrected + arcsinh state (before batch correction).
        intermediate_panels.append(df_group.copy())

        # ── Step 3: Batch correction ──────────────────────────────────────────
        ref_stats = build_reference_stats(df_group, valid_markers, anchor_name, dataset, panel, ctx)

        corrected_plates = []
        for plate, df_plate in df_group.groupby("plate"):
            if plate == ctx.ref_plate:
                # Reference plate: time + arcsinh only, no quantile alignment needed.
                corrected_plates.append(df_plate)
                continue

            df_anchor = df_plate[df_plate["sample_name"] == anchor_name]
            if df_anchor.empty or not ref_stats:
                reason = "no anchor in plate" if df_anchor.empty else "no ref stats"
                print(f"[SKIP-BATCH] plate={plate} ({reason}) — keeping time-corrected values.")
                corrected_plates.append(df_plate)
            else:
                corrected_plates.append(
                    batch_correct_plate(df_plate, valid_markers, ref_stats, anchor_name)
                )

        corrected_panels.append(pd.concat(corrected_plates, ignore_index=True))

    intermediate = pd.concat(intermediate_panels, ignore_index=True)
    corrected    = pd.concat(corrected_panels,    ignore_index=True)
    return intermediate, corrected


def run_cytof_correction(dataset_to_process, config_path=None, metadata_path=None, config_dict=None, metadata_df=None):
    """
    Main entry point for the sequential CyTOF normalization pipeline.

    Workflow:
    1. TIME CORRECTION: Removal of intra-run signal drift (Time-based).
    2. BATCH CORRECTION: Inter-plate alignment using anchor samples.
    3. SAVING: Exports clean CSV files to 'Data/Correction'.

    This ensures that cells from different acquisition days and technical batches 
    can be directly compared in downstream analysis.

    Args:
        dataset_to_process (str): ID of the dataset to run (e.g., 'D0').
        config_path (str, optional): Custom path to meta.json.
        metadata_path (str, optional): Custom path to SampleSheet.csv.
        config_dict (dict, optional): Custom configuration dictionary.
        metadata_df (pd.DataFrame, optional): Custom metadata DataFrame.
    """

    # Create CyTOFContext from configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if config_dict is not None:
        config = get_config(config_dict=config_dict)
    else:
        if config_path is None:
            config_path = os.path.join(project_root, "metadata", "meta.json")
        config = get_config(config_path=config_path)
        
    ctx = CyTOFContext(config)

    # ===== STEP 0: LOADING METADATA =====
    if metadata_df is not None:
        print("\n===== STEP 0: LOADING METADATA =====")
    else:
        if metadata_path is None:
            metadata_path = os.path.join(project_root, "metadata", "metadata.xlsx")

        if metadata_path is not None and os.path.exists(metadata_path):
            print("\n===== STEP 0: LOADING METADATA =====")
            metadata_df = pd.read_excel(metadata_path)
        else:
            print(f"[WARN] Metadata file not found at {metadata_path}.")

    # Index files by (dataset, panel, plate)
    file_index = defaultdict(list)
    if metadata_df is None:
        raise ValueError("metadata.xlsx is required but was not found in the metadata/ directory.")

    for f in glob(os.path.join(ctx.data_dir, "*")):
        if not f.lower().endswith(('.csv', '.fcs')):
            continue
        file_name = os.path.basename(f)
        
        # Strictly use metadata.xlsx to find dataset, panel, plate
        match = metadata_df[metadata_df["file_name"] == file_name]
        if match.empty:
            name_no_ext = os.path.splitext(file_name)[0]
            match = metadata_df[metadata_df["file_name"] == name_no_ext]
        
        if not match.empty:
            row = match.iloc[0]
            dataset = str(row.get("dataset", ""))
            panel = str(row.get("panel", ""))
            plate = str(row.get("plate", ""))
            if dataset != dataset_to_process:
                continue
            file_index[(dataset, panel, plate)].append(f)
        else:
            print(f"[WARN] Skipping file not found in metadata: {file_name}")

    ref_stats_dict = {}

    # STEP 1: Build reference statistics from the reference plate
    print("\n===== STEP 1: BUILD REFERENCE =====")
    for (dataset, panel, plate), files in file_index.items():
        if plate != ctx.ref_plate:
            continue
        print(f"[REF] {dataset} | {panel} | {plate}")

        dfs = []
        for f in files:
            df = read_cytof_file(f)
            # If celltypes already exists, assumed to be pre-processed
            if "celltypes" not in df.columns:
                df.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in df.columns]
                df = apply_marker_schema(df, ctx)
                df = assign_cell_type(df, ctx)
                df, markers = enforce_schema(df, ctx)
            else:
                df.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in df.columns]
                df = apply_marker_schema(df, ctx)
                df, markers = enforce_schema(df, ctx)

            df = add_metadata(df, f, metadata_df=metadata_df, ctx=ctx)
            dfs.append(df)

        # Concatenate all files for this plate
        df_plate = pd.concat(dfs, ignore_index=True)
        # Time drift correction
        df_plate = time_correction(df_plate, panel, ctx)
        
        # Identify all data markers (for Arcsinh)
        all_markers = [c for c in df_plate.columns if c not in ctx.meta_cols]
        # Arcsinh transformation for ALL markers (targets + extra)
        df_plate = arcsinh(df_plate, all_markers)

        anchor_sample = ctx.anchors.get(dataset)
        # Build reference statistics for this (dataset, panel)
        ref_stats_dict[(dataset, panel)] = build_reference_stats(df_plate, markers, anchor_sample, dataset, panel, ctx)

    # STEP 2: Batch correction for all plates
    print("\n===== STEP 2: BATCH CORRECTION =====")

    for (dataset, panel, plate), files in file_index.items():

        print(f"\n=== {dataset} | {panel} | {plate} ===")
        print(f"[LOAD] #files={len(files)}")

        dfs = []
        for f in files:
            df = read_cytof_file(f)
            # If celltypes already exists, assumed to be pre-processed
            if "celltypes" not in df.columns:
                df.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in df.columns]
                df = apply_marker_schema(df, ctx)
                df = assign_cell_type(df, ctx)
                df, markers = enforce_schema(df, ctx)
            else:
                markers = [c for c in df.columns if c not in ctx.meta_cols and c != "celltypes"]

            df = add_metadata(df, f, metadata_df=metadata_df, ctx=ctx)
            dfs.append(df)

        # Concatenate all files for this plate
        df_plate = pd.concat(dfs, ignore_index=True)

        print(f"[LOAD] total cells={len(df_plate)}")
        print(f"[LOAD] markers={len(markers)}")

        # Time drift correction and arcsinh transformation
        df_plate = time_correction(df_plate, panel, ctx)

        # SAVE RAW TIME-CORRECTED RESULTS (FOR VISUALIZATION)
        time_dir = os.path.join(ctx.output_dir, "time_corrected")
        os.makedirs(time_dir, exist_ok=True)

        for original_name, df_sample in df_plate.groupby("original_filename"):
            # Drop helper column and save as Raw CSV
            df_to_save = df_sample.drop(columns=["original_filename"], errors="ignore")
            base, _ = os.path.splitext(original_name)
            out_name = f"{base}.csv"
            out_path = os.path.join(time_dir, out_name)
            df_to_save.to_csv(out_path, index=False)
            print(f"[SAVE] time_corrected result as CSV={out_name} | path={out_path}")

        # ARCSINH TRANSFORMATION (FOR BATCH CORRECTION)
        # Identify all data markers (for Arcsinh)
        all_markers = [c for c in df_plate.columns if c not in ctx.meta_cols]
        # Arcsinh transformation in memory
        df_plate = arcsinh(df_plate, all_markers)

        skip_flag = ""
        if plate == ctx.ref_plate:
            # Reference plate does not need batch correction
            df_corrected = df_plate
        else:
            # Get reference statistics for this (dataset, panel)
            ref_stats = ref_stats_dict.get((dataset, panel), None)
            anchor_sample = ctx.anchors.get(dataset)
            df_anchor = df_plate[df_plate["sample_name"] == anchor_sample] if anchor_sample else pd.DataFrame()
            if df_anchor.empty or not ref_stats:
                reason = "no anchor" if df_anchor.empty else "no ref stats"
                print(f"[SKIP-BATCH] No anchor in {dataset}|{panel}|{plate} ({reason}) | using uncorrected data")
                df_corrected = df_plate
                skip_flag = "_UNCORRECTED"
            else:
                # Apply batch correction using reference statistics
                df_corrected = batch_correct_plate(df_plate, markers, ref_stats, anchor_sample)

        # SAVE: Write corrected data for each sample to output directory
        for original_name, df_sample in df_corrected.groupby("original_filename"):
            # Drop the helper column before saving
            df_to_save = df_sample.drop(columns=["original_filename"], errors="ignore")
            # Force CSV extension and handle skip_flag
            base, _ = os.path.splitext(original_name)
            if skip_flag:
                out_name = f"{base}{skip_flag}.csv"
            else:
                out_name = f"{base}.csv"
                
            out_path = os.path.join(ctx.output_dir, out_name)
            os.makedirs(ctx.output_dir, exist_ok=True)
            df_to_save.to_csv(out_path, index=False)
            print(f"[SAVE] corrected result as CSV={out_name} | path={out_path}")



# For reference, here is the time_correction visualization function
def run_time_analysis(datasets=None, plates_to_visualize=None, verbose=True, config_path=None):
    """
    Executes the diagnostic pipeline for evaluating time-drift correction.

    Generates visualizations to assess whether signal drift within acquisition
    runs has been successfully removed without distorting biology.

    Args:
        datasets (list/str, optional): Names of datasets to analyze.
        plates_to_visualize (list, optional): Specific plates to focus on.
        verbose (bool): Whether to print progress details.
        config_path (str, optional): Path to meta.json; defaults to metadata/meta.json.
    """
    # Create CyTOFContext from configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(project_root, "metadata", "meta.json")
    config = get_config(config_path=config_path)
    ctx = CyTOFContext(config)

    # Load metadata.xlsx
    metadata_path = os.path.join(project_root, "metadata", "metadata.xlsx")
    metadata_df = pd.read_excel(metadata_path) if os.path.exists(metadata_path) else None

    # Initial scan
    raw_files = glob(os.path.join(ctx.data_dir, "*"))
    combinations = set()
    for f in raw_files:
        if not f.lower().endswith(('.csv', '.fcs')):
            continue
        file_name = os.path.basename(f)
        if metadata_df is not None:
            match = metadata_df[metadata_df["file_name"] == file_name]
            if match.empty:
                name_no_ext = os.path.splitext(file_name)[0]
                match = metadata_df[metadata_df["file_name"] == name_no_ext]
            
            if not match.empty:
                row = match.iloc[0]
                dataset = str(row.get("dataset", ""))
                panel = str(row.get("panel", ""))
                if datasets is None or dataset == datasets or (isinstance(datasets, list) and dataset in datasets):
                    combinations.add((dataset, panel))
                continue

        continue

    combinations = sorted(list(combinations))
    results_summary = []
    
    # For cross-panel variance comparison, we will store the variance results
    all_variance_results = {}

    print(f"\n{'='*60}")
    print(f"STARTING TIME ANALYSIS")
    print(f"{'='*60}\n")

    for dataset, panel in combinations:
        print(f"\n>>> Processing Group: {dataset} | {panel}")
        
        try:
            # 1. Load and Preprocess
            df_b, df_a, markers = load_and_preprocess_panel_time(panel, dataset, ctx, verbose=verbose)
            if df_b is None or df_b.empty or df_a.empty:
                continue
            
            # Results root: results/{dataset}/{panel}/
            base_panel_dir = os.path.join(ctx.output_dir, "results", dataset, panel)
            
            # Subfolders:
            path_Bf_Time = os.path.join(base_panel_dir, "BfTime")              # Before trends
            path_Af_Time = os.path.join(base_panel_dir, "AfTime")              # After trends
            path_TimeCorrection = os.path.join(base_panel_dir, "TimeCorrection") # Density plots
            path_TimeCorrectionSignal = os.path.join(base_panel_dir, "TimeCorrectionSignal") # Distribution plots
            path_Variance = os.path.join(base_panel_dir, "Time_Variance")          # PCA/Stats results
            path_UMAP = os.path.join(base_panel_dir, "Time_UMAP")                  # UMAP plots
            
            for d in [path_Bf_Time, path_Af_Time, path_TimeCorrection, path_TimeCorrectionSignal, path_Variance, path_UMAP]:
                os.makedirs(d, exist_ok=True)
            
            # 2. PCA Variance Analysis
            print(f"[ANALYZE] Running variation_pca...")
            # Use columns that actually exist in the data
            factors = [c for c in ["plate", "sample_name", "celltypes", "Time"] if c in df_b.columns]
            var_b = variation_pca(df_b, markers=markers, group_col="condition", info_cols=factors, panel=panel)
            var_a = variation_pca(df_a, markers=markers, group_col="condition", info_cols=factors, panel=panel)
            all_variance_results[(dataset, panel)] = {"before": var_b, "after": var_a}
            
            var_b.to_csv(os.path.join(path_Variance, f"variance_before_{dataset}_{panel}.csv"))
            var_a.to_csv(os.path.join(path_Variance, f"variance_after_{dataset}_{panel}.csv"))

            # 2.2 Marker-level Variance Boxplots (Time Correction)
            # Group by sample_name so each group is one run; Time is the only factor.
            # This isolates within-sample time drift — the signal time correction actually removes.
            print(f"[ANALYZE] Generating Marker-level variance boxplots (Time)...")
            mvar_b = variation_marker(df_b, markers=markers, group_col='sample_name', info_cols=['Time'], panel=panel)
            mvar_a = variation_marker(df_a, markers=markers, group_col='sample_name', info_cols=['Time'], panel=panel)

            # Save the raw marker-level variance data to CSV
            mvar_b.to_csv(os.path.join(path_Variance, f"variance_marker_raw_before_{dataset}_{panel}_Time.csv"), index=False)
            mvar_a.to_csv(os.path.join(path_Variance, f"variance_marker_raw_after_{dataset}_{panel}_Time.csv"), index=False)

            plot_variance_marker_boxplot(mvar_b, mvar_a, save_path=os.path.join(path_Variance, f"variance_marker_boxplot_{dataset}_{panel}_Time.png"), use_log=False)

            # 3. UMAP
            print(f"[ANALYZE] Generating UMAP...")
            # Resolve phenotypic markers for UMAP
            umap_markers_time = None
            if ctx.pheno_markers:
                umap_markers_time = [m for m in ctx.pheno_markers if m in df_b.columns]
            if not umap_markers_time:
                umap_markers_time = _load_pheno_markers(project_root, dataset, panel, df_b.columns)
            if not umap_markers_time:
                umap_markers_time = markers
            adata = umap_by_celltype(df_b, df_a, panel=panel, markers=umap_markers_time, celltype_col="celltypes", subsample=50000)
            _savefig(os.path.join(path_UMAP, f"umap_comparison_{dataset}_{panel}_TimeCorrection.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Shared Markers (Reference)
            shared_markers_list = identify_shared_markers(df_b, info_col="panel")
            with open(os.path.join(base_panel_dir, "shared_markers.txt"), "w") as f_out:
                f_out.write("\n".join(shared_markers_list))

            # # 5. Marker Trends 
            # print(f"[ANALYZE] Running marker_trends...")
            # all_plates = df_b['plate'].unique()
            # if plates_to_visualize:
            #     plates_to_run = [p for p in all_plates if p in plates_to_visualize]
            #     print(f"[INFO] Filtering trends visualization to: {plates_to_run}")
            # else:
            #     plates_to_run = all_plates

            # for plate in plates_to_run:
            #     # Before trends
            #     marker_trends(df_b, plate=plate, panel=panel, markers=markers, save_dir=path_Bf_Time)
            #     # After trends
            #     marker_trends(df_a, plate=plate, panel=panel, markers=markers, save_dir=path_Af_Time)

            # 6. Time Effect Density
            print(f"[ANALYZE] Running plot_time_effect_density...")
            plot_time_effect_density(before_df=df_b, after_df=df_a, panel=panel, markers=markers, save_dir=path_TimeCorrection, subsample=100000)

            # 7. Marker Correlation
            print(f"[ANALYZE] Running marker correlation preservation plots (Time)...")
            path_TimeMarkerCorr = os.path.join(base_panel_dir, "Time_MarkerCorrelation")
            os.makedirs(path_TimeMarkerCorr, exist_ok=True)
            marker_corr(df_before=df_b, df_after=df_a, markers=markers, save_dir=path_TimeMarkerCorr)

            # 8. Stats Calculation (KL)
            print(f"[ANALYZE] Calculating stats (KD and MWU)...")
            kd_df = calculate_kd(df_before=df_b, df_after=df_a, panel=panel, ctx=ctx)
            plot_kd_distribution(kd_df, save_path=os.path.join(path_Variance, f"kd_distribution_{dataset}_{panel}.png"))
            
            # 9. Signal Distributions (Separate and Joint) - Now using pre-calculated kd_df
            print(f"[ANALYZE] Running signal distribution plots with pre-calculated KD...")
            # plot_signal_distributions(df_before=df_b, df_after=df_a, panel=panel, markers=markers, save_dir=path_TimeCorrectionSignal, subsample=50000)
            plot_signal_distributions_joint(df_before=df_b, df_after=df_a, panel=panel, markers=markers, 
                                            save_dir=path_TimeCorrectionSignal, subsample=10000, kd_df=kd_df)

            # mwu_df = calculate_mwu(df_b, df_a, panel, ctx=ctx)
            # plot_mwu_distribution(mwu_df, save_path=os.path.join(path_Variance, f"mwu_significance_{dataset}_{panel}.png"))

            results_summary.append({"dataset": dataset, "panel": panel, "status": "Success"})
            del df_b, df_a, var_b, var_a
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            results_summary.append({"dataset": dataset, "panel": panel, "status": f"Error: {str(e)}"})

    summary_df = pd.DataFrame(results_summary)

    # POST-PROCESS: Cross-Panel Variance Comparison
    # Group by dataset to compare panels within the same dataset
    if not summary_df.empty:
        for dataset in summary_df[summary_df['status'] == 'Success']['dataset'].unique():
            success_keys = [k for k, v in all_variance_results.items() if k[0] == dataset]
            
            if len(success_keys) >= 2:
                try:
                    p1, p2 = success_keys[0], success_keys[1]
                    print(f"\n[POST-PROCESS] Generating cross-panel variance plot for {dataset} ({p1[1]} vs {p2[1]})...")
                    
                    # Target directory: results/{dataset}/Variance/
                    variance_dir = os.path.join(ctx.output_dir, "results", dataset, "Variance")
                    os.makedirs(variance_dir, exist_ok=True)
                    
                    save_path = os.path.join(variance_dir, f"{dataset}_cross_panel_variance_comparison_time_correction.png")
                    
                    # Create plot
                    plot_variance_bar(
                        all_variance_results[p1]["before"], all_variance_results[p1]["after"],
                        all_variance_results[p2]["before"], all_variance_results[p2]["after"],
                        save_path=save_path
                    )
                    
                    plt.close()
                except Exception as e:
                    print(f"  [WARN] Cross-panel plotting failed for {dataset}: {e}")

    print(f"\n{'='*60}\nTime ANALYSIS COMPLETE\n{'='*60}\n")
    
    return summary_df, all_variance_results




# For reference, here is the batch_correction visualization function
def run_batch_analysis(datasets=None, verbose=True, subsample_umap=50000, config_path=None):
    """
    Main entry point for batch processing - Batch Correction Visualization.
    Visualizes [Time-corrected + arcsinh] vs [Batch-corrected].

    Args:
        datasets (list/str, optional): Names of datasets to analyze.
        verbose (bool): Whether to print progress details.
        subsample_umap (int): Max cells for UMAP subsampling.
        config_path (str, optional): Path to meta.json; defaults to metadata/meta.json.
    """
    # Create CyTOFContext from configuration if not provided
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(project_root, "metadata", "meta.json")
    config = get_config(config_path=config_path)
    ctx = CyTOFContext(config)

    # Load metadata.xlsx
    metadata_path = os.path.join(project_root, "metadata", "metadata.xlsx")
    metadata_df = pd.read_excel(metadata_path) if os.path.exists(metadata_path) else None

    # Initial scan
    raw_files = glob(os.path.join(ctx.data_dir, "*"))
    combinations = set()
    for f in raw_files:
        if not f.lower().endswith(('.csv', '.fcs')):
            continue
        file_name = os.path.basename(f)
        if metadata_df is not None:
            match = metadata_df[metadata_df["file_name"] == file_name]
            if match.empty:
                name_no_ext = os.path.splitext(file_name)[0]
                match = metadata_df[metadata_df["file_name"] == name_no_ext]
            
            if not match.empty:
                row = match.iloc[0]
                dataset = str(row.get("dataset", ""))
                panel = str(row.get("panel", ""))
                if datasets is None or dataset == datasets or (isinstance(datasets, list) and dataset in datasets):
                    combinations.add((dataset, panel))
                continue

        continue

    combinations = sorted(list(combinations))
    results_summary = []
    
    # Store variance and other results for cross-panel analysis
    all_variance_results = {}
    all_celltype_variance = {}
    
    # Store data for cross-panel shared marker analysis
    all_dfs = {}
    all_markers = {}
    all_umap_markers = {}   # pheno_markers-filtered, used for all UMAPs including cross-panel

    print(f"\n{'='*60}")
    print(f"STARTING BATCH CORRECTION ANALYSIS")
    print(f"{'='*60}\n")

    for dataset, panel in combinations:
        print(f"\n>>> Processing Group: {dataset} | {panel}")
        
        try:
            # 1. Load and Preprocess (Loading Time+arcsinh vs Batch-corrected)
            df_b, df_a, markers = load_and_preprocess_panel_batch(panel, dataset, ctx, verbose=verbose)
            if df_b is None or df_b.empty or df_a.empty:
                continue
            
            print(f"  [MEMORY] Sampling data for cross-panel comparison...")
            all_dfs[(dataset, panel)] = {
                "before": df_b.sample(n=min(len(df_b), 100000), random_state=42) if not df_b.empty else df_b, 
                "after": df_a.sample(n=min(len(df_a), 100000), random_state=42) if not df_a.empty else df_a
            }
            all_markers[(dataset, panel)] = markers
            
            # DIRECTORY STRUCTURE
            base_panel_dir = os.path.join(ctx.output_dir, "results", dataset, panel)
            
            path_Variance = os.path.join(base_panel_dir, "Batch_Variance")
            path_AnchorSignal = os.path.join(base_panel_dir, "Batch_AnchorSignal")
            path_AnchorCorrection = os.path.join(base_panel_dir, "Batch_AnchorCorrectionSignal")
            path_UMAP = os.path.join(base_panel_dir, "Batch_UMAP")
            path_EMD = os.path.join(base_panel_dir, "Batch_EMD")
            path_MarkerCorr = os.path.join(base_panel_dir, "Batch_MarkerCorrelation")
            
            for d in [path_Variance, path_AnchorSignal, path_AnchorCorrection, path_UMAP, path_EMD, path_MarkerCorr]:
                os.makedirs(d, exist_ok=True)
            
            # 2. PCA Variance Analysis
            print(f"[ANALYZE] Running variation_pca...")
            # Use columns that actually exist in the data to avoid KeyError
            factors = [c for c in ["plate", "sample_name", "celltypes", "Time"] if c in df_b.columns]
            
            # Use only true_markers for marker analysis if provided
            analysis_markers = [m for m in markers if m in ctx.true_markers] if ctx.true_markers else markers
            
            # Ensure group_col exists or fallback to None/plate
            group_col = "condition" if "condition" in df_b.columns else (None if "condition" not in df_b.columns else "condition")
            # If the user explicitly set some values to None or the column is missing:
            if group_col == "condition" and df_b["condition"].isna().all():
                group_col = None

            var_b = variation_pca(df_b, markers=analysis_markers, group_col=group_col, info_cols=factors, panel=panel)
            var_a = variation_pca(df_a, markers=analysis_markers, group_col=group_col, info_cols=factors, panel=panel)
            # Store in the correct cross-panel key (dataset, panel)
            all_variance_results[(dataset, panel)] = {"before": var_b, "after": var_a}
            
            var_b.to_csv(os.path.join(path_Variance, f"variance_before_{dataset}_{panel}_Batch.csv"))
            var_a.to_csv(os.path.join(path_Variance, f"variance_after_{dataset}_{panel}_Batch.csv"))

            # 2.2 Marker-level Variance Boxplots
            print(f"[ANALYZE] Generating Marker-level variance boxplots...")
            mvar_b = variation_marker(df_b, markers=analysis_markers, group_col=None, info_cols=factors, panel=panel)
            mvar_a = variation_marker(df_a, markers=analysis_markers, group_col=None, info_cols=factors, panel=panel)
            
            # Save the raw marker-level variance data to CSV
            mvar_b.to_csv(os.path.join(path_Variance, f"variance_marker_raw_before_{dataset}_{panel}_Batch.csv"), index=False)
            mvar_a.to_csv(os.path.join(path_Variance, f"variance_marker_raw_after_{dataset}_{panel}_Batch.csv"), index=False)
            
            plot_variance_marker_boxplot(mvar_b, mvar_a, save_path=os.path.join(path_Variance, f"variance_marker_boxplot_{dataset}_{panel}_Batch.png"), use_log=True)

            # 2.3 SCIB Metrics (ASW, ARI, NMI, Graph Connectivity, PCR)
            print(f"[ANALYZE] Computing Batch Correction Metrics")
            try:
                # Use analysis_markers for global metrics
                metrics_before, metrics_after = compute_metrics(
                    df_after=df_a, 
                    markers=analysis_markers, 
                    df_before=df_b, 
                    batch_key='plate', 
                    label_key='celltypes', 
                    subsample=10000, 
                    n_iterations=20
                )
                plot_metrics_comparison(metrics_before, metrics_after, panel=panel, save_dir=path_Variance)
            except Exception as e:
                traceback.print_exc()
                print(f"  [WARN] Failed to compute SCIB metrics for {dataset}|{panel}: {e}")

            # 3. Anchor Visualizations
            anchor_name = ctx.anchors.get(dataset)
            # # print(f"[ANALYZE] Visualizing Anchor signal...")
            # # visualize_anchor_across_plates(df=df_b, panel=panel, markers=analysis_markers, save_dir=path_AnchorSignal, anchor_name=anchor_name, true_markers=ctx.true_markers)
            
            # Anchor Correction Comparison
            df_anchor_b = df_b[df_b["sample_name"] == anchor_name].copy()
            df_anchor_a = df_a[df_a["sample_name"] == anchor_name].copy()
            if not df_anchor_b.empty and not df_anchor_a.empty:
                plot_signal_distributions(df_before=df_anchor_b, df_after=df_anchor_a, panel=panel, markers=analysis_markers, 
                                            save_dir=path_AnchorCorrection)

            # 4. UMAP Evaluatons
            print(f"[ANALYZE] Generating UMAPs...")
            # Resolve phenotypic markers: ctx > study config JSON > all analysis_markers
            umap_markers = None
            if ctx.pheno_markers:
                umap_markers = [m for m in ctx.pheno_markers if m in df_b.columns]
            if not umap_markers:
                umap_markers = _load_pheno_markers(project_root, dataset, panel, df_b.columns)
            if not umap_markers:
                umap_markers = analysis_markers
            all_umap_markers[(dataset, panel)] = umap_markers

            # Plate effect
            umap_plate_effect(df_b, df_a, markers=umap_markers, subsample=subsample_umap,
                                save_path=os.path.join(path_UMAP, f"umap_plate_effect_{dataset}_{panel}_Batch.png"))

            # Batch Evaluation
            umap_batch_evaluation(df_raw=df_b, df_corrected=df_a, markers=umap_markers,
                                    save_path=os.path.join(path_UMAP, f"umap_batch_evaluation_{dataset}_{panel}_Batch.png"),
                                    true_markers=ctx.true_markers)

            # Celltype by Plate
            umap_celltype_by_plate(df_raw=df_b, df_corrected=df_a, markers=umap_markers, subsample=subsample_umap,
                                    save_path=os.path.join(path_UMAP, f"umap_celltype_by_plate_{dataset}_{panel}_Batch.png"))

            # Marker Expression UMAP
            umap_marker_expression(df_b, df_a, umap_markers, panel,
                                    save_path=os.path.join(path_UMAP, f"umap_marker_expression_{dataset}_{panel}_Batch.png"))

            # 5. EMD and MAD
            print(f"[ANALYZE] Computing EMD/MAD...")
            # If 'celltypes' exists, pass it to use biological clusters instead of SOM
            cluster_param = 'celltypes' if 'celltypes' in df_b.columns else None
            emd_df, _, mad_score = compute_emd_mad(
                df_raw=df_b, 
                df_corr=df_a, 
                markers=analysis_markers, 
                cluster_col=cluster_param,
                anchor=anchor_name, 
                n_sample=50000, 
                save_dir=path_EMD
            )
            plot_emd(emd_df, mad_score, save_path=path_EMD)
            plot_emd_per_marker(emd_df, save_path=path_EMD)

            # 6. Marker Correlation
            print(f"[ANALYZE] Marker Correlation...")
            marker_corr(df_before=df_b, df_after=df_a, markers=analysis_markers, save_dir=path_MarkerCorr, group_col='condition',)

            # 7. Celltype Variance (Pre-storage)
            v_ct_b, v_ct_a = compute_celltype_variance(df_b, df_a, ctx=ctx)
            all_celltype_variance[(dataset, panel)] = {"before": v_ct_b, "after": v_ct_a}

            # CLEANUP: Final free of big dataframes
            del df_b, df_a
            gc.collect() 

            # 8. Time Drift Correction Visualization (Raw vs Corrected)
            print(f"Generating Time Drift Alignment for {panel}...")
            
            # Use metadata to find exactly which files belong to this dataset and panel
            panel_metadata = metadata_df[
                (metadata_df["dataset"] == dataset) & 
                (metadata_df["panel"] == panel)
            ]

            if not panel_metadata.empty:
                # To save memory, Subsample each file early
                subsample_per_file = 5000 
                dfs_raw = []
                dfs_corr_vis = []
                
                print(f"  [LOAD] Processing {len(panel_metadata)} files from metadata for visualization...")
                
                for _, row in panel_metadata.iterrows():
                    f_name = row["file_name"]
                    sample_name = str(row.get("sample_name", "Unknown"))
                    
                    # 1. Check RAW file first
                    rf = os.path.join(ctx.data_dir, f_name)
                    if not os.path.exists(rf):
                        base_name = os.path.splitext(f_name)[0]
                        potential_files = glob(os.path.join(ctx.data_dir, f"{base_name}.*"))
                        if potential_files: rf = potential_files[0]
                    
                    if os.path.exists(rf):
                        if rf.lower().endswith('.fcs'):
                            try:
                                # Use FlowCytometryTools to check columns without reading everything
                                f_check = FCMeasurement(ID='temp', datafile=rf)
                                channels = list(f_check.get_data().columns)
                                if 'Time' not in channels:
                                    print(f"  [SKIP-FILE] Raw file '{f_name}' missing 'Time'. Skipping visualization.")
                                    continue
                            except Exception as e:
                                print(f"  [WARN] Could not read FCS metadata for {f_name}: {e}")
                                continue
                        else:
                            try:
                                header_r = pd.read_csv(rf, nrows=0)
                            except UnicodeDecodeError:
                                header_r = pd.read_csv(rf, nrows=0, encoding='latin1')
                            
                            if 'Time' not in header_r.columns:
                                print(f"  [SKIP-FILE] Raw file '{f_name}' missing 'Time'. Skipping visualization.")
                                continue
                    else:
                        continue

                    # 2. Check CORRECTED file
                    cf_base = os.path.splitext(f_name)[0]
                    cf_path = os.path.join(ctx.output_dir, f"{cf_base}.csv")
                    is_uncorrected = False
                    if not os.path.exists(cf_path):
                        cf_path = os.path.join(ctx.output_dir, f"{cf_base}_UNCORRECTED.csv")
                        is_uncorrected = True
                    
                    if os.path.exists(cf_path):
                        if is_uncorrected: continue
                        header_c = pd.read_csv(cf_path, nrows=0)
                        if 'Time' not in header_c.columns:
                            print(f"  [SKIP-FILE] Corrected file '{os.path.basename(cf_path)}' missing 'Time'. Skipping visualization.")
                            continue
                    else:
                        continue
                    
                    # --- ALL CHECKS PASSED: START ACTUAL LOADING ---
                    
                    # 1. Load RAW data
                    rdf_full = read_cytof_file(rf)
                    rdf_full.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in rdf_full.columns]
                    needed_cols = ["Time"] + analysis_markers
                    actual_cols = [c for c in needed_cols if c in rdf_full.columns]
                    
                    rdf_sub = rdf_full.sample(n=min(len(rdf_full), subsample_per_file), random_state=42)[actual_cols].astype('float32').copy()
                    del rdf_full
                    
                    rdf_sub = arcsinh(rdf_sub, [c for c in rdf_sub.columns if c != "Time"])
                    rdf_sub['plate'] = str(row.get('plate', 'Unknown'))
                    rdf_sub['sample_name'] = sample_name
                    rdf_sub['condition'] = str(row.get('condition', 'Unknown'))
                    rdf_sub['panel'] = str(panel)
                    
                    for col in ['plate', 'sample_name', 'condition', 'panel']:
                        rdf_sub[col] = rdf_sub[col].astype('category')
                    dfs_raw.append(rdf_sub)

                    # 2. Handle CORRECTED data (Look in ctx.output_dir)
                    # cf_path was already determined in pre-check section
                    is_uncorrected = "_UNCORRECTED" in os.path.basename(cf_path)
                    
                    if os.path.exists(cf_path):
                        cdf_full = read_cytof_file(cf_path)
                        cdf_full.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in cdf_full.columns]
                        
                        # Use analysis_markers + Time for visualization since pre-check confirmed Time exists
                        needed_cols_a = analysis_markers + ['Time']
                        actual_cols_a = [c for c in needed_cols_a if c in cdf_full.columns]
                        
                        cdf_sub = cdf_full.sample(n=min(len(cdf_full), subsample_per_file), random_state=42)[actual_cols_a].astype('float32').copy()
                        del cdf_full
                        
                        cdf_sub['plate'] = str(row.get('plate', 'Unknown'))
                    is_uncorrected = False
                    
                    if not os.path.exists(cf_path):
                        cf_path = os.path.join(ctx.output_dir, f"{cf_base}_UNCORRECTED.csv")
                        is_uncorrected = True
                    
                    if os.path.exists(cf_path):
                        if is_uncorrected:
                            continue
                            
                        cdf_full = read_cytof_file(cf_path)
                        cdf_full.columns = [harmonize_marker(c) if not c.startswith("OmiqFilter") else c.replace("OmiqFilter","") for c in cdf_full.columns]
                        
                        # Use analysis_markers + Time for visualization if available
                        needed_cols_a = analysis_markers.copy()
                        if 'Time' in cdf_full.columns:
                            needed_cols_a.append('Time')
                        
                        actual_cols_a = [c for c in needed_cols_a if c in cdf_full.columns]
                        
                        cdf_sub = cdf_full.sample(n=min(len(cdf_full), subsample_per_file), random_state=42)[actual_cols_a].astype('float32').copy()
                        del cdf_full
                        
                        if 'Time' not in cdf_sub.columns and 'Time' in rdf_sub.columns:
                            # Direct index assignment only works if lengths match (which they don't due to random sampling)
                            rdf_sub_sync = rdf_sub.copy() # Already has Time
                            cdf_sub['Time'] = rdf_sub_sync.head(len(cdf_sub))['Time'].values # Approximation or re-read
                            pass

                        cdf_sub['plate'] = str(row.get('plate', 'Unknown'))
                        cdf_sub['sample_name'] = sample_name
                        cdf_sub['condition'] = str(row.get('condition', 'Unknown'))
                        cdf_sub['panel'] = str(panel)
                        
                        for col in ['plate', 'sample_name', 'condition', 'panel']:
                            cdf_sub[col] = cdf_sub[col].astype('category')
                        dfs_corr_vis.append(cdf_sub)
                
                if dfs_raw and dfs_corr_vis:
                    df_raw_vis = pd.concat(dfs_raw, ignore_index=True)
                    df_corr_vis = pd.concat(dfs_corr_vis, ignore_index=True)
                    
                    vis_save_dir = os.path.join(base_panel_dir, "TimeCorrection_Alignment")
                    print(f"  [PLOT] Saving alignment plots to {vis_save_dir}...")
                    plot_time_effect_density_merged(
                        before_df=df_raw_vis,
                        after_df=df_corr_vis, 
                        panel=panel, 
                        markers=analysis_markers, 
                        save_dir=vis_save_dir,
                        subsample=None
                    )
                    del df_raw_vis, df_corr_vis
                    gc.collect()
                else:
                    print(f"  [WARN] Missing files for visualization: raw={len(dfs_raw)}, corrected={len(dfs_corr_vis)}")
            else:
                print(f"  [WARN] No metadata entries found for {dataset} | {panel}")

            results_summary.append({"dataset": dataset, "panel": panel, "status": "Success"})
            
        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            traceback.print_exc()
            results_summary.append({"dataset": dataset, "panel": panel, "status": f"Error: {str(e)}"})

    summary_df = pd.DataFrame(results_summary)

    # CROSS-PANEL ANALYSES (Global Variance and CellType Variance)
    if not summary_df.empty:
        # Group by dataset to find P1/P2 pairs
        for dataset in summary_df[summary_df['status'] == 'Success']['dataset'].unique():
            success_keys = [k for k in all_variance_results.keys() if k[0] == dataset]
            
            if len(success_keys) >= 1:
                # Ensure we have Panel1 and Panel2 for this dataset
                p1_key = next((k for k in success_keys if k[1] == 'Panel1'), None)
                p2_key = next((k for k in success_keys if k[1] == 'Panel2'), None)
                
                # If Panel1 doesn't exist but others do, take the first one
                if not p1_key: p1_key = success_keys[0]

                res_root = os.path.join(ctx.output_dir, "results", dataset)
                path_shared_comp = os.path.join(res_root, "SharedMarker_Comparison")
                path_shared_corr = os.path.join(res_root, "SharedMarker_Correlation")
                path_variance_shared = os.path.join(res_root, "Variance")
                
                for d in [path_shared_comp, path_shared_corr, path_variance_shared]:
                    os.makedirs(d, exist_ok=True)

                print(f"\n[POST-PROCESS] Post-correction analyses for {dataset}...")
                
                # These functions now internally handle cases where p2 data is None
                p2_args = {
                    "p2_before": all_variance_results[p2_key]["before"] if p2_key else None,
                    "p2_after": all_variance_results[p2_key]["after"] if p2_key else None
                }
                
                # Global Factor Variance Comparison
                plot_variance_bar(
                    all_variance_results[p1_key]["before"], all_variance_results[p1_key]["after"],
                    **p2_args,
                    save_path=os.path.join(path_variance_shared, f"{dataset}_global_variance.png")
                )

                # Celltype Variance Mean Comparison
                p2_celltype_before = all_celltype_variance[p2_key]["before"] if p2_key else None
                p2_celltype_after = all_celltype_variance[p2_key]["after"] if p2_key else None
                plot_variance_bar_celltype(
                    all_celltype_variance[p1_key]["before"], all_celltype_variance[p1_key]["after"],
                    var_P2_before=p2_celltype_before, var_P2_after=p2_celltype_after,
                    save_path=os.path.join(path_variance_shared, f"{dataset}_celltype_variance.png")
                )

                # Shared Marker Analysis
                if p1_key and p2_key:
                    print(f"[POST-PROCESS] Analyzing Markers (P1 vs P2)...")
                    visualize_shared_marker_correction(
                        all_dfs[p1_key]["before"], all_dfs[p1_key]["after"],
                        df_raw_P2=all_dfs[p2_key]["before"] if p2_key else None,
                        df_corrected_P2=all_dfs[p2_key]["after"] if p2_key else None,
                        markers_P1=all_umap_markers.get(p1_key, all_markers[p1_key]),
                        markers_P2=all_umap_markers.get(p2_key, all_markers[p2_key]) if p2_key else None,
                        save_dir=path_shared_comp
                    )
                else:
                    print(f"[POST-PROCESS] Skipping Shared Marker Analysis (Only one panel found).")

                marker_expression_correlation_plot(
                    all_dfs[p1_key]["before"], all_dfs[p1_key]["after"],
                    df_P2_before=all_dfs[p2_key]["before"] if p2_key else None,
                    df_P2_after=all_dfs[p2_key]["after"] if p2_key else None,
                    save_dir=path_shared_corr,
                    ctx=ctx
                )

    print(f"\n{'='*60}\nBATCH CORRECTION ANALYSIS COMPLETE\n{'='*60}\n")
    return summary_df


if __name__ == "__main__":

    # Command-line interface for running the pipeline with specific steps and datasets
    parser = argparse.ArgumentParser(description="CyTOF Batch Correction Pipeline CLI")
    parser.add_argument("step", choices=["correction", "time-analysis", "batch-analysis", "all"], 
                        help="Choose which step to run")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to process (e.g. D0)")
    
    # Set default config path to metadata/meta.json relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(project_root, "metadata", "meta.json")
    parser.add_argument("--config", type=str, default=default_config, help="Path to meta.json config file")

    args = parser.parse_args()

    if args.step == "correction" or args.step == "all":
        print(f"\n>>> Running [Correction] for {args.dataset}...")
        run_cytof_correction(dataset_to_process=args.dataset, config_path=args.config)

    if args.step == "time-analysis" or args.step == "all":
        print(f"\n>>> Running [Time Analysis] for {args.dataset}...")
        run_time_analysis(datasets=[args.dataset], config_path=args.config)

    if args.step == "batch-analysis" or args.step == "all":
        print(f"\n>>> Running [Batch Analysis] for {args.dataset}...")
        run_batch_analysis(datasets=[args.dataset], config_path=args.config)

    print("\n[FINISH] Operation completed successfully.")