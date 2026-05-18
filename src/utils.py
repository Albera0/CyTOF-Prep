import os
import re
import json

from sklearn.decomposition import PCA
import numpy as np
import pickle
import colorsys
import pandas as pd
import pickle
import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
from statsmodels.formula.api import ols
from FlowCytometryTools import FCMeasurement
from glob import glob
from collections import defaultdict


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


def get_indexed_files(directory, dataset_filter, panel_filter, metadata_df, exclude_uncorrected=False):
    """
    Scans a directory for CyTOF files and indexes them by plate based on metadata.
    """
    if metadata_df is None:
        return {}

    file_index = defaultdict(list)
    
    def clean(name):
        return str(name).lower().replace(".csv", "").replace(".fcs", "").replace("_time", "").replace("_anchor", "").replace("_batch", "").replace("_uncorrected", "").replace("_", "")
    
    metadata_df_copy = metadata_df.copy()
    metadata_df_copy["_clean_name"] = metadata_df_copy["file_name"].apply(clean)

    for f in glob(os.path.join(directory, "*")):
        if not f.lower().endswith(('.csv', '.fcs')):
            continue
        
        file_name = os.path.basename(f)

        # Skip uncorrected files if requested
        if exclude_uncorrected and "_uncorrected" in file_name.lower():
            continue

        f_clean = clean(file_name)
        
        match = metadata_df_copy[metadata_df_copy["_clean_name"] == f_clean]
        
        if match.empty:
            # Fallback to substring matching if exact clean match fails
            match = metadata_df_copy[metadata_df_copy["_clean_name"].apply(lambda x: x in f_clean or f_clean in x)]

        if not match.empty:
            row = match.iloc[0]
            if str(row.get("dataset", "")) == str(dataset_filter) and str(row.get("panel", "")) == str(panel_filter):
                plate = str(row.get("plate", "Unknown"))
                file_index[plate].append(f)
    
    return file_index


def read_cytof_file(file_path, **kwargs):
    """
    Unified reader for CyTOF data (CSV or FCS).
    
    Parameters
    ----------
    file_path : str
        Path to the file (.csv or .fcs).
    **kwargs : dict
        Additional arguments passed to pd.read_csv (if CSV).
    
    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path, **kwargs)
        # Drop 'Unnamed' columns that often appear due to trailing commas in CSVs
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        return df
    elif ext == ".fcs":
        if FCMeasurement is None:
            raise ImportError("FlowCytometryTools is required to read FCS files. Install it via 'pip install FlowCytometryTools'.")
        
        # Using FlowCytometryTools to read FCS
        sample = FCMeasurement(ID='sample', datafile=file_path)
        df = sample.get_data()
        
        # Handle usecols if provided
        if "usecols" in kwargs:
            cols = kwargs["usecols"]
            df = df[[c for c in cols if c in df.columns]]
        return df
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_config(path):
    """
    Load a JSON configuration file.

    Parameters
    ----------
    path : str
        Path to config.json

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    with open(path, "r") as f:
        return json.load(f)



def get_config(config_path=None, config_dict=None):
    """
    Load and validate config for CyTOF pipeline.

    This is the main entry function you should use in:
    - notebook
    - scripts
    - python package

    Parameters
    ----------
    config_path : str, optional
        Path to config.json
    config_dict : dict, optional
        Dictionary containing configuration

    Returns
    -------
    dict
        Cleaned and validated config
    """

    if config_dict is not None:
        config = config_dict.copy()
    elif config_path is not None:
        config = load_config(config_path)
    else:
        raise ValueError("Either config_path or config_dict must be provided.")

    # required fields check
    required_keys = [
        "data_dir",
        "output_dir",
        "ref_plate",
        "anchors",
        "meta_cols",
        "celltypes",
        "celltype2correct",
        "stable_marker",
        "true_markers"
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # optional fields
    config["celltype_gating"] = config.get("celltype_gating", [])
    config["metal_to_marker"] = config.get("metal_to_marker", {})
    config["pheno_markers"]   = config.get("pheno_markers", None)

    # ensure output dir exists
    os.makedirs(config["output_dir"], exist_ok=True)

    return config


class CyTOFContext:
    """
    Data class used to centralize pipeline settings and metadata.
    
    Acts as a single source of truth for directory paths, technical column names,
    anchors sample mapping, and marker definitions used across all correction
    and visualization modules.
    """
    def __init__(self, config):
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.ref_plate = config["ref_plate"]
        self.anchors = config["anchors"]
        self.meta_cols = config["meta_cols"]
        self.celltypes = config["celltypes"]
        self.celltype_gating = config["celltype_gating"]
        self.celltype2correct = config["celltype2correct"]
        self.stable_marker = config["stable_marker"]
        self.true_markers = config["true_markers"]
        self.metal_to_marker = config.get("metal_to_marker", {})
        self.pheno_markers = config.get("pheno_markers", None)


# Utils

def apply_marker_schema(df, ctx):
    """
    Standardizes marker naming by mapping detector channels (metals) to biological targets.
    
    Uses 'metal_to_marker' mapping from the configuration to rename columns like 
    '175Lu' to 'PARPg', ensuring consistent analysis regardless of panel design.

    Args:
        df (pd.DataFrame): Dataframe with raw channel names.
        ctx (CyTOFContext): Context containing the metal-to-marker mapping.

    Returns:
        pd.DataFrame: Dataframe with target marker names.
    """
    for metal, marker in ctx.metal_to_marker.items():
        # If the metal channel exists in the dataframe, copy it to the marker name
        if metal in df.columns:
            df[marker] = df[metal]
            print(f"[SCHEMA] {metal} → {marker}")
    # Drop the original metal channels
    df = df.drop(columns=list(ctx.metal_to_marker.keys()), errors="ignore")
    return df

def harmonize_marker(name):
    """
    Standardize marker names by removing prefixes and special delimiters.
    """
    if "___" in name:
        name = name.split("___")[-1]  # Remove extra prefix
    name = re.sub(r'^\d+[A-Za-z]+_', '', name)  # Remove leading digits+letters prefix
    return name.strip()

def add_metadata(df, file_path, metadata_df=None, ctx=None):
    """
    Extract metadata using metadata_df. Supports dynamic metadata columns from ctx.meta_cols.
    """
    if metadata_df is None:
        raise ValueError("Metadata DataFrame must be provided for file identification.")

    def clean(name):
        return str(name).lower().replace(".csv", "").replace(".fcs", "").replace("_time", "").replace("_anchor", "").replace("_batch", "").replace("_uncorrected", "").replace("_", "")

    file_name = os.path.basename(file_path)
    f_clean = clean(file_name)
    
    if not hasattr(metadata_df, "_clean_names"):
        metadata_df["_clean_names"] = metadata_df["file_name"].apply(clean)

    # 1. Exact match
    match = metadata_df[metadata_df["file_name"] == file_name]
    
    # 2. Match without extension
    if match.empty:
        name_no_ext = os.path.splitext(file_name)[0]
        match = metadata_df[metadata_df["file_name"] == name_no_ext]

    # 3. Fuzzy match
    if match.empty:
        match = metadata_df[metadata_df["_clean_names"] == f_clean]
        
    if match.empty:
        match = metadata_df[metadata_df["_clean_names"].apply(lambda x: x in f_clean or f_clean in x)]
        
    if match.empty:
        raise FileNotFoundError(f"File {file_name} not found in metadata.xlsx.")

    row = match.iloc[0]
    
    # Use dynamic metadata columns from ctx if available
    if ctx is not None:
        target_meta_cols = ctx.meta_cols
    else:
        target_meta_cols = ["Time", "Event_length", "dataset", "panel", "plate", "sample_name", "condition"]

    # Inject metadata columns
    for col in target_meta_cols:
        if col in row.index:
            df[col] = row.get(col, "")
        elif col == "original_filename":
            df["original_filename"] = file_name
    
    # Special case for columns already in df (like Time/Event_length)
    if "original_filename" not in df.columns:
        df["original_filename"] = file_name

    # Define logical reordering — deduplicate while preserving order
    _seen: set = set()
    all_meta_order = [
        c for c in ["original_filename"] + target_meta_cols
        if not (c in _seen or _seen.add(c))  # type: ignore[func-returns-value]
    ]
    actual_meta    = [c for c in all_meta_order if c in df.columns]
    actual_markers = [c for c in df.columns if c not in set(actual_meta)]

    return df[actual_meta + actual_markers]

def assign_cell_type(df, ctx, gating_cols=None):
    """
    Assign cell type labels from binary flag columns; only keep CD45 positive cells.

    Args:
        df: cell DataFrame (post-harmonization).
        ctx: CyTOFContext.
        gating_cols: explicit list of column names (already in df) to use as cell type
            flags. When provided, ctx.celltype_gating is ignored. When None, falls back
            to ctx.celltype_gating (checking both the clean name and OmiqFilter variant).
    """
    if "CD45pos" in df.columns:
        cd45 = pd.to_numeric(df["CD45pos"], errors="coerce").fillna(0)
        df = df[cd45 > 0].copy()

    if gating_cols is not None:
        # Use the explicitly provided column names directly
        present = [c for c in gating_cols if c in df.columns]
    else:
        if not ctx.celltype_gating:
            print("[INFO] Skip cell type assignment (no gating provided)")
            return df
        # Match ctx.celltype_gating names against df, also trying the OmiqFilter variant
        present = []
        for ct in ctx.celltype_gating:
            if ct in df.columns:
                present.append(ct)
            elif f"OmiqFilter{ct}" in df.columns:
                present.append(f"OmiqFilter{ct}")

    if not present:
        print("[INFO] Skip cell type assignment (no gating columns found in data)")
        return df

    # Show a sample value per gating column so the user can verify format
    for col in present[:3]:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        print(f"[GATE] '{col}' sample value: {repr(sample)} (type={type(sample).__name__})")

    def _is_positive(val):
        """Return True for any truthy gate membership value (1, 1.0, True, 'True', 'true', etc.)."""
        if val is None:
            return False
        if isinstance(val, bool):
            return val
        try:
            return float(val) > 0
        except (ValueError, TypeError):
            return str(val).strip().lower() == "true"

    def get_type(row):
        for col in present:
            if _is_positive(row.get(col)):
                return col
        return None

    df["celltypes"] = df.apply(get_type, axis=1)
    n_assigned = df["celltypes"].notna().sum()
    if n_assigned == 0:
        print(f"[WARN] 0 cells assigned a cell type from columns: {present}")
        print(f"       Check that column values represent gate membership (1, True, etc.)")
        return df
    return df[df["celltypes"].notna()]

def enforce_schema(df, ctx):
    """
    Clean columns and identify targets for correction.
    - Identifies 'markers' (intersection of file columns and true_markers) for Batch Correction.
    - Identifies 'extra_markers' (present in file but not in true_markers) to be preserved and Arcsinh transformed.
    """
    # 1. Identify existing meta columns
    meta = [c for c in ctx.meta_cols if c in df.columns]
    
    # 2. Remove helper gating columns after cell_type assignment
    gating_cols = set(getattr(ctx, "celltypes", [])) | set(getattr(ctx, "celltype_gating", []))
    gating_cols.add("CD45pos")
    drop_gating = [c for c in gating_cols if c in df.columns]
    if drop_gating:
        df = df.drop(columns=drop_gating, errors="ignore")

    # 3. Identify all potential data columns (not meta)
    potential_markers = [c for c in df.columns if c not in meta]
    
    # 4. Filter out 'ghost' columns: All zeros, all NaNs, or constant zero
    valid_data_cols = []
    for c in potential_markers:
        series = pd.to_numeric(df[c], errors='coerce')
        if not (series.isna().all() or (series == 0).all()):
            valid_data_cols.append(c)
    
    # 5. Separate into target markers (for correction) and extra markers (just arcsinh)
    markers = [m for m in ctx.true_markers if m in valid_data_cols]
    extra_markers = [c for c in valid_data_cols if c not in ctx.true_markers]
    
    # Reorder: Meta + Correction Markers + Extra Markers
    full_df = df[meta + ["celltypes"] + markers + extra_markers].copy()
    
    # Return ONLY 2 values as expected by the rest of the pipeline
    return full_df, markers



def generate_and_save_colors(
    save_path,
    metadata_path=None,
    df=None,
    celltype_col="celltypes",
    panel_col="panel",
    plate_col="plate",
    sample_col="sample_name",
    seed=40
):
    """
    Generate color dictionary from either a dataframe or an Excel metadata file and save to file.

    Parameters
    ----------
    save_path : str
        Path to save color dictionary (e.g., "colors_dict.pkl")
    metadata_path : str, optional
        Path to metadata.xlsx. If provided, df will be ignored.
    df : pd.DataFrame, optional
        Input dataframe if metadata_path is not provided.
    """

    if metadata_path is not None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        df = pd.read_excel(metadata_path)
    elif df is None:
        raise ValueError("Either metadata_path or df must be provided.")

    np.random.seed(seed)
    colors_dict = {}

    def hsv_colors(n, h_start=0, h_end=1, s=0.8, l=0.7, shuffle=True):
        hs = np.linspace(h_start, h_end, n, endpoint=False)
        colors = [colorsys.hls_to_rgb(h, l, s) for h in hs]
        if shuffle:
            np.random.shuffle(colors)
        return colors

    # Celltype use a shared palette with panel
    cell_types = list(df[celltype_col].unique()) if celltype_col in df.columns else []
    panels = list(df[panel_col].unique()) if panel_col in df.columns else []

    combined_cats = list(dict.fromkeys(cell_types + panels))
    combined_palette = hsv_colors(len(combined_cats), s=0.6, l=0.6, shuffle=True)
    cat_to_color = dict(zip(combined_cats, combined_palette))

    colors_dict[celltype_col] = {c: cat_to_color[c] for c in cell_types}
    colors_dict[panel_col] = {c: cat_to_color[c] for c in panels}

    # Plate generate its own palette (bright and distinct)
    if plate_col in df.columns:
        plates = df[plate_col].unique()
        plate_palette = hsv_colors(len(plates), s=0.8, l=0.7, shuffle=True)
        colors_dict[plate_col] = dict(zip(plates, plate_palette))

    # Sample generate its own palette (bright and distinct)
    if sample_col in df.columns:
        samples = df[sample_col].unique()
        sample_palette = hsv_colors(len(samples), s=0.7, l=0.8, shuffle=True)
        colors_dict[sample_col] = dict(zip(samples, sample_palette))

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Save
    with open(save_path, "wb") as f:
        pickle.dump(colors_dict, f)

    print(f"[INFO] Colors generated from {metadata_path if metadata_path else 'DataFrame'} and saved to {save_path}")
    return colors_dict



def load_and_subset_colors(df, load_path):
    """
    Load color dictionary and subset to match dataframe categories.

    Parameters
    ----------
    df : pd.DataFrame
        Target dataframe
    load_path : str
        Path to saved color dictionary

    Returns
    -------
    dict
        Subset color dictionary
    """

    with open(load_path, "rb") as f:
        colors_dict = pickle.load(f)

    new_colors_dict = {}

    for key in colors_dict.keys():
        if key in df.columns:
            categories = df[key].unique()
            new_colors_dict[key] = {
                k: colors_dict[key][k]
                for k in categories
                if k in colors_dict[key]
            }
        else:
            new_colors_dict[key] = colors_dict[key]

    return new_colors_dict


# Additional utility functions for time correction and batch correction

# Compute variance explained by metadata factors for each group based on PCA
def variation_pca(df, markers, group_col, info_cols, panel=None, n_pcs=5, subsample=100000):
    # Subset to selected panel
    df_panel = df[df['panel'] == panel].dropna(axis=1, how="all").copy()

    # Keep only markers that exist in dataframe
    markers = [m for m in markers if m in df_panel.columns]

    results = {}

    # Loop over each condition separately
    if group_col is not None and group_col in df_panel.columns:
        grouped = df_panel.groupby(group_col)
    else:
        # If no grouping column, treat the whole panel as one group
        grouped = [("all", df_panel)]

    for group, df_group in grouped:
        
        # ADDED SUBSAMPLE
        if len(df_group) > subsample:
            df_group = df_group.sample(n=subsample, random_state=42)
            print(f"  [INFO] Subsampling {group} to {subsample} cells for PCA memory safety.")

        # Extract marker matrix and replace NaNs with 0
        df_markers = df_group[markers].fillna(0)
        
        # Ensure only numeric markers are passed to PCA (exclude strings like original_filename)
        numeric_markers = [m for m in df_markers.columns if pd.api.types.is_numeric_dtype(df_markers[m])]
        df_markers = df_markers[numeric_markers].astype(float)

        # Skip groups with too few cells
        if df_markers.shape[0] < 5:
            continue

        # Perform PCA
        pca = PCA(n_components=min(n_pcs, df_markers.shape[1], df_markers.shape[0]))
        X_pca = pca.fit_transform(df_markers)

        # Variance ratio of each PC
        var_ratio = pca.explained_variance_ratio_

        # Build DataFrame of PCs
        df_pcs = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
        )

        # Append metadata columns for ANOVA - Only include columns that actually exist in df_group
        present_info_cols = [col for col in info_cols if col in df_group.columns]
        df_pcs = pd.concat(
            [df_pcs, df_group[present_info_cols].reset_index(drop=True)],
            axis=1
        )

        # Ensure Time is numeric if present
        if 'Time' in df_pcs.columns:
            df_pcs['Time'] = df_pcs['Time'].astype(float)

        # DataFrame to store variance explained per PC - use present_info_cols as index
        var_exp = pd.DataFrame(index=present_info_cols)

        # Select valid info cols (factors) that are present in the current dataframe
        valid_info_cols = [col for col in present_info_cols if df_pcs[col].nunique() > 1]

        if not valid_info_cols:
            print(f"  [WARN] No valid metadata factors found for group {group}. Skipping ANOVA.")
            var_exp = pd.DataFrame(index=info_cols)
            results[group] = var_exp
            continue

        # Run ANOVA for each principal component
        for i, pc in enumerate(df_pcs.columns[:X_pca.shape[1]]):
            # Identification of columns with variation to avoid ANOVA failure
            active_cols = [col for col in present_info_cols if df_pcs[col].nunique() > 1]
            
            if not active_cols:
                # No predictors found, assume all variance is residual for this PC
                var_exp[f"PC{i+1}"] = 0
                continue

            # Build regression formula using only active predictors
            formula = f'Q("{pc}") ~ ' + " + ".join([f'Q("{col}")' for col in active_cols])

            try:
                # Fit linear model
                model = ols(formula, data=df_pcs).fit()
                # Type II ANOVA (more robust for unbalanced designs)
                anova_res = sm.stats.anova_lm(model, typ=2)
                
                # Compute proportion of variance explained
                # sum_sq for categorical factors includes the entire factor
                var_exp_per_factor = anova_res["sum_sq"] / anova_res["sum_sq"].sum()
                
                # MAP Q("factor") BACK TO factor IN THE SERIES INDEX
                clean_index = []
                for idx in var_exp_per_factor.index:
                    if idx.startswith('Q("') and idx.endswith('")'):
                        clean_index.append(idx[3:-2])
                    else:
                        clean_index.append(idx)
                var_exp_per_factor.index = clean_index
                
                var_exp[f"PC{i+1}"] = var_exp_per_factor
            except Exception as e:
                print(f"  [WARN] ANOVA failed for {group} {pc}: {e}")
                var_exp[f"PC{i+1}"] = 0

        # Weight each PC by its explained variance ratio
        var_exp = var_exp.fillna(0)
        weighted_var_explained = (var_exp * var_ratio).sum(axis=1)

        # Calculate Percentage relative to TOTAL variance (including residual from ignored PCs)
        res_dict = {}
        for factor in info_cols:
            if factor in weighted_var_explained.index:
                    # The sum_sq / sum_sq.sum() gives the fraction of that PC's variance.
                    # Multiplying by var_ratio gives fraction of TOTAL variance.
                res_dict[factor] = weighted_var_explained[factor]
            else:
                    res_dict[factor] = 0.0
                    
        # Calculate Residual
        if 'Residual' in weighted_var_explained.index:
            res_dict['Residual'] = weighted_var_explained['Residual']
        else:
            # If no explicit residual, it's the remaining part of the total variance scale
            total_explained = sum(res_dict.values())
            res_dict['Residual'] = (var_ratio.sum()) - total_explained

        results[group] = pd.Series(res_dict)

    # Convert dictionary to DataFrame
    results = pd.DataFrame(results).T
    
    return results


def variation_marker(df, markers, group_col, info_cols, panel=None, subsample=100000):
    """
    Calculates variance explained by metadata factors for EACH marker separately.
    Returns a long-form DataFrame with columns: [group, marker, factor, variance_explained].
    """
    # Subset to selected panel
    df_panel = df[df['panel'] == panel].dropna(axis=1, how="all").copy()

    # Keep only markers that exist in dataframe
    markers = [m for m in markers if m in df_panel.columns]

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
        
        # Identify columns with variation to avoid ANOVA failure
        active_cols = [col for col in info_cols if col in df_group.columns and df_group[col].nunique() > 1]
        
        if not active_cols:
            print(f"  [WARN] No valid metadata factors found for group {group}. Skipping marker-level variance.")
            continue

        for marker in markers:
            # Build regression formula: Marker ~ factors
            formula = f'Q("{marker}") ~ ' + " + ".join([f'Q("{col}")' for col in active_cols])

            try:
                # Fit linear model
                model = ols(formula, data=df_group).fit()
                # Type II ANOVA (more robust for unbalanced designs)
                anova_res = sm.stats.anova_lm(model, typ=2)
                
                # Compute proportion of variance explained
                var_exp_per_factor = anova_res["sum_sq"] / anova_res["sum_sq"].sum()
                
                for factor_raw, val in var_exp_per_factor.items():
                    # Clean the factor name from Q("...")
                    factor = factor_raw
                    if factor_raw.startswith('Q("') and factor_raw.endswith('")'):
                        factor = factor_raw[3:-2]
                    
                    all_res.append({
                        'group': group,
                        'marker': marker,
                        'factor': factor,
                        'variance_explained': val
                    })
            except Exception as e:
                print(f"  [WARN] ANOVA failed for {group} marker {marker}: {e}")

    return pd.DataFrame(all_res)


# Bar plot comparing variance explained by each factor before and after correction, for both panels side by side
def plot_variance_bar(p1_before, p1_after, p2_before=None, p2_after=None, save_path=None):
    # Determine which panels are present
    has_p2 = (p2_before is not None and p2_after is not None)
    
    # Panel1
    df1_before = p1_before.copy().reset_index()
    df1_before = df1_before.melt(id_vars='index', var_name='factor', value_name='before')
    df1_before = df1_before.rename(columns={'index':'condition'})

    df1_after = p1_after.copy().reset_index()
    df1_after = df1_after.melt(id_vars='index', var_name='factor', value_name='after')
    df1_after = df1_after.rename(columns={'index':'condition'})

    df1 = df1_before.merge(df1_after, on=['condition','factor'])
    df1['panel'] = 'Panel1'

    dfs_to_concat = [df1]

    # Panel2
    if has_p2:
        df2_before = p2_before.copy().reset_index()
        df2_before = df2_before.melt(id_vars='index', var_name='factor', value_name='before')
        df2_before = df2_before.rename(columns={'index':'condition'})

        df2_after = p2_after.copy().reset_index()
        df2_after = df2_after.melt(id_vars='index', var_name='factor', value_name='after')
        df2_after = df2_after.rename(columns={'index':'condition'})

        df2 = df2_before.merge(df2_after, on=['condition','factor'])
        df2['panel'] = 'Panel2'
        dfs_to_concat.append(df2)

    df = pd.concat(dfs_to_concat, ignore_index=True)
    df = df[df['factor'] != 'Residual']

    panels = ['Panel1']
    if has_p2:
        panels.append('Panel2')
        
    age_groups = df['condition'].unique()
    factors = [f for f in df['factor'].unique() if f != 'Residual']
    n_factors = len(factors)
    colors = [_PLATE_COLORS[i % len(_PLATE_COLORS)] for i in range(len(age_groups))]

    # Dynamic figure width based on number of factors and panels
    fig_width = max(8 if not has_p2 else 12, len(factors) * 0.8)
    n_axes = len(panels)
    fig, axes = plt.subplots(1, n_axes, figsize=(fig_width, 6), sharey=True, squeeze=False)
    # axes is always 2D due to squeeze=False

    for i, panel in enumerate(panels):
        ax = axes[0, i]
        df_panel = df[df['panel'] == panel]
        
        x = np.arange(len(factors))
        width = 0.8 / len(age_groups)
        
        for k, ag in enumerate(age_groups):
            df_ag = df_panel[df_panel['condition'] == ag].set_index('factor').reindex(factors).fillna(0)
            # Before
            ax.bar(x + k*width - width/4, df_ag['before'], width=width/2, color=colors[k % len(colors)], label=f"{ag} - Before", edgecolor='black')
            # After
            ax.bar(x + k*width + width/4, df_ag['after'], width=width/2, color=colors[k % len(colors)], alpha=0.5, label=f"{ag} - After", edgecolor='black')

        ax.set_xticks(x + (len(age_groups)-1)*width / 2)
        ax.set_xticklabels(factors, ha='right', rotation=45, fontsize=13, fontweight='bold')
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_title(panel, fontsize=16, fontweight='bold')
        ax.set_ylabel('Variance Explained', fontsize=15, fontweight='bold')
        ax.tick_params(axis='y', labelsize=13)

    # Create a single legend for both panels (or one panel)
    handles, labels = axes[0, n_axes-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, n_axes-1].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', title='Condition - Stage', fontsize=13, title_fontsize=14, prop={'weight':'bold'})
    plt.suptitle("Variance Explained by Factor - Correction Comparison", fontsize=20, fontweight='bold', y=1.03)

    plt.tight_layout()
    if save_path:
        _savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [SAVE] Variance bar plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return df


# Identify markers shared across all panels
def identify_shared_markers(df, info_col):

    # Define metadata columns to exclude
    meta = ['panel','plate','sample_name','celltypes',
                    'dataset','OmiqFilter','Event_length','Time','Anchor','condition', 'original_filename']

    panels = df[info_col].unique()
    marker_sets = []
    
    for panel in panels:
        df_panel = df[df[info_col] == panel]
        markers = set(
            col for col in df_panel.columns
            if col not in meta and df_panel[col].notna().any() and pd.api.types.is_numeric_dtype(df_panel[col])
        )
        marker_sets.append(markers)
    
    shared_markers = set.intersection(*marker_sets)
    shared_markers = list(shared_markers)
    return shared_markers


# Plot signal distributions for each marker across plates before and after correction
def plot_signal_distributions(df_before, df_after, panel, markers, save_dir, lower_pct=1, upper_pct=99, subsample=50000):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    df_before = df_before[df_before['panel'] == panel].dropna(axis=1, how="all")
    df_after = df_after[df_after['panel'] == panel].dropna(axis=1, how="all")

    # Filter numeric markers
    markers = [m for m in markers if m in df_before.columns and pd.api.types.is_numeric_dtype(df_before[m])]

    # Sort plates numerically/alphabetically (Plate1, Plate2...)
    plates = sorted([p for p in df_before['plate'].unique() if p in df_after['plate'].unique()], 
                    key=lambda x: str(x))
    if len(plates) == 0:
        print(f"No plates found for panel={panel}.")
        return

    # Color palette for all plates
    palette = [_PLATE_COLORS[i % len(_PLATE_COLORS)] for i in range(len(plates))]
    plate_colors = dict(zip(plates, palette))

    # Calculate global marker ranges using a sample for speed
    marker_ranges = {}
    df_sample_range = df_before.sample(n=min(len(df_before), subsample), random_state=42)
    for marker in markers:
        if marker not in df_sample_range.columns:
            continue
        all_vals = df_sample_range[marker].dropna()
        if len(all_vals) == 0:
            continue
        lower = np.percentile(all_vals, lower_pct)
        upper = np.percentile(all_vals, upper_pct)
        buffer = 0.2 * (upper - lower) if upper > lower else 0.5
        marker_ranges[marker] = (lower - buffer, upper + buffer)

    n_cols = 5
    n_rows = int(np.ceil(len(markers) / n_cols))

    # Function to draw plot for a given dataframe
    def draw_plots(df_source, title_prefix, save_name):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)
        
        for m_idx, marker in enumerate(markers):
            row, col = m_idx // n_cols, m_idx % n_cols
            ax = axes[row, col]

            if marker not in marker_ranges:
                ax.axis("off")
                continue

            # Pre-filter dataframe for efficiency
            df_marker = df_source[['plate', marker]].dropna()

            for plate in plates:
                data_plate = df_marker[df_marker["plate"] == plate][marker]
                
                # Speed up: Subsample for KDE
                if len(data_plate) > subsample:
                    data_plate = data_plate.sample(n=subsample, random_state=42)
                
                if len(data_plate) > 0:
                    sns.kdeplot(data_plate, ax=ax,
                                color=plate_colors[plate], clip=marker_ranges[marker],
                                linewidth=2)

            ax.set_xlim(marker_ranges[marker])
            ax.set_title(marker, fontsize=15, fontweight='bold')
            ax.set_xlabel("Signal", fontsize=13, fontweight='bold')
            ax.set_ylabel("Density", fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)

        # Turn off unused axes
        for empty_idx in range(len(markers), n_rows * n_cols):
            row, col = empty_idx // n_cols, empty_idx % n_cols
            axes[row, col].axis("off")

        legend_elements = [Line2D([0], [0], color=plate_colors[p], lw=3, label=str(p)) for p in plates]
        plt.tight_layout(rect=[0,0,0.85,0.95])
        fig.subplots_adjust(right=0.9)
        fig.legend(handles=legend_elements, title="Plate", loc='upper right', bbox_to_anchor=(1, 0.95), 
                    fontsize=12, title_fontsize=13, prop={'weight':'bold'})
        plt.suptitle(f"{title_prefix} - {panel}", fontsize=20, fontweight='bold')

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{panel}_{save_name}_distribution.png")
            _savefig(save_path, fig=fig, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        else:
            plt.show()
            plt.close(fig)

    draw_plots(df_before, "Before", "Before")
    draw_plots(df_after, "After", "After")







