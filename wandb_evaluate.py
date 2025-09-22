import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PROJECT_PATH = "lukas_heinrich-university-of-otago/eeg-object-eval-EEGImageNet"
#PROJECT_PATH = "lukas_heinrich-university-of-otago/eeg-object-eval"
RESULTS_ROOT = "results"

dataset_details = {
    "EEGImageNet": {
        "display_name": "filtered EEGImageNet",
        "chance_level": 1/40,
    },
    "Kaneshiro": {
        "display_name": "Kaneshiro",
        "chance_level": 1/6,
    },
    "ThingsEEG2": {
        "display_name": "ThingsEEG2",
        "chance_level": 1/1654,
    },
    "ThingsEEG2Averaged": {
        "display_name": "averaged ThingsEEG2",
        "chance_level": 1/1654,
    },
    "EOOD": {
        "display_name": "EOOD",
        "chance_level": 1/10,
    }
}

# list of metrics to process
METRICS = [
    "test_accuracy",
    "test_balanced_acc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_cohen_kappa",
    "test_auc",
]


# Order in which models will appear in the plot
MODEL_ORDER = ["ATMS", "BiLSTM", "EEGClip", "CAWMASASTST",
               "NiceEEG", "EEGNet", "EEGChannelNet", "CBraMod"]

# Mapping for display names for split types
SPLIT_LABELS = {
    "per_subject": "Within-Subject",
    "CV": "CV",
    "cross_subject": "LOSO"
}

SPLIT_ORDER = ["per_subject", "CV", "cross_subject"]


# --- FUNCTIONS ---
def initialize_api() -> wandb.Api:
    """Initializes and returns the WandB API."""
    return wandb.Api()


def download_runs(api: wandb.Api, project_path: str):
    """Fetch all runs from a WandB project."""
    return api.runs(project_path)


def fetch_table_df(run) -> pd.DataFrame:
    """
    Process a single run to extract the test metrics table into a DataFrame.
    Caches the result in a local directory to avoid re-downloading artifacts.
    Returns None if processing fails.
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{run.id}.pkl")

    # Check if the cache file exists
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            return df
        except Exception as e:
            print(f"Error loading cached data for run {run.id}: {e}")

    try:
        artifact = run.logged_artifacts()[0]  # Grab the first logged artifact
        table_artifact = artifact.get("test_metrics_table")
        df = pd.DataFrame(table_artifact.data, columns=table_artifact.columns)
        df.to_pickle(cache_file)
        return df
    except Exception as e:
        print(f"Error processing run {run.id}: {e}")
        return None


def fetch_all_runs_data(runs) -> pd.DataFrame:
    """
    Process runs in parallel using ThreadPoolExecutor and combine the resulting DataFrames.
    """
    dataframes = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        future_to_run = {executor.submit(fetch_table_df, run): run for run in runs}
        for future in as_completed(future_to_run):
            result = future.result()
            if result is not None:
                dataframes.append(result)
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        raise RuntimeError("No valid run data was retrieved.")


def compute_summary(final_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Group the DataFrame by model, dataset, and split then compute mean and std for the metric.
    Only uses rows where metric_col exists and is finite.
    Returns a DataFrame with columns: model_name, dataset_name, split_type, <metric_col>, <metric_col>_std, formatted
    The 'formatted' column uses percentage formatting only when metric_col contains 'accuracy'.
    """
    required_cols = ["model_name", "dataset_name", "split_type", metric_col]

    # drop rows where metric is missing / NaN
    df_metric = final_df[required_cols].copy()
    df_metric = df_metric.replace([np.inf, -np.inf], np.nan)
    df_metric = df_metric.dropna(subset=[metric_col])

    group_cols = ["model_name", "dataset_name", "split_type"]
    # Compute mean and std for provided metric
    mean_df = df_metric.groupby(group_cols)[metric_col].mean().reset_index().rename(columns={metric_col: metric_col})
    std_df = df_metric.groupby(group_cols)[metric_col].std().reset_index().rename(columns={metric_col: metric_col + "_std"})
    summary_df = pd.merge(mean_df, std_df, on=group_cols, how='left')

    # replace NaN std with 0.0 (single sample)
    summary_df[metric_col + "_std"] = summary_df[metric_col + "_std"].fillna(0.0)

    # Decide formatting: percentages only for accuracy-like metrics
    is_accuracy = "accuracy" in metric_col.lower()
    if is_accuracy:
        # format as percentage (one decimal)
        summary_df['formatted'] = summary_df.apply(
            lambda row: f"{row[metric_col] * 100:.1f}% (±{row[metric_col + '_std'] * 100:.1f}%)",
            axis=1
        )
    else:
        # format as raw numbers (three decimals)
        summary_df['formatted'] = summary_df.apply(
            lambda row: f"{row[metric_col]:.3f} (±{row[metric_col + '_std']:.3f})",
            axis=1
        )
    
    return summary_df


def create_numeric_pivots(summary_df: pd.DataFrame, metric_col: str):
    """
    Create pivot tables (mean & std) for plotting.
    If metric is an accuracy metric, convert to percentages for plotting (multiply by 100).
    - Drops any splits (columns) that have no data.
    - Drops any models (rows) that have no data.
    - Orders models according to MODEL_ORDER but only keeps those present.
    Returns pivot_mean, pivot_std (values already scaled for plotting: percent if accuracy, raw otherwise).
    """
    is_accuracy = metric_col.lower() in ["test_accuracy", "test_balanced_acc"]

    pivot_mean = summary_df.pivot(index="model_name", columns="split_type", values=metric_col)
    pivot_std = summary_df.pivot(index="model_name", columns="split_type", values=metric_col + "_std")

    if is_accuracy:
        pivot_mean = pivot_mean * 100.0
        pivot_std = pivot_std * 100.0
    
    # drop columns (splits) that are all NaN
    pivot_mean = pivot_mean.dropna(axis=1, how="all")
    pivot_std = pivot_std.dropna(axis=1, how="all")

    # drop rows (models) that are all NaN
    pivot_mean = pivot_mean.dropna(axis=0, how="all")
    pivot_std = pivot_std.dropna(axis=0, how="all")

    # keep MODEL_ORDER but only the models present
    present_models = [m for m in MODEL_ORDER if m in pivot_mean.index]
    # append any other models present but not in MODEL_ORDER (preserve alphabetical)
    other_models = [m for m in sorted(pivot_mean.index) if m not in present_models]
    ordered_models = present_models + other_models
    pivot_mean = pivot_mean.reindex(ordered_models)
    pivot_std = pivot_std.reindex(ordered_models)

    # keep SPLIT_ORDER but only splits present
    present_splits = [s for s in SPLIT_ORDER if s in pivot_mean.columns]
    other_splits = [s for s in sorted(pivot_mean.columns) if s not in present_splits]
    ordered_splits = present_splits + other_splits
    pivot_mean = pivot_mean.reindex(columns=ordered_splits)
    pivot_std = pivot_std.reindex(columns=ordered_splits)

    return pivot_mean, pivot_std


def plot_results(pivot_mean: pd.DataFrame, pivot_std: pd.DataFrame, 
                 metric: str, dataset_display_name: str, out_dir: str, 
                 chance_level: float):
    """
    Create a bar plot for the metric with error bars showing std.
    - If metric contains 'accuracy', values are assumed to be percentages and chance_level (a fraction) will be shown.
    - Otherwise, no percent sign or chance line will be used.
    Y-axis max is computed dynamically from the data.
    """
    is_accuracy = metric.lower() in ["test_accuracy", "test_balanced_acc"]
    models = pivot_mean.index.tolist()
    n_models = len(models)
    n_splits = pivot_mean.shape[1]
    x = np.arange(n_models)
    width = 0.8 / max(1, n_splits)

    plt.rcParams.update({'font.size': 14})
    # width scaled by models
    fig, ax = plt.subplots(figsize=(max(8, n_models * 0.9), 5))
    
    # Define hatch patterns for splits (repeat if needed)
    base_hatches = ['*', 'O', 'x', '.', '/', '//']
    hatches = [base_hatches[i % len(base_hatches)] for i in range(n_splits)]
    
    # Plot each split
    for i, split in enumerate(pivot_mean.columns):
        means = pivot_mean[split].values
        errs = pivot_std[split].values
        offset = (i - (n_splits - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=errs, capsize=3,
                      label=SPLIT_LABELS.get(split, split))
        for bar in bars:
            bar.set_hatch(hatches[i])
    
    # Plot chance level line only for accuracy metrics and when chance_level is provided
    if is_accuracy and (chance_level is not None):
        chance_pct = chance_level * 100.0
        ax.axhline(chance_pct, color='red', linestyle='--', linewidth=1,
                   label=f'Chance level ({chance_pct:.3f}%)')
    
    # Y label and title
    ylabel = f"{metric} (%)" if is_accuracy else metric
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"{dataset_display_name} - {metric} Across Split Strategies", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=14)

    ax.set_ylim(0, float(np.nanmax(pivot_mean.values + pivot_std.values)) * 1.15)

    ax.legend(fontsize=14)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout(pad=0.2)

    # save
    safe_metric = metric.replace("/", "_")
    pdf_file = os.path.join(out_dir, f"{safe_metric}.pdf")
    png_file = os.path.join(out_dir, f"{safe_metric}.png")
    plt.savefig(pdf_file, format="pdf", dpi=300)
    plt.savefig(png_file, format="png", dpi=200)
    plt.close()
    print(f"Saved plot to {os.path.abspath(pdf_file)} and {os.path.abspath(png_file)}")

def save_summaries(summary_df: pd.DataFrame, metric: str, out_dir: str):
    """
    Save numeric summary and formatted human-readable summary to CSV files.
    - raw numeric summary (mean, std)
    - formatted pivot (mean ± std) as CSV
    """
    # raw numeric
    raw_file = os.path.join(out_dir, f"{metric}_raw_summary.csv")
    summary_df.to_csv(raw_file, index=False)
    print(f"Saved numeric summary to {os.path.abspath(raw_file)}")

    # formatted pivot (models x splits with formatted strings)
    formatted_pivot = summary_df.pivot(index="model_name", columns="split_type", values="formatted")
    
    # reorder rows and cols sensibly
    rows = [m for m in MODEL_ORDER if m in formatted_pivot.index] + \
           [m for m in formatted_pivot.index if m not in MODEL_ORDER]
    cols = [s for s in SPLIT_ORDER if s in formatted_pivot.columns] + \
           [s for s in formatted_pivot.columns if s not in SPLIT_ORDER]
    formatted_pivot = formatted_pivot.reindex(index=rows, columns=cols)
    # rename columns for display
    formatted_pivot.columns = [SPLIT_LABELS.get(c, c) for c in formatted_pivot.columns]
    
    fmt_file = os.path.join(out_dir, f"{metric}_formatted_pivot.csv")
    formatted_pivot.fillna("-").to_csv(fmt_file)
    print(f"Saved formatted summary to {os.path.abspath(fmt_file)}")

# --- MAIN PROCESS ---
def process_metric_for_dataset(final_df: pd.DataFrame, dataset_name: str,
                               dataset_display_name: str, metric: str, out_root: str,
                               chance_level: float):
    """
    For a given dataset and metric: compute summary, create pivot tables,
    drop empty splits/models, plot and save summary & plot.
    """
    df_ds = final_df[final_df["dataset_name"] == dataset_name].copy()

    print(f"Processing dataset='{dataset_name}', metric='{metric}'...")
    summary_df = compute_summary(df_ds, metric)

    pivot_mean, pivot_std = create_numeric_pivots(summary_df, metric)

    # prepare output dir
    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # save summary files
    save_summaries(summary_df, metric, out_dir)

    # plot and save
    plot_results(pivot_mean, pivot_std, metric, dataset_display_name, out_dir, chance_level)


def main():
    api = initialize_api()
    runs = download_runs(api, PROJECT_PATH)

    print("Fetching run data from W&B...")
    final_df = fetch_all_runs_data(runs)

    # Determine which datasets to process: use dataset_details keys but only those present in the runs
    available_datasets = set(final_df["dataset_name"].unique())
    to_process = []
    for ds_key, ds_info in dataset_details.items():
        if ds_key in available_datasets:
            display_name = ds_info.get("display_name", ds_key)
            chance_level = ds_info.get("chance_level", None)
            to_process.append((ds_key, display_name, chance_level))

    for ds_name, ds_display, chance_level in to_process:
        print(f"\n=== Processing dataset: {ds_name} ===")
        for metric in METRICS:
            try:
                process_metric_for_dataset(final_df, ds_name, ds_display, metric,
                                           RESULTS_ROOT, chance_level)
            except KeyError as e:
                print(f"Skipping metric '{metric}' due to missing columns: {e}")
            except Exception as e:
                print(f"Unexpected error processing metric '{metric}': {e}")

    print("\nAll done. Results saved under:", os.path.abspath(RESULTS_ROOT))


if __name__ == "__main__":
    main()



