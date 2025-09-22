#!/usr/bin/env python3
"""
preprocess_EOOD.py

This script preprocesses anonymized raw FIF files and their corresponding CSV files
embedded in an HDF5 file (created by anonymize_and_package_dataset.py). It applies
filtering, notch filtering, and re-referencing to the raw EEG data, extracts events
from annotations, and extracts epochs either per event/image or per sequence. The epochs
can undergo baseline correction and optionally z-score normalization before being saved,
along with metadata, into an output HDF5 file.

Usage Example:
--------------
python preprocess_EOOD.py \
  --in_raw_dir ./anonymized_raws/ \
  --in_csv_h5 anonymized_and_csvs.h5 \
  --out_h5 processed_epochs_new.h5 \
  --use_sequence \
  --baseline_t -0.2 0.0

Required Inputs:
----------------
--in_raw_dir      Directory containing the anonymized raw FIF files (expected naming:
                  subject_{:02d}_session_{n}_raw.fif).
--in_csv_h5       HDF5 file created by the anonymize script that contains CSV data under
                  the path /sequences/subject_{:02d}/session_{}/csv.
--out_h5          Output HDF5 file where extracted epochs and metadata will be saved.

Optional Filtering Options:
---------------------------
--high_pass             High-pass cutoff frequency (Hz). Default is 0.1 Hz.
--low_pass              Low-pass cutoff frequency (Hz). Default is 100.0 Hz.
--notch_freqs           One or more notch filter frequencies (Hz) to remove. Default is [50.0].
--no_average_reference  Disable average reference re-referencing when set.

Epoch Extraction Options:
-------------------------
--baseline_t     Two floats (start and end in seconds) specifying the baseline window for
                correction. Omit to disable baseline correction.
--no_zscore_norm Disable z-score normalization when set.
--use_sequence   Extract epochs based on sequence information instead of per-event extraction.

Output:
-------
The processed epochs, along with accompanying metadata, are stored in an output HDF5 file
with the following structure:
  - Each epoch is saved as a separate group.
  - Each group contains a "data" dataset with the epoch data cast to float32.
  - Metadata for each epoch is stored in the group attributes.
"""

from __future__ import annotations
import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import mne

# -------------------------
# Processing helper funcs
# -------------------------


def preprocess_raw(
    raw: mne.io.BaseRaw,
    high_pass: float,
    low_pass: float,
    notch_freqs: List[float],
    average_reference: bool,
) -> mne.io.BaseRaw:
    """
    Preprocess the raw EEG data by applying filtering, notch filtering, and re-referencing.

    Parameters:
        raw (mne.io.BaseRaw): The raw EEG data with preload=True.
        high_pass (float): The low cutoff frequency for the high-pass filter.
        low_pass (float): The high cutoff frequency for the low-pass filter.
        notch_freqs (List[float]): A list of frequencies for notch filtering.
        average_reference (bool): If True, perform average re-referencing.

    Returns:
        mne.io.BaseRaw: The preprocessed raw data.
    """
    raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=False)
    raw.notch_filter(freqs=notch_freqs, verbose=False)
    if average_reference:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def get_events(raw: mne.io.BaseRaw, annotations_df: pd.DataFrame) -> np.ndarray:
    """
    Extract events from raw annotations and validate against a CSV annotations DataFrame.

    Parameters:
        raw (mne.io.BaseRaw): The preprocessed raw EEG data.
        annotations_df (pd.DataFrame): The CSV data containing annotations.

    Returns:
        np.ndarray: An array of events formatted for MNE processing.

    Raises:
        RuntimeError: If the number of extracted events does not match the number of CSV rows.
    """
    unique_codes = annotations_df["stim_code"].astype(int).unique().tolist()
    event_id_map = {str(code): int(code) for code in unique_codes}
    events, event_id = mne.events_from_annotations(
        raw, event_id=event_id_map, regexp=r"(\d+)", verbose=False
    )
    if len(events) != len(annotations_df):
        raise RuntimeError(
            f"Number of annotation events ({len(events)}) "
            f"does not match CSV rows ({len(annotations_df)})"
        )
    return events


def apply_baseline(
    data: np.ndarray,
    baseline_window: Tuple[float, float],
    baseline_start: int,
    sfreq: float,
) -> np.ndarray:
    """
    Apply baseline correction to epoch data using a specified time window.

    Parameters:
        data (np.ndarray): The epoch data array with shape (n_channels, n_times).
        baseline_window (Tuple[float, float]): The time window (in seconds) to compute the baseline.
        baseline_start (int): The starting index (in samples) corresponding to the baseline.
        sfreq (float): The sampling frequency of the data.

    Returns:
        np.ndarray: The baseline-corrected data.
    """
    times = (baseline_start + np.arange(data.shape[1])) / sfreq
    return mne.baseline.rescale(
        data[np.newaxis, ...],
        times=times,
        baseline=baseline_window,
        mode="mean",
        copy=False,
        verbose=False,
    )[0]


def apply_zscore(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Apply z-score normalization across the entire epoch.

    Parameters:
        data (np.ndarray): The epoch data array with shape (n_channels, n_times).
        sfreq (float): The sampling frequency of the data.

    Returns:
        np.ndarray: The z-score normalized data.
    """
    data_len = data.shape[1]
    times = np.arange(data_len) / sfreq
    return mne.baseline.rescale(
        data[np.newaxis, ...],
        times=times,
        baseline=(None, None),
        mode="zscore",
        copy=False,
        verbose=False,
    )[0]


def extract_epochs(
    df: pd.DataFrame,
    events: np.ndarray,
    raw: mne.io.BaseRaw,
    baseline_window: Optional[Tuple[float, float]],
    do_zscore: bool,
    use_sequence: bool,
    session_id: int,
    subject_id: int,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extract epochs from the raw data, either per event/image or per sequence, along with metadata.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the CSV annotation data.
        events (np.ndarray): The events array extracted from the raw data.
        raw (mne.io.BaseRaw): The preprocessed raw EEG data.
        baseline_window (Optional[Tuple[float, float]]): The baseline correction time window.
        do_zscore (bool): If True, apply z-score normalization to the epoch data.
        use_sequence (bool): If True, extract epochs based on sequence information.
        session_id (int): Identifier for the session.
        subject_id (int): Identifier for the subject.

    Returns:
        Tuple[List[np.ndarray], List[Dict]]:
            - A list of extracted epoch data arrays.
            - A corresponding list of metadata dictionaries.
    """
    sfreq = raw.info["sfreq"]
    epochs, meta_list = [], []
    baseline_start = int(round(baseline_window[0] * sfreq)) if baseline_window else 0

    groups = df.groupby("sequence_index") if use_sequence else [(None, df)]

    for seq_idx, group in groups:
        if use_sequence:
            row = group[group["sequence_ordinal"] == 1].iloc[0]
            ev_index = group.index[0]
            duration = float(row["sequence_duration_ms"]) / 1000.0
            n_samples = int(round(duration * sfreq))
            sample = int(events[ev_index, 0])
            start = sample + (baseline_start if baseline_window else 0)
            start = max(start, 0)
            stop = sample + n_samples
            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
            data = raw.get_data(picks=picks, start=start, stop=stop)
            if baseline_window:
                data = apply_baseline(data, baseline_window, baseline_start, sfreq)
                data = data[:, abs(baseline_start) :]
            if do_zscore:
                data = apply_zscore(data, sfreq)
            assert (
                data.shape[1] == n_samples
            ), f"Epoch length {data.shape[1]} != expected {n_samples}"
            meta = {
                "class_idx": int(row["class_id"]),
                "class_name": row["class_name"],
                "subject": subject_id,
                "session_id": session_id,
                "sequence_index": int(row["sequence_index"]),
                "type": row["type"],
            }
            epochs.append(data)
            meta_list.append(meta)
        else:
            for idx, row in group.iterrows():
                sample = int(events[idx, 0])
                duration = float(row["image_duration_ms"]) / 1000.0
                n_samples = int(round(duration * sfreq))
                start = sample + (baseline_start if baseline_window else 0)
                start = max(start, 0)
                stop = sample + n_samples
                picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
                data = raw.get_data(picks=picks, start=start, stop=stop)
                if baseline_window:
                    data = apply_baseline(data, baseline_window, baseline_start, sfreq)
                    data = data[:, abs(baseline_start) :]
                if do_zscore:
                    data = apply_zscore(data, sfreq)
                assert (
                    data.shape[1] == n_samples
                ), f"Epoch length {data.shape[1]} != expected {n_samples}"
                meta = {
                    "class_idx": int(row["class_id"]),
                    "class_name": row["class_name"],
                    "subject": subject_id,
                    "session_id": session_id,
                    "sequence_index": int(row["sequence_index"]),
                    "type": row["type"],
                    "sequence_ordinal": int(row["sequence_ordinal"]),
                    "image_idx": int(row["image_id"]),
                }
                epochs.append(data)
                meta_list.append(meta)

    return epochs, meta_list


def save_to_h5(
    out_fname: str,
    epochs: List[np.ndarray],
    metas: List[Dict],
    sfreq: float,
    ch_names: List[str],
) -> None:
    """
    Save the extracted epochs and their metadata to an HDF5 file.

    Parameters:
        out_fname (str): The output filename for the HDF5 file.
        epochs (List[np.ndarray]): A list of epoch data arrays.
        metas (List[Dict]): A list of metadata dictionaries corresponding to each epoch.
        sfreq (float): The sampling frequency of the data.
        ch_names (List[str]): List of channel names.

    Returns:
        None
    """
    with h5py.File(out_fname, "w") as f:
        f.attrs["sfreq"] = float(sfreq)
        f.attrs["channels"] = ch_names
        for idx, (epoch, meta) in enumerate(zip(epochs, metas)):
            grp = f.create_group(f"{idx}")
            grp.create_dataset("data", data=epoch.astype("float32"))
            for key, value in meta.items():
                grp.attrs[key] = value
                grp.attrs["epoch_idx"] = idx
    print(f"Saved {len(epochs)} epochs to {out_fname}")


def load_csv_from_h5(h5path: str, subject: int, session: int) -> pd.DataFrame:
    """
    Load CSV data from a specified location in an HDF5 file.

    Parameters:
        h5path (str): The file path to the HDF5 file.
        subject (int): The subject identifier.
        session (int): The session identifier.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the expected CSV dataset is not found in the HDF5 file.
    """
    with h5py.File(h5path, "r") as f:
        key = f"sequences/subject_{subject:02d}/session_{session}/csv"
        if key not in f:
            raise FileNotFoundError(f"CSV dataset not found in HDF5 at {key}")
        csv_bytes = f[key][()]
        if isinstance(csv_bytes, bytes):
            csv_text = csv_bytes.decode("utf-8")
        else:
            csv_text = str(csv_bytes)
        from io import StringIO

        df = pd.read_csv(StringIO(csv_text))
    return df


def run_preprocessing(
    in_raw_dir: str,
    in_csv_h5: str,
    out_h5: str,
    baseline_t: Optional[List[float]] = [-0.2, 0.0],
    high_pass: Optional[float] = 0.1,
    low_pass: Optional[float] = 100.0,
    notch_freqs: Optional[List[float]] = [50.0],
    average_reference: bool = True,
    zscore_norm: bool = True,
    use_sequence: bool = False,
) -> int:
    """
    Run the complete preprocessing pipeline:
      - Loop over subjects and sessions.
      - Load raw FIF and CSV data.
      - Preprocess the raw data.
      - Extract events and epochs.
      - Save the processed epochs and metadata into an HDF5 file.

    Parameters:
        in_raw_dir (str): Directory containing the raw FIF files.
        in_csv_h5 (str): HDF5 file path containing embedded CSV annotation data.
        out_h5 (str): Output HDF5 file for saving epochs.
        baseline_t (Optional[List[float]]): Two floats specifying the baseline window (seconds).
        high_pass (Optional[float]): High-pass filter cutoff frequency.
        low_pass (Optional[float]): Low-pass filter cutoff frequency.
        notch_freqs (Optional[List[float]]): Frequency values for notch filtering.
        average_reference (bool): Whether to apply average referencing.
        zscore_norm (bool): Whether to perform z-score normalization.
        use_sequence (bool): Whether to extract epochs per sequence (True) or per event (False).

    Returns:
        int: Exit code. 0 indicates success, 1 indicates that no epochs were extracted.
    """
    all_epochs: List[np.ndarray] = []
    all_meta: List[Dict] = []
    last_raw_info = None

    for subj in range(1, 12):
        for session in range(1, 4):
            raw_path = os.path.join(
                in_raw_dir, f"subject_{subj:02d}_session_{session}_raw.fif"
            )
            if not os.path.exists(raw_path):
                print(f"Raw file {raw_path} not found. Skipping.")
                continue

            print(f"\nLoading raw: {raw_path} (subject {subj}, session {session})")
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
            except Exception as e:
                print(f"  ERROR reading raw FIF {raw_path}: {e}. Skipping.")
                continue

            try:
                df = load_csv_from_h5(in_csv_h5, subj, session)
            except Exception as e:
                print(
                    f"  ERROR reading CSV from HDF5 for subject {subj}, session {session}: {e}. Skipping."
                )
                continue

            if "sequence_ordinal" not in df.columns:
                df["sequence_ordinal"] = df.groupby("sequence_index").cumcount() + 1

            try:
                raw = preprocess_raw(
                    raw,
                    high_pass=high_pass,
                    low_pass=low_pass,
                    notch_freqs=list(notch_freqs) if notch_freqs else [],
                    average_reference=average_reference,
                )
            except Exception as e:
                print(f"  ERROR during preprocess_raw: {e}. Skipping file.")
                continue

            try:
                events = get_events(raw, df)
            except RuntimeError as e:
                print(
                    f"  Error extracting events for subject {subj}, session {session}: {e}. Skipping."
                )
                continue

            last_raw_info = raw.info

            try:
                epochs, metas = extract_epochs(
                    df,
                    events,
                    raw,
                    baseline_window=tuple(baseline_t) if baseline_t else None,
                    do_zscore=zscore_norm,
                    use_sequence=use_sequence,
                    session_id=session,
                    subject_id=subj,
                )
            except AssertionError as e:
                print(
                    f"  Error extracting epochs for subject {subj}, session {session}: {e}. Skipping."
                )
                continue

            all_epochs.extend(epochs)
            all_meta.extend(metas)
            print(
                f"  Extracted {len(epochs)} epochs for subject {subj}, session {session}."
            )

    if not all_epochs:
        print("No epochs extracted. Exiting.")
        return 1

    picks = mne.pick_types(last_raw_info, meg=False, eeg=True, stim=False)
    ch_names = [last_raw_info["ch_names"][p] for p in picks]
    sfreq = float(last_raw_info["sfreq"])

    print(f"\nSaving a total of {len(all_epochs)} epochs to {out_h5}")
    save_to_h5(out_h5, all_epochs, all_meta, sfreq, ch_names)

    print("Done.")
    return 0


if __name__ == "__main__":

    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments for the preprocessing script.

        Returns:
            argparse.Namespace: Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Preprocess anonymized FIF raws and embedded CSVs into epochs HDF5"
        )
        parser.add_argument(
            "--in_raw_dir",
            default="../Datasets/EOOD/raws/",
            help="Directory with anonymized raw FIF files (expected naming: subject_{:02d}_session_{n}.fif)",
        )
        parser.add_argument(
            "--in_csv_h5",
            default="../Datasets/EOOD/raws/sequences.h5",
            help="HDF5 file created by anonymize script that contains CSVs under /csvs/subject_{:02d}/session_{}/csv",
        )
        parser.add_argument(
            "--out_h5",
            default="processed_epochs.h5",
            help="Output HDF5 file with epochs",
        )
        parser.add_argument(
            "--baseline_t",
            type=float,
            nargs=2,
            default=[-0.2, 0.0],
            help="Baseline window in seconds as two floats; omit to disable baseline correction",
        )
        parser.add_argument(
            "--high_pass",
            type=float,
            default=0.1,
            help="High-pass filter frequency (Hz)",
        )
        parser.add_argument(
            "--low_pass",
            type=float,
            default=100.0,
            help="Low-pass filter frequency (Hz)",
        )
        parser.add_argument(
            "--notch_freqs",
            nargs="*",
            type=float,
            default=[50.0],
            help="Notch filter frequencies (Hz)",
        )
        parser.add_argument(
            "--no_average_reference",
            action="store_false",
            dest="average_reference",
            default=True,
            help="Disable average reference",
        )
        parser.add_argument(
            "--no_zscore_norm",
            action="store_false",
            dest="zscore_norm",
            default=True,
            help="Disable z-score normalization",
        )
        parser.add_argument(
            "--use_sequence",
            action="store_true",
            default=False,
            help="Extract epochs per sequence (if not set, extracts per event/image)",
        )
        return parser.parse_args()

    args = parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    exit(run_preprocessing(**vars(args)))
