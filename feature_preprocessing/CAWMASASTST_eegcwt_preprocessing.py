import os
import sys
import torch
import numpy as np
from multiprocessing import set_start_method, Pool, cpu_count
import pywt


def make_cwt_scales(sampling_rate: float):
    """
    Compute a concatenated array of wavelet scales that cover:
      - delta  0.5-4 Hz
      - theta  4-8   Hz
      - alpha  8-12  Hz
      - beta   12-30 Hz
      - gamma  30-50 Hz
    Returns:
      scales: np.ndarray, shape [n_scales]
      freqs : np.ndarray, same shape, the actual pseudo-frequencies in Hz
    """
    dt      = 1.0 / sampling_rate
    f_psi   = pywt.central_frequency('mexh')  # or 'morl' if you prefer
    # define each band with a reasonable number of steps
    band_defs = {
        'delta': np.linspace(0.5,   4.0,  num=5,  endpoint=False),
        'theta': np.linspace(4.0,   8.0,  num=5,  endpoint=False),
        'alpha': np.linspace(8.0,  12.0,  num=5,  endpoint=False),
        'beta' : np.linspace(12.0, 30.0,  num=5, endpoint=False),
        'gamma': np.linspace(30.0, 50.0,  num=5, endpoint=True),
    }
    # stack all bands
    all_freqs = np.concatenate([freqs for freqs in band_defs.values()])
    # sort
    all_freqs = np.sort(all_freqs)
    # convert to scales
    scales = f_psi / (all_freqs * dt)
    return scales, all_freqs

# ─── Worker ────────────────────────────────────────────────────────────────────

def _process_trial(args):
    """
    Worker function: load one .pt, compute CWT, save cwt_<orig>.pt
    args = (full_path, scales, sampling_rate, overwrite )
    """
    trial_path, scales, sampling_rate, overwrite  = args
    base, filename = os.path.split(trial_path)
    # save with "cwt_" prefix
    out_path = os.path.join(base, "cwt_" + filename)

    if not overwrite and os.path.exists(out_path):
        # skip if out file exists and we're not overwriting
        return trial_path, False

    try:
        trial = torch.load(trial_path, weights_only=False)  # expect [channels, time]

        if not isinstance(trial, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(trial)} at {trial_path}")
        if trial.ndim != 2:
            raise ValueError(f"Expected [chan, time], got {trial.shape} at {trial_path}")

        # compute CWT per channel
        coeffs = []
        for ch in range(trial.shape[0]):
            c, _ = pywt.cwt(trial[ch].cpu().numpy(), scales, 'mexh', 1.0 / sampling_rate)
            coeffs.append(c)

        # stack into tensor [channels, scales, time]
        cwt_tensor = torch.from_numpy(np.stack(coeffs, axis=0))
        torch.save(cwt_tensor, out_path)

        # deleting because of memory leakage...
        del trial, coeffs, cwt_tensor
        return trial_path, True
    except Exception as e:
        print(f"ERROR  {filename}: {e}")
        return trial_path, False

def process_dataset(root, sampling_rate, overwrite = True):
    scales, freqs = make_cwt_scales(sampling_rate)
    print(f"CWT Preprocessing: sampling_rate={sampling_rate} Hz → {len(scales)} scales from {freqs.min():.1f}-{freqs.max():.1f} Hz")

    # collect all .pt files in this folder
    pt_files = [os.path.join(root, fn)
                    for fn in os.listdir(root) if fn.endswith('.pt') and fn.startswith('trial')]
    n = len(pt_files)
    if n == 0:
        return
    print(f"  ↳ {n} trials, processing on {cpu_count()} cores…")

    # prepare args list for Pool
    tasks = [(trial_path, scales, sampling_rate, overwrite) for trial_path in pt_files]

    # parallel map
    processed = 0
    skipped   = 0

    try:
        set_start_method('spawn', force=True)
        with Pool(processes=cpu_count()) as pool:
            for i, (_, did) in enumerate(pool.imap_unordered(_process_trial, tasks), start=1):
                if did:
                    processed += 1
                else:
                    skipped += 1
                # simple 10%-step progress
                if i % max(1, n//10) == 0 or i == n:
                    print(f"    {i}/{n} done (new: {processed}, skipped: {skipped})")
    except KeyboardInterrupt:
        print("Interrupted by user, terminating workers…", file=sys.stderr)
        pool.terminate()
        pool.join()
        sys.exit(1)
