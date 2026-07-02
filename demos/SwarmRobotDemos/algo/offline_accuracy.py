# -*- coding: utf-8 -*-
"""Evaluate brainda's FBSCCA implementation on the Wang2016 dataset."""

from pathlib import Path
import warnings

import numpy as np
from mne import set_config
from sklearn.metrics import precision_score, recall_score, f1_score

from metabci.brainda.algorithms.decomposition import FBSCCA
from metabci.brainda.algorithms.decomposition.base import (
    generate_cca_references,
    generate_filterbank,
)
from metabci.brainda.datasets.tsinghua import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainflow.logger import get_logger


logger = get_logger("offline_accuracy")
warnings.filterwarnings("ignore")

PASSBANDS = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
STOPBANDS = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
N_HARMONICS = 5
OCCIPITAL_CHANNELS = [
    "PZ",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "O1",
    "OZ",
    "O2",
]
EPOCH_DURATION = 1.5


def raw_hook(raw, caches):
    """Apply the original preprocessing filter to each continuous recording."""
    raw.filter(
        5,
        55,
        l_trans_bandwidth=2,
        h_trans_bandwidth=5,
        phase="zero-double",
    )
    caches["raw_stage"] = caches.get("raw_stage", -1) + 1
    return raw, caches


def label_encoder(y, labels):
    """Encode dataset event labels as zero-based FBSCCA class indices."""
    encoded = np.empty_like(y)
    for index, label in enumerate(labels):
        encoded[y == label] = index
    return encoded


def normalize_trials(X):
    """Apply the same centering and scaling used by the previous implementation."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError("X must have shape (trials, channels, samples)")

    X = X - np.mean(X, axis=-1, keepdims=True)
    scale = np.std(X, axis=(-1, -2), keepdims=True)
    if np.any(scale == 0):
        raise ValueError("X contains a trial with zero variance")
    return X / scale


def build_fbscca(n_channels, n_samples, srate=250):
    """Create and initialize the native brainda FBSCCA decoder."""
    filterbank = generate_filterbank(PASSBANDS, STOPBANDS, srate)
    filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

    references = generate_cca_references(
        Wang2016._FREQS,
        srate=srate,
        T=n_samples / srate,
        n_harmonics=N_HARMONICS,
    )
    if references.shape[-1] != n_samples:
        raise RuntimeError(
            "CCA reference length does not match the EEG epoch length: "
            f"{references.shape[-1]} != {n_samples}"
        )

    model = FBSCCA(
        filterbank=filterbank,
        n_components=1,
        filterweights=filterweights,
        n_jobs=-1,
    )

    # FBSCCA is training-free. fit() initializes one SCCA estimator per band
    # and stores the reference signals; the dummy EEG values are not learned.
    dummy_X = np.zeros((1, n_channels, n_samples), dtype=float)
    model.fit(dummy_X, np.zeros(1, dtype=int), Yf=references)
    return model


def offline_validation(X, y, srate=250):
    """Evaluate calibration-free FBSCCA predictions on all supplied trials."""
    logger.info("FBSCCA offline validation started")
    X = normalize_trials(X)
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError("X and y contain different numbers of trials")

    model = build_fbscca(X.shape[-2], X.shape[-1], srate=srate)
    predicted = model.predict(X)

    accuracy = float(np.mean(predicted == y))
    precision = precision_score(
        y, predicted, average="weighted", zero_division=0)
    recall = recall_score(y, predicted, average="weighted", zero_division=0)
    f1 = f1_score(y, predicted, average="weighted", zero_division=0)

    logger.info(f"Current Model accuracy: {accuracy:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1 Score: {f1:.2f}")
    return accuracy, precision, recall, f1


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    set_config(
        "MNE_DATASETS_TSINGHUA_PATH", str(data_dir), set_env=True)
    logger.info(f"Wang2016 data path: {data_dir}")

    srate = 250
    stim_interval = [(0.5, 0.5 + EPOCH_DURATION)]
    subjects = list(range(1, 6))

    dataset = Wang2016()
    paradigm = SSVEP(
        channels=OCCIPITAL_CHANNELS,
        events=dataset.events,
        intervals=stim_interval,
        srate=srate,
    )
    paradigm.register_raw_hook(raw_hook)
    X, y, _ = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=-1,
        verbose=False,
    )
    y = label_encoder(y, np.unique(y))
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    accuracy, precision, recall, f1 = offline_validation(
        X, y, srate=srate)
    print(f"Current Model accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    main()
