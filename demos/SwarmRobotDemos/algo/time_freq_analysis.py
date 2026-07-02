# -*- coding: utf-8 -*-
"""Time-frequency analysis for Wang2016 SSVEP recordings.

The script loads one selected stimulus event, computes its trial-averaged
power spectrum, STFT, and narrow-band Hilbert envelope, then saves one figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mne import set_config
from scipy import signal

from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import (
    TimeFrequencyAnalysis,
)
from metabci.brainda.datasets.tsinghua import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainflow.logger import get_logger


logger = get_logger("SSVEP_Time_Freq_Analysis")
warnings.filterwarnings("ignore")

SRATE = 250
EPOCH_START = 0.5
EPOCH_DURATION = 1.5
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


def configure_data_path() -> Path:
    """Use the SwarmRobotDemos data directory for MNE datasets."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    set_config(
        "MNE_DATASETS_TSINGHUA_PATH", str(data_dir), set_env=True)
    logger.info(f"Wang2016 data path: {data_dir}")
    return data_dir


def raw_hook(raw, caches):
    """Retain the SSVEP fundamental frequencies and their harmonics."""
    raw.filter(
        4,
        60,
        l_trans_bandwidth=2,
        h_trans_bandwidth=5,
        phase="zero-double",
    )
    caches["raw_stage"] = caches.get("raw_stage", -1) + 1
    return raw, caches


def normalize_event_name(event: str) -> tuple[str, float]:
    """Convert a frequency argument to the event naming used by Wang2016."""
    frequency = float(event)
    return f"{frequency:.1f}", frequency


def load_event_data(subjects: list[int], event_name: str):
    """Load one event only, using the compact analysis channel/time selection."""
    dataset = Wang2016()
    if event_name not in dataset.events:
        available = ", ".join(dataset.events.keys())
        raise ValueError(
            f"Unknown Wang2016 event {event_name!r}. Available events: {available}"
        )

    paradigm = SSVEP(
        channels=OCCIPITAL_CHANNELS,
        events=[event_name],
        intervals=[(EPOCH_START, EPOCH_START + EPOCH_DURATION)],
        srate=SRATE,
    )
    paradigm.register_raw_hook(raw_hook)

    # Dataset download/extraction and MNE path updates are deliberately kept in
    # the main process. This avoids stdin prompts and interrupted files in loky
    # workers while loading uncached subjects.
    X, _, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=1,
        verbose=False,
    )

    event_mask = meta["event"].astype(str).to_numpy() == event_name
    X = np.asarray(X[event_mask], dtype=float)
    if X.size == 0:
        raise ValueError(f"No trials found for event {event_name}")
    if not np.all(np.isfinite(X)):
        raise ValueError("Loaded EEG data contains NaN or Inf")

    logger.info(
        f"Loaded event {event_name}: X shape={X.shape}, subjects={subjects}")
    return X


def calculate_snr(
    frequencies: np.ndarray,
    power: np.ndarray,
    target_frequency: float,
    noise_half_width: float = 2.0,
) -> tuple[float, int]:
    """Calculate target-bin SNR against neighboring frequency bins."""
    target_index = int(np.argmin(np.abs(frequencies - target_frequency)))
    noise_mask = (
        (frequencies >= target_frequency - noise_half_width)
        & (frequencies <= target_frequency + noise_half_width)
    )
    noise_mask[target_index] = False
    noise_power = power[noise_mask]
    if noise_power.size == 0:
        raise ValueError("The epoch is too short to estimate neighboring noise")

    eps = np.finfo(float).tiny
    snr_db = 10 * np.log10(
        max(power[target_index], eps) / max(float(np.mean(noise_power)), eps)
    )
    return float(snr_db), target_index


def analyze_event(
    X: np.ndarray,
    event_name: str,
    target_frequency: float,
    channel: str,
    subjects: list[int],
    output_dir: Path,
    show: bool = False,
) -> tuple[list[Path], float]:
    """Create separate PSD, STFT, and Hilbert figures for one event/channel."""
    channel = channel.upper()
    if channel not in OCCIPITAL_CHANNELS:
        raise ValueError(
            f"Channel {channel!r} must be one of {OCCIPITAL_CHANNELS}")

    channel_index = OCCIPITAL_CHANNELS.index(channel)
    mean_signal = np.mean(X[:, channel_index, :], axis=0)
    mean_signal = signal.detrend(mean_signal, type="linear")

    frequencies, power = signal.periodogram(
        mean_signal,
        fs=SRATE,
        window="hann",
        detrend=False,
        scaling="density",
    )
    snr_db, target_index = calculate_snr(
        frequencies, power, target_frequency)

    feature = TimeFrequencyAnalysis(SRATE)
    nperseg = min(128, mean_signal.size)
    stft_frequencies, stft_times, stft_values = feature.fun_stft(
        mean_signal,
        nperseg=nperseg,
        noverlap=3 * nperseg // 4,
        nfft=max(512, nperseg),
    )

    low = max(1.0, target_frequency - 1.0)
    high = min(SRATE / 2 - 1.0, target_frequency + 1.0)
    target_filter = signal.butter(
        4, [low, high], btype="bandpass", fs=SRATE, output="sos")
    target_band = signal.sosfiltfilt(target_filter, mean_signal)
    _, _, _, phase, envelope = feature.fun_hilbert(target_band)

    times = np.arange(mean_signal.size) / SRATE
    eps = np.finfo(float).tiny
    subject_text = "-".join(str(subject) for subject in subjects)
    title_prefix = f"Wang2016 subjects {subject_text} — {event_name} Hz"

    psd_figure, psd_axis = plt.subplots(figsize=(11, 5))
    psd_db = 10 * np.log10(np.maximum(power, eps))
    visible_psd = psd_db[(frequencies >= 1) & (frequencies <= 60)]
    psd_axis.plot(frequencies, psd_db, color="tab:blue")
    psd_axis.axvline(target_frequency, color="tab:red", linestyle="--")
    psd_axis.scatter(
        frequencies[target_index], psd_db[target_index], color="tab:red", zorder=3)
    psd_axis.set(
        xlim=(0, 60),
        ylim=(float(np.min(visible_psd)) - 5, float(np.max(visible_psd)) + 5),
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB µV²/Hz)",
        title=(
            f"{title_prefix}\nPSD — {channel}, SNR {snr_db:.2f} dB"
        ),
    )
    psd_axis.grid(True, alpha=0.3)
    psd_figure.tight_layout()

    stft_figure, stft_axis = plt.subplots(figsize=(11, 5))
    stft_db = 20 * np.log10(np.maximum(np.abs(stft_values), eps))
    visible_stft = stft_db[stft_frequencies <= 60]
    stft_vmin, stft_vmax = np.percentile(visible_stft, [5, 99])
    mesh = stft_axis.pcolormesh(
        stft_times,
        stft_frequencies,
        stft_db,
        shading="auto",
        vmin=stft_vmin,
        vmax=stft_vmax,
    )
    stft_axis.axhline(target_frequency, color="white", linestyle="--")
    stft_axis.set(
        ylim=(0, 60),
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        title=f"{title_prefix}\nSTFT magnitude — {channel}",
    )
    stft_figure.colorbar(mesh, ax=stft_axis, label="Magnitude (dB)")
    stft_figure.tight_layout()

    hilbert_figure, hilbert_axis = plt.subplots(figsize=(11, 5))
    hilbert_axis.plot(
        times, target_band, label=f"{low:.1f}–{high:.1f} Hz signal")
    hilbert_axis.plot(times, envelope, label="Hilbert envelope", linewidth=2)
    phase_axis = hilbert_axis.twinx()
    phase_axis.plot(
        times, phase, color="tab:green", alpha=0.25, label="Phase")
    hilbert_axis.set(
        xlim=(0, EPOCH_DURATION),
        xlabel="Time (s)",
        ylabel="Amplitude (µV)",
        title=f"{title_prefix}\nTarget-band Hilbert analysis — {channel}",
    )
    phase_axis.set_ylabel("Phase (rad)")
    lines = hilbert_axis.get_lines() + phase_axis.get_lines()
    hilbert_axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
    hilbert_axis.grid(True, alpha=0.3)
    hilbert_figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"subjects-{subject_text}_event-{event_name}_channel-{channel}"
    figures = {
        output_dir / f"{output_stem}_psd.png": psd_figure,
        output_dir / f"{output_stem}_stft.png": stft_figure,
        output_dir / f"{output_stem}_hilbert.png": hilbert_figure,
    }
    for output_path, figure in figures.items():
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        logger.info(f"Analysis figure saved to: {output_path}")
    logger.info(f"Target-frequency SNR: {snr_db:.2f} dB")

    if show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return list(figures), snr_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Wang2016 SSVEP PSD, STFT, and Hilbert envelope.")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=[1], help="Subject IDs")
    parser.add_argument(
        "--event", default="8.0", help="Stimulus frequency, for example 8.0")
    parser.add_argument(
        "--channel", default="PO5", help="One of the nine occipital channels")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Figure output directory")
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the figure after saving it (default)",
    )
    display_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save the figure without opening a display window",
    )
    parser.set_defaults(show=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = configure_data_path()
    event_name, target_frequency = normalize_event_name(args.event)
    output_dir = args.output_dir or data_dir / "time_freq_analysis"

    X = load_event_data(args.subjects, event_name)
    output_paths, snr_db = analyze_event(
        X,
        event_name,
        target_frequency,
        args.channel,
        args.subjects,
        output_dir,
        show=args.show,
    )
    for output_path in output_paths:
        print(f"Figure: {output_path}")
    print(f"SNR: {snr_db:.2f} dB")


if __name__ == "__main__":
    main()
