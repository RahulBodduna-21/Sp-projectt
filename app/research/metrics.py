import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.fft import rfft, rfftfreq
from app.core.config import get_project_root
from app.core.logging_config import logger

# Try importing matplotlib safely in case Application Control blocks DLL execution
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    logger.warning(f"Matplotlib import blocked by system Application Control: {e}. Figure image plots will be skipped.")

def get_figures_dir() -> str:
    fig_dir = os.path.join(get_project_root(), "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png"
) -> str:
    """Generates and saves publication-quality confusion matrix plot."""
    filepath = os.path.join(get_figures_dir(), filename)
    if not MATPLOTLIB_AVAILABLE:
        return filepath

    try:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label"
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d"
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
    return filepath

def plot_class_distribution(
    label_counts: Dict[str, int],
    title: str = "Dataset Label Distribution",
    filename: str = "class_distribution.png"
) -> str:
    """Plots and saves class distribution bar chart."""
    filepath = os.path.join(get_figures_dir(), filename)
    if not MATPLOTLIB_AVAILABLE:
        return filepath

    try:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        classes = list(label_counts.keys())
        counts = list(label_counts.values())

        colors = ["#2563EB", "#D97706", "#DC2626", "#059669", "#4B5563"]
        bar_colors = colors[:len(classes)]

        bars = ax.bar(classes, counts, color=bar_colors, edgecolor="black", width=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Road Condition Class", fontsize=11)
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.01), f"{int(yval)}", ha="center", va="bottom", fontweight="bold")

        fig.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
    return filepath

def plot_raw_vs_filtered(
    df_dsp: pd.DataFrame,
    filename: str = "raw_vs_filtered.png"
) -> str:
    """Plots comparison of raw signal vs filtered signal over time."""
    filepath = os.path.join(get_figures_dir(), filename)
    if not MATPLOTLIB_AVAILABLE:
        return filepath

    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=300)
        t = df_dsp["elapsed_time_s"].values if "elapsed_time_s" in df_dsp.columns else np.arange(len(df_dsp))

        channels = [("acc_x", "Acc X (m/s²)"), ("acc_y", "Acc Y (m/s²)"), ("acc_z", "Acc Z (m/s²)")]
        for i, (col, label) in enumerate(channels):
            raw_col = f"{col}_raw_hampel" if f"{col}_raw_hampel" in df_dsp.columns else col
            filt_col = col

            axes[i].plot(t[:1000], df_dsp[raw_col].values[:1000], color="#9CA3AF", alpha=0.7, label="Raw/Unfiltered")
            axes[i].plot(t[:1000], df_dsp[filt_col].values[:1000], color="#2563EB", linewidth=1.5, label="Butterworth Filtered")
            axes[i].set_ylabel(label, fontsize=10)
            axes[i].grid(True, linestyle="--", alpha=0.5)
            axes[i].legend(loc="upper right")

        axes[-1].set_xlabel("Elapsed Time (seconds)", fontsize=11)
        fig.suptitle("Raw vs Butterworth Filtered Acceleration Signals", fontsize=13, fontweight="bold")
        fig.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
    return filepath

def plot_filtered_vs_calibrated(
    df_dsp: pd.DataFrame,
    filename: str = "filtered_vs_calibrated.png"
) -> str:
    """Plots comparison of filtered acceleration vs Median-MAD calibrated acceleration."""
    filepath = os.path.join(get_figures_dir(), filename)
    if not MATPLOTLIB_AVAILABLE:
        return filepath

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)
        t = df_dsp["elapsed_time_s"].values if "elapsed_time_s" in df_dsp.columns else np.arange(len(df_dsp))

        filt_mag = df_dsp["acc_magnitude"].values if "acc_magnitude" in df_dsp.columns else np.zeros(len(df_dsp))
        calib_mag = df_dsp["acc_magnitude_calibrated"].values if "acc_magnitude_calibrated" in df_dsp.columns else np.zeros(len(df_dsp))

        ax1.plot(t[:1000], filt_mag[:1000], color="#D97706", linewidth=1.2)
        ax1.set_title("Filtered Acceleration Magnitude (m/s²)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Magnitude (m/s²)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(t[:1000], calib_mag[:1000], color="#059669", linewidth=1.2)
        ax2.set_title("Median-MAD Calibrated Acceleration Z-Score", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Calibrated Z-Score", fontsize=10)
        ax2.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Signal Calibration Effect: Filtered vs Median-MAD Calibrated", fontsize=13, fontweight="bold")
        fig.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
    return filepath

def plot_fft_analysis(
    df_dsp: pd.DataFrame,
    sampling_rate_hz: float = 50.0,
    filename: str = "fft_analysis.png"
) -> str:
    """Plots time domain signal and its corresponding FFT magnitude frequency spectrum."""
    filepath = os.path.join(get_figures_dir(), filename)
    if not MATPLOTLIB_AVAILABLE:
        return filepath

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=300)
        t = df_dsp["elapsed_time_s"].values if "elapsed_time_s" in df_dsp.columns else np.arange(len(df_dsp))

        mag = df_dsp["acc_magnitude_calibrated"].values if "acc_magnitude_calibrated" in df_dsp.columns else df_dsp["acc_magnitude"].values
        sub_sig = mag[:500] - np.mean(mag[:500])

        ax1.plot(t[:500], sub_sig, color="#2563EB", linewidth=1.2)
        ax1.set_title("Time-Domain Dynamic Acceleration Magnitude", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Elapsed Time (seconds)", fontsize=10)
        ax1.set_ylabel("Amplitude", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        n = len(sub_sig)
        fft_vals = np.abs(rfft(sub_sig))
        freqs = rfftfreq(n, d=1.0 / sampling_rate_hz)

        # Exclude DC component
        ac_mask = freqs > 0.1
        ax2.plot(freqs[ac_mask], fft_vals[ac_mask], color="#DC2626", linewidth=1.5)
        ax2.set_title("FFT Frequency Spectrum (scipy.fft)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)", fontsize=10)
        ax2.set_ylabel("Magnitude", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Frequency Domain (FFT) Spectral Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save {filename}: {e}")
    return filepath

def plot_jerk_and_impact(
    df_dsp: pd.DataFrame,
    filename_jerk: str = "jerk_analysis.png",
    filename_impact: str = "impact_score.png"
) -> Tuple[str, str]:
    """Plots jerk signal and calculated impact score over time."""
    path_jerk = os.path.join(get_figures_dir(), filename_jerk)
    path_impact = os.path.join(get_figures_dir(), filename_impact)
    if not MATPLOTLIB_AVAILABLE:
        return path_jerk, path_impact

    try:
        t = df_dsp["elapsed_time_s"].values if "elapsed_time_s" in df_dsp.columns else np.arange(len(df_dsp))

        # Jerk plot
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        jerk_vals = df_dsp["jerk"].values if "jerk" in df_dsp.columns else np.zeros(len(df_dsp))
        ax.plot(t[:1000], jerk_vals[:1000], color="#7C3AED", linewidth=1.2)
        ax.set_title("Numerical Jerk Signal (dAcc/dt)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Jerk (m/s³)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plt.savefig(path_jerk, bbox_inches="tight")
        plt.close(fig)

        # Impact Score plot
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        impact_vals = df_dsp["impact_score"].values if "impact_score" in df_dsp.columns else np.zeros(len(df_dsp))
        ax.plot(t[:1000], impact_vals[:1000], color="#DC2626", linewidth=1.2)
        ax.set_title("DSP Signal-Derived Impact Score Over Time", fontsize=12, fontweight="bold")
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Impact Score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plt.savefig(path_impact, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to save jerk/impact plots: {e}")

    return path_jerk, path_impact

def generate_all_research_figures(
    df_valid: pd.DataFrame,
    df_dsp: pd.DataFrame,
    cm: np.ndarray,
    class_names: List[str]
) -> Dict[str, str]:
    """Generates all 7 research figures safely."""
    paths = {}
    label_counts = df_valid["ground_truth_label"].value_counts().to_dict() if "ground_truth_label" in df_valid.columns else df_valid["label"].value_counts().to_dict()
    
    paths["class_distribution"] = plot_class_distribution(label_counts)
    paths["confusion_matrix"] = plot_confusion_matrix(cm, class_names)
    paths["raw_vs_filtered"] = plot_raw_vs_filtered(df_dsp)
    paths["filtered_vs_calibrated"] = plot_filtered_vs_calibrated(df_dsp)
    paths["fft_analysis"] = plot_fft_analysis(df_dsp)
    pj, pi = plot_jerk_and_impact(df_dsp)
    paths["jerk_analysis"] = pj
    paths["impact_score"] = pi

    if MATPLOTLIB_AVAILABLE:
        logger.info(f"Generated all {len(paths)} research figures in results/figures/")
    else:
        logger.warning("Matplotlib DLL execution was blocked by Windows Application Control policy. Core pipeline completed successfully.")
    return paths
