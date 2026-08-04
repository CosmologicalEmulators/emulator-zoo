import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validation_plots.py ARTIFACT_DIRECTORY")
    artifact = Path(sys.argv[1]).resolve()
    metrics = np.load(artifact / "validation_metrics.npz")
    report = json.loads((artifact / "validation_report.json").read_text())
    spectrum = report["spectrum"]
    plots = artifact / "validation_plots"
    plots.mkdir(exist_ok=True)

    ell = metrics["ell_dense"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ell, metrics["dense_knox_p50"], label="dense median")
    ax.plot(ell, metrics["dense_knox_p95"], label="dense p95")
    ax.plot(metrics["ell_training"], metrics["node_knox_p95"], "o", ms=2, label="node p95")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="1 Knox sigma")
    ax.set(xlabel=r"$\ell$", ylabel="absolute error / cosmic-variance error")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "knox_error_vs_ell.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(metrics["dense_sample_rms"][np.isfinite(metrics["dense_sample_rms"])], bins=60)
    ax.set(xlabel="per-sample RMS Knox error", ylabel="number of validation samples")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "sample_rms_knox_histogram.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
