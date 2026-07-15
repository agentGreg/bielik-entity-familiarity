from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_auroc_vs_depth(auroc_by_metric: dict, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, by_layer in auroc_by_metric.items():
        layers = sorted(by_layer)
        ax.plot(layers, [by_layer[l] for l in layers], marker="o", label=name)
    ax.axhline(0.75, ls="--", color="gray", label="threshold 0.75")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs layer depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
