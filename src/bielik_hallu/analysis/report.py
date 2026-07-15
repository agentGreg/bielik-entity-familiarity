import math

import pandas as pd

from bielik_hallu.analysis.auroc import auroc_per_layer
from bielik_hallu.extract.extended import EXTENDED_METRICS

# Primary contrast for the extended-metrics report: KNOWN(0) vs FABRICATED(1).
EXTENDED_PRIMARY_POSITIVE = "FABRICATED"
EXTENDED_PRIMARY_NEGATIVE = "KNOWN"


def extended_auroc_curve(sig: pd.DataFrame, metric: str, point: str) -> dict[int, float]:
    """Per-layer AUROC for one extended metric at one point on the primary contrast.

    NaN metric rows (e.g. attention entropy on the layer-N residual row) are
    dropped, so each metric is evaluated only on its native layers.
    """
    subset = sig[(sig["point"] == point)
                 & (sig["condition"].isin(
                     [EXTENDED_PRIMARY_POSITIVE, EXTENDED_PRIMARY_NEGATIVE]))]
    subset = subset.dropna(subset=[metric])
    metric_by_layer = {int(layer): grp[metric].to_numpy()
                       for layer, grp in subset.groupby("layer")}
    labels_by_layer = {int(layer): (grp["condition"] == EXTENDED_PRIMARY_POSITIVE)
                       .astype(int).to_numpy()
                       for layer, grp in subset.groupby("layer")}
    if not metric_by_layer:
        return {}
    any_layer = next(iter(labels_by_layer))
    return auroc_per_layer(metric_by_layer, labels_by_layer[any_layer])


def build_extended_report(sig: pd.DataFrame, model_id: str) -> dict:
    """Assemble the extended-metrics AUROC report from an extended_signals frame.

    Same structure emitted by run_extended.py: per metric, per point
    (prompt/entity), best AUROC + best layer + the full per-layer curve on the
    KNOWN-vs-FABRICATED contrast.
    """
    report = {
        "model_id": model_id,
        "contrast": f"{EXTENDED_PRIMARY_NEGATIVE}(0) vs {EXTENDED_PRIMARY_POSITIVE}(1)",
        "note": ("AUROC values are separability scores max(auc, 1 - auc); "
                 "metric direction may flip across layers/models. Layer indices "
                 "are native per metric family: residual stream 0..N for "
                 "effective_rank/logitlens_entropy, attention blocks 0..N-1 "
                 "for attn_entropy_*."),
        "metrics": {},
    }
    for metric in EXTENDED_METRICS:
        report["metrics"][metric] = {}
        for point in ("prompt", "entity"):
            curve = extended_auroc_curve(sig, metric, point)
            finite = {layer: v for layer, v in curve.items() if v == v}
            best_layer = max(finite, key=finite.get) if finite else None
            report["metrics"][metric][point] = {
                "best_auroc": finite[best_layer] if best_layer is not None else None,
                "best_layer": best_layer,
                "per_layer": curve,
            }
    return report


def go_no_go_summary(probe_auroc, global_auroc, threshold: float = 0.75) -> dict:
    usable_probe = {
        layer: v for layer, v in probe_auroc.items() if not math.isnan(v)
    }

    if not usable_probe:
        probe_best_layer = None
        probe_max = float("nan")
    else:
        probe_best_layer = max(usable_probe, key=usable_probe.get)
        probe_max = usable_probe[probe_best_layer]

    global_values = [
        v for m in global_auroc.values() for v in m.values() if not math.isnan(v)
    ]
    global_max = max(global_values, default=0.0)

    if probe_best_layer is None:
        verdict, kind = "NO-GO", "none"
    elif probe_max >= threshold and global_max >= threshold:
        verdict, kind = "GO", "volumetric"
    elif probe_max >= threshold:
        verdict, kind = "GO", "directional"
    else:
        verdict, kind = "NO-GO", "none"

    return {
        "probe_max": probe_max,
        "probe_best_layer": probe_best_layer,
        "global_max": global_max,
        "verdict": verdict,
        "signal_kind": kind,
    }
