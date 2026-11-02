"""AutoGluon model selection, image-wide prediction, metrics, and core-stick views.
For the labelling of the classification, the visualization and label is hardcoded using the existing data in this project"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix,
)

MODEL_FAMILIES = {
    "LightGBM": "GBM", "CatBoost": "CAT", "RandomForest": "RF",
    "ExtraTrees": "XT", "XGBoost": "XGB", "NeuralNetwork": "NN_TORCH",
}

AUTOML_HYPERPARAMETERS = {
    "NN_TORCH": {},
    "FASTAI": {},
    "RF": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
    ],
    "XT": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
    ],
    "CAT": {},
    "XGB": {},
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {},
    ],
}


def fit_tabular_model(
    X, y, *, output_dir, mode="automl", single_model="LightGBM",
    presets="medium_quality", time_limit=3600, num_bag_folds=5, 
    num_bag_sets=1, num_stack_levels=1, sample_weight=True,
    random_seed=0, verbosity=2,
):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install AutoGluon Tabular to train the classifier.") from exc
    if mode not in {"automl", "single_model"}:
        raise ValueError("mode must be 'automl' or 'single_model'.")
    if mode == "single_model" and single_model not in MODEL_FAMILIES:
        raise ValueError(f"single_model must be one of {list(MODEL_FAMILIES)}")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValueError("X and y must be aligned, nonempty training arrays.")
    columns = [f"band_{i:03d}" for i in range(X.shape[1])]
    train = pd.DataFrame(X, columns=columns)
    train["label"] = y
    counts = pd.Series(y).value_counts()
    if sample_weight:
        weights = {int(cls): float(len(y) / (len(counts) * count)) for cls, count in counts.items()}
        train["__class_weight__"] = train["label"].map(weights).astype(np.float32)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    predictor_kwargs = {
        "label": "label", "path": str(output_dir),
        "problem_type": "multiclass", "eval_metric": "balanced_accuracy",
    }
    if sample_weight:
        predictor_kwargs["sample_weight"] = "__class_weight__"
    predictor = TabularPredictor(**predictor_kwargs)
    fit_kwargs = {
        "train_data": train,
        "time_limit": time_limit,
        "num_bag_folds": int(num_bag_folds),
        "num_bag_sets": int(num_bag_sets),
        "num_stack_levels": int(num_stack_levels),
        "verbosity": int(verbosity),
    }
    if mode == "automl":
        fit_kwargs["presets"] = presets
        fit_kwargs["hyperparameters"] = AUTOML_HYPERPARAMETERS
    else:
        fit_kwargs["hyperparameters"] = {MODEL_FAMILIES[single_model]: {}}
    predictor.fit(**fit_kwargs)
    leaderboard = predictor.leaderboard(data=train, silent=True)
    return predictor, leaderboard, columns


def predict_full_cube(predictor, cube, normalizer, feature_columns, *, batch_size=200_000,
                      smoke_test=False, smoke_max_pixels=200_000, model=None):
    """Predict valid image pixels in batches; invalid and skipped pixels remain NaN."""
    cube = np.asarray(cube)
    h, w, bands = cube.shape
    n_bands = len(feature_columns)
    if bands < n_bands:
        raise ValueError(f"Cube has {bands} bands but model expects {n_bands}.")
    flat = cube[..., :n_bands].reshape(-1, n_bands)
    valid = np.isfinite(flat).all(axis=1) & (flat != 0).any(axis=1)
    valid_indices = np.flatnonzero(valid)
    if smoke_test:
        valid_indices = valid_indices[:int(smoke_max_pixels)]
    prediction = np.full(h * w, np.nan, dtype=np.float32)
    for start in range(0, len(valid_indices), int(batch_size)):
        idx = valid_indices[start:start + int(batch_size)]
        frame = pd.DataFrame(normalizer.transform(flat[idx]), columns=feature_columns)
        kwargs = {} if model in (None, "best") else {"model": model}
        prediction[idx] = np.asarray(predictor.predict(frame, **kwargs), dtype=np.float32)
        print(f"Predicted {min(start + len(idx), len(valid_indices)):,} / {len(valid_indices):,} valid pixels")
    return prediction.reshape(h, w)


def evaluate_maps(prediction_map, pixel_reference_map, full_reference_map,
                  *, mineral_labels=(101, 102, 103, 104, 105, 106), exclude_label=107): # Hard coded, adjust accordingly
    """Evaluate against the pixel-merged map and full reference map."""
    prediction = np.asarray(prediction_map, dtype=float)
    results = {}
    for name, reference in (("pixel_merged_reference", pixel_reference_map),
                            ("full_reference", full_reference_map)):
        reference = np.asarray(reference, dtype=float)
        if reference.shape != prediction.shape:
            raise ValueError(f"{name} shape {reference.shape} differs from prediction {prediction.shape}.")
        mask = np.isfinite(reference) & np.isfinite(prediction) & (reference != exclude_label)
        mask &= np.isin(reference, mineral_labels)
        y_true = reference[mask].astype(np.int64)
        y_pred = prediction[mask].astype(np.int64)
        labels = list(mineral_labels)
        results[name] = {
            "n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
            "report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        }
    return results


def plot_prediction_comparison(prediction_map, pixel_reference_map, full_reference_map,
                               *, mineral_labels=(101, 102, 103, 104, 105, 106), # Hard coded, adjust accordingly
                               legend_labels=None):
    if legend_labels is None:
        legend_labels = tuple(mineral_labels)
        if 107 not in legend_labels:
            legend_labels += (107,)
    else:
        legend_labels = tuple(legend_labels)
    cmap = plt.get_cmap("viridis", len(legend_labels)).copy()
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(legend_labels[0]-0.5, legend_labels[-1]+1.5, 1), len(legend_labels))
    prediction = np.asarray(prediction_map, dtype=float).copy()
    merged = np.asarray(pixel_reference_map, dtype=float).copy()
    full = np.asarray(full_reference_map, dtype=float).copy()
    prediction[prediction == 107] = np.nan
    merged[merged == 107] = np.nan
    full[full == 107] = np.nan
    panels = [
        (merged, "Pixel-level VAE-derived mineral labels"),
        (np.where(np.isfinite(merged), prediction, np.nan), "Prediction at labelled sample pixels"),
        (full, "Full reference labels (class 107 excluded)"),
        (np.where(np.isfinite(full), prediction, np.nan), "Prediction aligned to full reference"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for ax, (values, title) in zip(axes.flat, panels):
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, ticks=list(legend_labels), label="Mineral class", shrink=0.75)
    plt.show()


def metric_table(evaluation):
    rows = []
    for comparison, result in evaluation.items():
        rows.append({"comparison": comparison, "class": "overall", "n": result["n"],
                     "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"]})
        for cls, values in result["report"].items():
            if str(cls).isdigit():
                rows.append({"comparison": comparison, "class": int(cls), "n": int(values["support"]),
                             "precision": values["precision"], "recall": values["recall"],
                             "f1_score": values["f1-score"]})
    return pd.DataFrame(rows)


def detect_core_sticks(cube, *, core_width_px=None, core_centers_px=None):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    cube = np.asarray(cube)
    h, w, _ = cube.shape
    mask = np.zeros((h, w), dtype=bool)
    for r0 in range(0, h, 32):
        block = cube[r0:min(r0+32, h)]
        mask[r0:min(r0+32, h)] = np.any(np.isfinite(block) & (block != 0), axis=2)
    if not mask.any():
        raise RuntimeError("No finite, nonzero core pixels were found.")
    if core_width_px is None:
        widths = []
        for row in np.unique(np.linspace(0, h-1, min(64, h)).astype(int)):
            edges = np.diff(np.r_[False, mask[row], False].astype(np.int8))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            widths.extend((ends-starts).tolist())
        widths = np.asarray(widths)
        widths = widths[widths >= 2]
        if not len(widths):
            raise RuntimeError("Could not estimate stick width; set CORE_WIDTH_PX manually.")
        width = float(np.percentile(widths, 65))
    else:
        width = float(core_width_px)
    profile = gaussian_filter1d(mask.sum(axis=0).astype(float), sigma=max(1.0, 0.10*width))
    if core_centers_px is None:
        peaks, _ = find_peaks(np.r_[0.0, profile, 0.0],
                              distance=max(2, int(round(0.90*width))),
                              prominence=0.10*profile.max())
        centers = peaks - 1
    else:
        centers = np.asarray(core_centers_px, dtype=int)
    if len(centers) == 0:
        raise RuntimeError("No stick centres detected. Set CORE_CENTERS_PX manually.")
    stick_map = np.zeros((h, w), dtype=np.int32)
    half = max(1, int(round(width / 2)))
    for stick_id, center in enumerate(centers, start=1):
        c0, c1 = max(0, int(center)-half), min(w, int(center)+half+1)
        stick_map[:, c0:c1] = np.where(mask[:, c0:c1], stick_id, 0)
    return stick_map, np.asarray(centers), width


def summarize_core_sticks(stick_map, reference_map, prediction_map,
                          *, mineral_labels=(101, 102, 103, 104, 105, 106), exclude_label=107): # Hard coded, adjust accordingly
    stick_map = np.asarray(stick_map)
    reference = np.asarray(reference_map, dtype=float)
    prediction = np.asarray(prediction_map, dtype=float)
    rows = []
    for stick_id in sorted(int(x) for x in np.unique(stick_map) if x > 0):
        mask = (stick_map == stick_id) & np.isfinite(reference) & np.isfinite(prediction)
        mask &= np.isin(reference, mineral_labels) & (reference != exclude_label)
        n = int(mask.sum())
        mismatches = int(np.count_nonzero(reference[mask] != prediction[mask].astype(int)))
        rows.append({"stick_id": stick_id, "compared_pixels": n,
                     "mismatched_pixels": mismatches,
                     "disagreement_fraction": mismatches/n if n else np.nan,
                     "accuracy": 1-mismatches/n if n else np.nan})
    return pd.DataFrame(rows).sort_values("disagreement_fraction", ascending=False, na_position="last")


def plot_core_stick(stick_id, stick_map, reference_map, prediction_map,
                    *, mineral_labels=(101, 102, 103, 104, 105, 106), # Hard coded, adjust accordingly
                    legend_labels=None):
    stick_map = np.asarray(stick_map)
    reference = np.asarray(reference_map, dtype=float)
    prediction = np.asarray(prediction_map, dtype=float)
    selected = stick_map == int(stick_id)
    if not selected.any():
        raise ValueError(f"Stick {stick_id} is not present in stick_map.")
    rr, cc = np.nonzero(selected)
    r0, r1, c0, c1 = rr.min(), rr.max()+1, cc.min(), cc.max()+1
    selected = selected[r0:r1, c0:c1]
    ref, pred = reference[r0:r1, c0:c1], prediction[r0:r1, c0:c1]
    valid = selected & np.isin(ref, mineral_labels) & np.isfinite(pred)
    n = int(valid.sum())
    mismatches = int(np.count_nonzero(ref[valid] != pred[valid].astype(int)))
    if legend_labels is None:
        legend_labels = tuple(mineral_labels)
        if 107 not in legend_labels:
            legend_labels += (107,)
    else:
        legend_labels = tuple(legend_labels)
    cmap = plt.get_cmap("viridis", len(legend_labels)).copy()
    cmap.set_bad("white")
    norm = BoundaryNorm(np.arange(legend_labels[0]-0.5, legend_labels[-1]+1.5, 1), len(legend_labels))
    fig, axes = plt.subplots(1, 2, figsize=(7, 10), sharex=True, sharey=True, constrained_layout=True)
    for ax, arr, title in zip(axes, (ref, pred), ("Reference labels", "Selected-model prediction")):
        show = np.where(valid, arr, np.nan)
        im = ax.imshow(np.ma.masked_invalid(show), cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal",
                       extent=(c0-0.5, c1-0.5, r1-0.5, r0-0.5))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xlabel(f"Core stick {stick_id}")
    axes[0].set_ylabel("Image row")
    fig.colorbar(im, ax=axes, ticks=list(legend_labels), label="Mineral class", shrink=0.7)
    fig.suptitle(f"Stick {stick_id}: disagreement {mismatches/n:.2%} ({mismatches:,}/{n:,})" if n else
                 f"Stick {stick_id}: no comparable labelled pixels")
    plt.show()
