"""Patch sampling and spatial distribution visualisation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_tools import build_false_rgb


def select_patches(
    cube: np.ndarray,
    normalizer,
    *,
    strategy: str = "random_spatial",
    patch_size: int = 15,
    n_patches_target: int = 500,
    max_candidates: int = 300_000,
    min_valid_fraction: float = 0.85,
    pca_components: int = 20,
    candidate_seed: int = 0,
    sampling_seed: int = 0,
    label_map=None,
    exclude_label: int = 107,
    feature_bands: int | None = None,
    output_dir: str | Path | None = None,
    pca_batch_size: int = 8192,
):

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D cube; got {cube.shape}.")
    if strategy not in {"random_spatial", "pca_guided"}:
        raise ValueError("strategy must be 'random_spatial' or 'pca_guided'.")
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer.")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")
    if int(n_patches_target) < 1:
        raise ValueError("n_patches_target must be a positive integer.")
    n_bands = int(feature_bands or cube.shape[2])
    if cube.shape[2] < n_bands:
        raise ValueError(f"Cube has {cube.shape[2]} bands; {n_bands} requested.")
    h, w, _ = cube.shape
    radius = patch_size // 2
    valid = np.isfinite(cube[..., :n_bands]).all(axis=2) & (cube[..., :n_bands] != 0).any(axis=2)
    if label_map is not None:
        valid &= np.asarray(label_map) != exclude_label
    valid[:radius, :] = False
    valid[-radius:, :] = False
    valid[:, :radius] = False
    valid[:, -radius:] = False
    candidates = np.argwhere(valid)
    if not candidates.size:
        raise RuntimeError(
            "No valid candidate centers remain after masking and border exclusion."
        )

    rng = np.random.default_rng(candidate_seed)
    num_candidates = min(int(max_candidates), len(candidates))
    candidates = candidates[rng.choice(len(candidates), size=num_candidates, replace=False)]
    start = time.perf_counter()

    def extract(index):
        r, c = map(int, candidates[index])
        r0, r1 = r - radius, r + radius + 1
        c0, c1 = c - radius, c + radius + 1
        patch_valid = valid[r0:r1, c0:c1]
        frac = float(patch_valid.mean())
        if frac < min_valid_fraction:
            return None
        raw = cube[r0:r1, c0:c1, :n_bands].astype(np.float32, copy=False).copy()
        return normalizer.transform(raw), raw, (r, c), frac

    records = []
    pca_meta = {}
    if strategy == "pca_guided":
        spectra = cube[candidates[:, 0], candidates[:, 1], :n_bands].astype(np.float32, copy=False)
        scaled = StandardScaler().fit_transform(normalizer.transform(spectra))
        n_components = min(int(pca_components), scaled.shape[1], max(1, scaled.shape[0] - 1))
        embedding = PCA(n_components=n_components, svd_solver="randomized", random_state=candidate_seed).fit_transform(scaled)
        k = min(int(n_patches_target), len(candidates))
        km = MiniBatchKMeans(n_clusters=k, batch_size=pca_batch_size, random_state=candidate_seed)
        cluster_ids = km.fit_predict(embedding)
        d2 = np.sum((embedding - km.cluster_centers_[cluster_ids]) ** 2, axis=1)
        order = np.lexsort((d2, cluster_ids))
        ordered_labels = cluster_ids[order]
        first_per_cluster = np.r_[True, ordered_labels[1:] != ordered_labels[:-1]]
        representative_indices = order[first_per_cluster].astype(int).tolist()
        for idx in representative_indices:
            rec = extract(int(idx))
            if rec is not None:
                records.append(rec)
        pca_meta = {"target_patch_count": int(k), "pca_components": int(n_components)}
    else:
        target_count = int(n_patches_target)
        order = np.random.default_rng(sampling_seed).permutation(len(candidates))
        occupied = np.zeros((h, w), dtype=bool)
        for idx in order:
            r, c = map(int, candidates[idx])
            r0, r1 = r - radius, r + radius + 1
            c0, c1 = c - radius, c + radius + 1
            if occupied[r0:r1, c0:c1].any():
                continue
            rec = extract(int(idx))
            if rec is None:
                continue
            records.append(rec)
            occupied[r0:r1, c0:c1] = True
            if len(records) >= target_count:
                break
        if len(records) < target_count:
            raise RuntimeError(
                f"Selected {len(records):,} valid non-overlapping patches; {target_count:,} required. "
                "Reduce N_PATCHES_TARGET or MIN_VALID_FRAC, or use a smaller PATCH_SIZE."
            )

    if not records:
        raise RuntimeError("Patch selection produced no patches. Check the mask and selection settings.")
    if strategy == "random_spatial" and len(records) != int(n_patches_target):
        raise RuntimeError(
            f"{strategy} selection returned {len(records)} patches, but the target is "
            f"{int(n_patches_target)}. Check valid-window filtering and candidate settings."
        )
    patches, patches_raw, centers, valid_fracs = map(list, zip(*records))
    patches = np.stack(patches).astype(np.float32, copy=False)
    patches_raw = np.stack(patches_raw).astype(np.float32, copy=False)
    centers = np.asarray(centers, dtype=np.int32)
    valid_fracs = np.asarray(valid_fracs, dtype=np.float32)
    elapsed = float(time.perf_counter() - start)
    meta = {
        "strategy": strategy,
        "candidate_pixels": int(len(candidates)),
        "patch_count": int(len(patches)),
        "target_patch_count": int(n_patches_target),
        "patch_size": int(patch_size),
        "min_valid_fraction": float(min_valid_fraction),
        "candidate_seed": int(candidate_seed),
        "sampling_seed": int(sampling_seed),
        "nominal_window_area_fraction": float(len(patches) * patch_size**2 / (h * w)),
        "selection_seconds": elapsed,
        **pca_meta,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "patches.npy", patches)
        np.save(out / "patches_raw.npy", patches_raw)
        np.save(out / "patch_centers_rc.npy", centers)
        np.save(out / "patch_valid_frac.npy", valid_fracs)
        (out / "patch_sampling_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return patches, patches_raw, centers, valid_fracs, valid, meta


def plot_patch_distribution(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    patch_centers: np.ndarray,
    patch_size: int,
    *,
    selected_index: int = 0,
    height: float = 8,
    box_alpha: float = 0.22,
):
    centers = np.asarray(patch_centers)
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("patch_centers must be a nonempty (n_patches, 2) array.")
    if not 0 <= selected_index < len(centers):
        raise IndexError(f"selected_index must be between 0 and {len(centers)-1}.")
    rgb, _, actual_nm = build_false_rgb(cube, wavelengths)
    h, w = cube.shape[:2]
    fig, ax = plt.subplots(figsize=(max(8, w / h * height), height), dpi=160)
    ax.imshow(rgb, origin="upper", interpolation="bilinear")
    rad = patch_size // 2
    for idx, (row, col) in enumerate(centers):
        chosen = idx == selected_index
        ax.add_patch(Rectangle(
            (int(col) - rad, int(row) - rad), patch_size, patch_size,
            linewidth=1.6 if chosen else 0.45,
            edgecolor="red" if chosen else "cyan",
            facecolor="red" if chosen else "cyan",
            alpha=0.5 if chosen else box_alpha,
        ))
    ax.set_title(
        f"Patch distribution on false-colour SWIR image | "
        f"R≈{actual_nm[0]:.0f} nm, G≈{actual_nm[1]:.0f} nm, B≈{actual_nm[2]:.0f} nm"
    )
    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, facecolor="cyan", edgecolor="cyan", alpha=0.7,
                  label=f"Patch footprints (n={len(centers)})"),
        Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="red", alpha=0.7,
                  label=f"Selected patch #{selected_index}"),
    ], loc="lower right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    plt.show()
