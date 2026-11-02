"""Data loading, reference-label alignment, and spectral normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_envi_cube(header_path: str | Path, image_path: str | Path):
    try:
        import spectral.io.envi as envi
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError("Install the 'spectral' package to read ENVI files.") from exc

    header_path, image_path = Path(header_path), Path(image_path)
    if not header_path.is_file():
        raise FileNotFoundError(f"ENVI header not found: {header_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"ENVI image not found: {image_path}")
    image = envi.open(str(header_path), str(image_path))
    cube = np.asarray(image.load())
    if cube.ndim != 3:
        raise ValueError(f"Expected a rows × columns × bands cube; got {cube.shape}.")
    return cube, np.asarray(image.bands.centers, dtype=np.float32), dict(image.metadata)


def load_label_map(
    label_csv: str | Path,
    image_shape: tuple[int, int],
    index_column: str = "Unnamed: 0",   # Hard coded, adjust accordingly
    label_column: str = "0",            # Hard coded, adjust accordingly
) -> np.ndarray:
    """Align flattened pixel labels from CSV into a 2-D image map."""
    frame = pd.read_csv(label_csv)
    if index_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(
            f"Expected columns {index_column!r} and {label_column!r} in {label_csv}; "
            f"found {frame.columns.tolist()}. Set the CSV column controls in the notebook."
        )
    n_pixels = int(image_shape[0] * image_shape[1])
    flat = np.full(n_pixels, np.nan, dtype=np.float32)
    indices = pd.to_numeric(frame[index_column], errors="coerce").to_numpy()
    labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy()
    valid = np.isfinite(indices) & np.isfinite(labels)
    indices = indices[valid].astype(np.int64)
    labels = labels[valid].astype(np.float32)
    in_bounds = (indices >= 0) & (indices < n_pixels)
    flat[indices[in_bounds]] = labels[in_bounds]
    return flat.reshape(image_shape)


class BandMinMaxNormalizer:
    def __init__(self, band_min: np.ndarray, band_max: np.ndarray):
        self.band_min = np.asarray(band_min, dtype=np.float32)
        self.band_max = np.asarray(band_max, dtype=np.float32)
        denom = self.band_max - self.band_min
        denom[denom < 1e-12] = 1.0
        self.band_denom = denom

    @classmethod
    def fit(cls, cube: np.ndarray, n_bands: int, label_map=None, exclude_label=107): # Hard coded, adjust accordingly
        cube = np.asarray(cube)
        if cube.ndim != 3 or cube.shape[2] < n_bands:
            raise ValueError(f"Cube must have at least {n_bands} bands; got {cube.shape}.")
        excluded = None
        if label_map is not None:
            if np.shape(label_map) != cube.shape[:2]:
                raise ValueError("Label map shape does not match the hyperspectral image.")
            excluded = np.asarray(label_map) != exclude_label
        mins = np.zeros(n_bands, dtype=np.float32)
        maxs = np.ones(n_bands, dtype=np.float32)
        for band_idx in range(n_bands):
            band = np.asarray(cube[..., band_idx], dtype=np.float32)
            mask = np.isfinite(band) & (band != 0)
            if excluded is not None:
                mask &= excluded
            values = band[mask]
            if values.size:
                mins[band_idx] = np.min(values)
                maxs[band_idx] = np.max(values)
        return cls(mins, maxs)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        out = np.clip((values - self.band_min) / self.band_denom, 0.0, 1.0)
        out[~np.isfinite(out)] = 0.0
        return out.astype(np.float32, copy=False)

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "norm_band_min.npy", self.band_min)
        np.save(output_dir / "norm_band_max.npy", self.band_max)


def build_false_rgb(cube: np.ndarray, wavelengths: np.ndarray, bands_nm=(2200, 1650, 1100)): # Hard coded, adjust accordingly
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.size != cube.shape[2]:
        raise ValueError("Wavelength vector length does not match cube band count.")
    band_indices = [int(np.argmin(np.abs(wavelengths - nm))) for nm in bands_nm]

    def stretch(layer):
        layer = np.asarray(layer, dtype=np.float32)
        vals = layer[np.isfinite(layer) & (layer > 0)]
        if not vals.size:
            return np.zeros_like(layer)
        lo, hi = np.percentile(vals, [2, 98])
        return np.clip((layer - lo) / max(float(hi - lo), 1e-9), 0, 1)

    rgb = np.stack([stretch(cube[..., i]) for i in band_indices], axis=-1)
    nodata = ~np.any(np.isfinite(cube) & (cube != 0), axis=2)
    rgb[nodata] = 1.0
    actual_nm = wavelengths[band_indices]
    return rgb.astype(np.float32), band_indices, actual_nm
