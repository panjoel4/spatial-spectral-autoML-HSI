"""VAE training, pixel-level clustering, spectra plots, and cluster-to-mineral mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def _build_vae(input_dim, hidden_dims=(512, 256, 128), latent_dim=10, beta_kl=0.5): # Hard coded, adjust accordingly
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install TensorFlow to train the VAE.") from exc

    class Sampling(layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            epsilon = tf.random.normal(shape=tf.shape(mean))
            return mean + tf.exp(0.5 * log_var) * epsilon

    class VAELoss(layers.Layer):
        def __init__(self, beta, n_inputs, **kwargs):
            super().__init__(**kwargs)
            self.beta = float(beta)
            self.n_inputs = int(n_inputs)

        def call(self, inputs):
            original, reconstructed, mean, log_var = inputs
            reconstruction = tf.reduce_mean(tf.square(original - reconstructed)) * self.n_inputs
            kl = -0.5 * tf.reduce_sum(
                1.0 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1
            )
            self.add_loss(reconstruction + self.beta * tf.reduce_mean(kl))
            return reconstructed

    inputs = keras.Input(shape=(input_dim,), name="input_spectrum")
    hidden = inputs
    for i, units in enumerate(hidden_dims):
        hidden = layers.Dense(units, activation="relu", name=f"encoder_{i}")(hidden)
    z_mean = layers.Dense(latent_dim, name="z_mean")(hidden)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(hidden)
    z = Sampling(name="z_sampling")([z_mean, z_log_var])
    hidden = z
    for i, units in enumerate(reversed(hidden_dims)):
        hidden = layers.Dense(units, activation="relu", name=f"decoder_{i}")(hidden)
    reconstructed = layers.Dense(input_dim, name="reconstruction_linear")(hidden)
    output = VAELoss(beta_kl, input_dim, name="vae_objective")(
        [inputs, reconstructed, z_mean, z_log_var]
    )
    vae = keras.Model(inputs, output, name="spectral_vae")
    encoder_mean = keras.Model(inputs, z_mean, name="spectral_vae_encoder_mean")
    return vae, encoder_mean


def train_vae(
    patches: np.ndarray, *, label_map=None, patch_centers=None, exclude_label=107, # Hard coded, adjust accordingly
    latent_dim=10, hidden_dims=(512, 256, 128), beta_kl=0.5, epochs=50, # Hard coded, adjust accordingly
    batch_size=1024, max_training_spectra=200_000, random_seed=0,
    patience=5, verbose=1,
):

    import tensorflow as tf

    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 4:
        raise ValueError(f"Expected patches shaped (n, rows, cols, bands); got {patches.shape}.")
    if patches.shape[1] != patches.shape[2]:
        raise ValueError("Patch windows must be square.")
    tf.keras.backend.clear_session()
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    valid = np.isfinite(patches).all(axis=-1) & (patches != 0).any(axis=-1)
    if label_map is not None and patch_centers is not None:
        radius = patches.shape[1] // 2
        labels = np.asarray(label_map)
        for i, (row, col) in enumerate(np.asarray(patch_centers)):
            row, col = int(row), int(col)
            local = labels[row-radius:row+radius+1, col-radius:col+radius+1]
            if local.shape == valid[i].shape:
                valid[i] &= local != exclude_label
    spectra = patches.reshape(-1, patches.shape[-1])[valid.reshape(-1)]
    if not len(spectra):
        raise RuntimeError("No valid, nonzero spectra remain for VAE training.")
    rng = np.random.default_rng(random_seed)
    if len(spectra) > max_training_spectra:
        spectra = spectra[rng.choice(len(spectra), int(max_training_spectra), replace=False)]
    vae, encoder_mean = _build_vae(
        patches.shape[-1], hidden_dims=hidden_dims, latent_dim=latent_dim, beta_kl=beta_kl
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    history = vae.fit(
        spectra, epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True
        )],
    )
    return vae, encoder_mean, history, int(len(spectra))


def collect_patch_pixels(patches, patch_centers, *, patches_raw=None, label_map=None,
                         exclude_label=107, feature_bands=None):
    patches = np.asarray(patches, dtype=np.float32)
    centers = np.asarray(patch_centers, dtype=np.int32)
    if patches.ndim != 4 or centers.shape != (len(patches), 2):
        raise ValueError("patches and patch_centers are not aligned.")
    if patches.shape[1] != patches.shape[2]:
        raise ValueError("Patch windows must be square.")
    n_bands = int(feature_bands or patches.shape[-1])
    raw = None if patches_raw is None else np.asarray(patches_raw, dtype=np.float32)
    if raw is not None and raw.shape != patches.shape:
        raise ValueError("Raw and normalized patch arrays must have matching shapes.")
    radius = patches.shape[1] // 2
    specs, raw_specs, coords = [], [], []
    for i, (row, col) in enumerate(centers):
        row, col = int(row), int(col)
        patch = patches[i, ..., :n_bands]
        valid = np.isfinite(patch).all(axis=-1) & (patch != 0).any(axis=-1)
        if label_map is not None:
            local_labels = np.asarray(label_map)[row-radius:row+radius+1, col-radius:col+radius+1]
            if local_labels.shape == valid.shape:
                valid &= local_labels != exclude_label
        if not valid.any():
            continue
        rr, cc = np.meshgrid(
            np.arange(row-radius, row+radius+1, dtype=np.int32),
            np.arange(col-radius, col+radius+1, dtype=np.int32), indexing="ij"
        )
        specs.append(patch[valid].astype(np.float32, copy=False))
        coords.append(np.column_stack((rr[valid], cc[valid])).astype(np.int32))
        if raw is not None:
            raw_specs.append(raw[i, ..., :n_bands][valid].astype(np.float32, copy=False))
    if not specs:
        raise RuntimeError("No valid pixels were found in the selected patches.")
    pixel_specs = np.concatenate(specs, axis=0)
    pixel_coords = np.concatenate(coords, axis=0)
    pixel_specs_raw = np.concatenate(raw_specs, axis=0) if raw is not None else pixel_specs.copy()
    return pixel_specs, pixel_specs_raw, pixel_coords


def cluster_pixels(pixel_specs, encoder_mean, *, n_clusters=14, random_seed=0,
                   batch_size=4096, encode_batch_size=8192):
    pixel_specs = np.asarray(pixel_specs, dtype=np.float32)
    if pixel_specs.ndim != 2 or len(pixel_specs) < 2:
        raise ValueError("At least two pixel spectra are required.")
    latent = encoder_mean.predict(pixel_specs, batch_size=encode_batch_size, verbose=0)
    k = min(int(n_clusters), len(latent))
    model = MiniBatchKMeans(
        n_clusters=k, batch_size=batch_size, random_state=random_seed, n_init=10
    )
    clusters = model.fit_predict(latent)
    return latent, clusters, model


def compare_pca_vae_clusters(pixel_specs, latent, vae_labels, *, n_clusters=14,
                             pca_components=10, random_seed=0, metric_subsample=20_000):
    pixel_specs = np.asarray(pixel_specs, dtype=np.float32)
    latent = np.asarray(latent, dtype=np.float32)
    vae_labels = np.asarray(vae_labels)
    n_pc = min(int(pca_components), pixel_specs.shape[1], max(1, len(pixel_specs)-1))
    scaled = StandardScaler().fit_transform(pixel_specs)
    pca_space = PCA(n_components=n_pc, svd_solver="randomized", random_state=random_seed).fit_transform(scaled)
    pca_labels = MiniBatchKMeans(
        n_clusters=min(int(n_clusters), len(pca_space)), batch_size=4096,
        random_state=random_seed, n_init=10,
    ).fit_predict(pca_space)
    rng = np.random.default_rng(random_seed)

    def score(name, space, labels):
        labels = np.asarray(labels)
        unique = np.unique(labels)
        result = {"method": name, "n_pixels": int(len(labels)),
                  "n_clusters": int(len(unique)), "silhouette": np.nan,
                  "davies_bouldin": np.nan}
        if len(unique) >= 2 and len(unique) < len(labels):
            if len(labels) > metric_subsample:
                idx = rng.choice(len(labels), int(metric_subsample), replace=False)
                x_score, y_score = space[idx], labels[idx]
            else:
                x_score, y_score = space, labels
            if np.unique(y_score).size >= 2 and len(y_score) > np.unique(y_score).size:
                result["silhouette"] = float(silhouette_score(x_score, y_score))
            result["davies_bouldin"] = float(davies_bouldin_score(space, labels))
        return result

    table = pd.DataFrame([
        score("VAE z_mean + k-means (pixel-level)", latent, vae_labels),
        score(f"PCA ({n_pc}) + k-means (pixel-level)", pca_space, pca_labels),
    ])
    return table, pca_space, pca_labels


def plot_random_raw_spectra(pixel_specs_raw, pixel_clusters, wavelengths, *,
                            n_per_cluster=5, random_seed=0, ncols=2):
    spectra = np.asarray(pixel_specs_raw, dtype=np.float32)
    clusters = np.asarray(pixel_clusters)
    x = np.asarray(wavelengths, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] != len(clusters):
        raise ValueError("Raw spectra and pixel cluster labels are not aligned.")
    if len(x) != spectra.shape[1]:
        x = np.arange(spectra.shape[1])
        x_label = "Band index"
    else:
        x_label = "Wavelength (nm)"
    cluster_ids = np.unique(clusters)
    ncols = min(max(1, int(ncols)), len(cluster_ids))
    nrows = int(np.ceil(len(cluster_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.2*nrows), squeeze=False)
    rng = np.random.default_rng(random_seed)
    for ax, cluster_id in zip(axes.flat, cluster_ids):
        candidates = np.flatnonzero(clusters == cluster_id)
        chosen = rng.choice(candidates, min(int(n_per_cluster), len(candidates)), replace=False)
        for idx in chosen:
            ax.plot(x, spectra[idx], linewidth=0.9, alpha=0.8)
        ax.set_title(f"VAE pixel cluster {cluster_id} (n={len(candidates):,})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Raw reflectance")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(cluster_ids):]:
        ax.set_visible(False)
    fig.suptitle("Random raw pixel spectra by VAE cluster", y=1.01)
    fig.tight_layout()
    plt.show()


def summarize_and_merge_clusters(label_map, pixel_coords, pixel_clusters,
                                 valid_mineral_labels=(101, 102, 103, 104, 105, 106), # Hard coded, adjust accordingly
                                 excluded_labels=(0, 107)):
    labels = np.asarray(label_map, dtype=float)
    coords = np.asarray(pixel_coords, dtype=np.int32)
    clusters = np.asarray(pixel_clusters, dtype=np.int64)
    if labels.ndim != 2 or coords.shape != (len(clusters), 2):
        raise ValueError("label_map, pixel_coords, and pixel_clusters are not aligned.")
    in_bounds = ((coords[:, 0] >= 0) & (coords[:, 0] < labels.shape[0]) &
                 (coords[:, 1] >= 0) & (coords[:, 1] < labels.shape[1]))
    reference_at_pixels = np.full(len(coords), np.nan, dtype=float)
    reference_at_pixels[in_bounds] = labels[coords[in_bounds, 0], coords[in_bounds, 1]]
    excluded = set(int(x) for x in excluded_labels)
    rows = []
    for cluster_id in sorted(np.unique(clusters)):
        values = reference_at_pixels[clusters == cluster_id]
        values = values[np.isfinite(values)].astype(np.int64)
        values = values[~np.isin(values, list(excluded))]
        counts = pd.Series(values).value_counts().sort_values(ascending=False)
        n = int(counts.sum())
        rows.append({
            "cluster_id": int(cluster_id), "n_reference_pixels": n,
            "n_unique_labels": int(counts.size),
            "top1_label": int(counts.index[0]) if n else np.nan,
            "top1_count": int(counts.iloc[0]) if n else 0,
            "top1_pct": float(100 * counts.iloc[0] / n) if n else np.nan,
            "top2_label": int(counts.index[1]) if counts.size > 1 else np.nan,
            "top2_pct": float(100 * counts.iloc[1] / n) if counts.size > 1 else np.nan,
        })
    summary = pd.DataFrame(rows)
    merge_map = {int(label): [] for label in valid_mineral_labels}
    for _, row in summary.iterrows():
        if pd.notna(row["top1_label"]) and int(row["top1_label"]) in merge_map:
            merge_map[int(row["top1_label"])].append(int(row["cluster_id"]))
    lookup = {cid: label for label, cids in merge_map.items() for cid in cids}
    merged_labels = np.asarray([lookup.get(int(cid), -1) for cid in clusters], dtype=np.int64)
    pixel_merged_map = np.full(labels.shape, np.nan, dtype=np.float32)
    valid_merged = merged_labels >= min(valid_mineral_labels)
    valid_coords = coords[valid_merged]
    pixel_merged_map[valid_coords[:, 0], valid_coords[:, 1]] = merged_labels[valid_merged]
    return summary, merge_map, merged_labels, pixel_merged_map, reference_at_pixels
