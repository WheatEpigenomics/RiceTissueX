#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Stable + AdamW + LayerNorm + Dropout + PDF Logging

import os, argparse, json, datetime
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

# ============================================================
# 0. Runtime setup
# ============================================================
def setup_runtime(seed=42, use_xla=False, use_mixed=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if use_mixed:
        mixed_precision.set_global_policy("mixed_float16")
    if use_xla:
        tf.config.optimizer.set_jit(True)
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)

# ============================================================
# 1. Data loading & normalization
# ============================================================
META_COLS = ["peak_id","chr","start","end"]

def load_atac_matrix(path):
    df = pd.read_csv(path, sep="\t")
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")
    tissue_cols = [c for c in df.columns if c not in META_COLS]
    X = df[tissue_cols].to_numpy(np.float32)
    return df, tissue_cols, X

def standardize(X, log1p=True, eps=1e-6):
    X = np.clip(X, 0, np.percentile(X, 99.9))
    if log1p:
        X = np.log1p(X, dtype=np.float32)
    mean = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + eps
    Xz = np.nan_to_num((X - mean) / std)
    print(f"[Data] Scaled range: min={Xz.min():.3f}, max={Xz.max():.3f}")
    return Xz.astype(np.float32), {"mean": mean, "std": std}

def make_dataset(X, batch=2048, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, X))
    if shuffle:
        ds = ds.shuffle(min(len(X), 500_000))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 2. Model definition
# ============================================================
class ChannelMask(L.Layer):
    # mask  some
    def __init__(self, mask_rate=0.25, **kw):
        super().__init__(**kw)
        self.mask_rate = mask_rate

    def call(self, x, training=None):
        c = x.shape[-1] or tf.shape(x)[1]
        b = tf.shape(x)[0]
        if training:
            m = tf.cast(tf.random.uniform((b, c)) > self.mask_rate, x.dtype)
            return x * m, m
        return x, tf.ones_like(x)

def build_mae(n_input=22, latent_dim=128, width=256, depth=3, mask_rate=0.25, dropout=0.05):
    inp = L.Input(shape=(n_input,), dtype="float32")
    x, mask = ChannelMask(mask_rate)(inp)
    for i in range(depth):
        x = L.Dense(width, activation="gelu")(x)
        x = L.LayerNormalization()(x)
        if dropout > 0:
            x = L.Dropout(dropout)(x)
    z = L.Dense(latent_dim, name="z_peak")(x)
    y = z
    for i in range(depth):
        y = L.Dense(width, activation="gelu")(y)
    recon = L.Dense(n_input, dtype="float32", name="recon")(y)
    recon = tf.ensure_shape(recon, (None, n_input))
    mask = tf.ensure_shape(mask, (None, n_input))
    return tf.keras.Model(inp, [recon, mask], name="ATAC_MAE")

# ============================================================
# 3. Masked loss + Pearson R
# ============================================================
def masked_mse_pearson(y_true, y_pred_mask, min_masked=4):
    y_pred, mask = y_pred_mask
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask_inv = 1.0 - mask

    denom = tf.reduce_sum(mask_inv, axis=1)
    valid = denom >= min_masked

    def valid_branch():
        y_true_v = tf.boolean_mask(y_true, valid)
        y_pred_v = tf.boolean_mask(y_pred, valid)
        mask_inv_v = tf.boolean_mask(mask_inv, valid)
        denom_v = tf.boolean_mask(denom, valid) + 1e-6

        diff = (y_true_v - y_pred_v) * mask_inv_v
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1) / denom_v)

        yt = y_true_v * mask_inv_v
        yp = y_pred_v * mask_inv_v
        mean_t = tf.reduce_sum(yt, axis=1, keepdims=True) / denom_v[:, None]
        mean_p = tf.reduce_sum(yp, axis=1, keepdims=True) / denom_v[:, None]
        yt -= mean_t
        yp -= mean_p
        cov = tf.reduce_sum(yt * yp, axis=1)
        var_t = tf.reduce_sum(tf.square(yt), axis=1)
        var_p = tf.reduce_sum(tf.square(yp), axis=1)
        r = cov / tf.sqrt(var_t * var_p + 1e-6)
        r = tf.clip_by_value(r, -1.0, 1.0)
        r = tf.boolean_mask(r, tf.math.is_finite(r))
        return tf.reduce_mean(mse), tf.reduce_mean(r)

    def invalid_branch():
        return tf.constant(0.0), tf.constant(0.0)

    return tf.cond(tf.reduce_any(valid), valid_branch, invalid_branch)

# ============================================================
# 4. Model subclass
# ============================================================
class MAEModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, mask = self(x, training=True)
            loss, r = masked_mse_pearson(y, (y_pred, mask))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss, "masked_PearsonR": r}

    def test_step(self, data):
        x, y = data
        y_pred, mask = self(x, training=True)
        loss, r = masked_mse_pearson(y, (y_pred, mask))
        return {"loss": loss, "masked_PearsonR": r}

# ============================================================
# 5. Train + Logging
# ============================================================
def plot_curves(history, out_pdf):
    plt.figure(figsize=(7,5))
    if "loss" in history: plt.plot(history["loss"], label="Train Loss")
    if "val_loss" in history: plt.plot(history["val_loss"], label="Val Loss")
    if "masked_PearsonR" in history: plt.plot(history["masked_PearsonR"], label="Train R")
    if "val_masked_PearsonR" in history: plt.plot(history["val_masked_PearsonR"], label="Val R")
    plt.xlabel("Epoch"); plt.ylabel("Metric Value")
    plt.title("ATAC MAE (Masked MSE + PearsonR)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"📊 Curve saved: {out_pdf}")

def train(args):
    setup_runtime(args.seed, args.xla, not args.no_mixed)
    df, tissue_cols, X = load_atac_matrix(args.data)
    Xz, stats = standardize(X, log1p=not args.no_log1p)
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "scaler_stats.npz"), **stats)

    # Split train/val/test
    n = len(Xz)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_tr, n_val = int(0.8 * n), int(0.9 * n)
    X_tr, X_val, X_te = Xz[idx[:n_tr]], Xz[idx[n_tr:n_val]], Xz[idx[n_val:]]

    ds_tr = make_dataset(X_tr, args.batch_size, True)
    ds_val = make_dataset(X_val, args.batch_size, False)
    ds_te = make_dataset(X_te, args.batch_size, False)

    base = build_mae(len(tissue_cols), args.latent_dim, args.width, args.depth,
                     args.mask_rate, args.dropout)
    model = MAEModel(inputs=base.input, outputs=base.output)

    base_opt = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr, weight_decay=1e-4
    )
    opt = base_opt if args.no_mixed else mixed_precision.LossScaleOptimizer(base_opt)
    model.compile(optimizer=opt)

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
    ]

    print(" Start training ...")
    hist = model.fit(ds_tr, validation_data=ds_val, epochs=args.epochs, callbacks=cbs, verbose=1)

    # Save model
    model.save(os.path.join(args.outdir, "atac_mae_model_final.keras"))
    print(" Model saved")

    # Save embeddings
    enc = tf.keras.Model(base.input, base.get_layer("z_peak").output)
    z = enc.predict(Xz, batch_size=args.batch_size, verbose=1)
    z_df = pd.DataFrame(z, columns=[f"z{i+1}" for i in range(z.shape[1])])
    z_df.insert(0, "peak_id", df["peak_id"])
    z_df.to_csv(os.path.join(args.outdir, "z_peak_embeddings.tsv"), sep="\t", index=False)

    # Logs
    history = hist.history
    log_df = pd.DataFrame(history)
    csv_path = os.path.join(args.outdir, "training_log.csv")
    log_df.to_csv(csv_path, index=False)
    print(f" Log saved: {csv_path}")

    # Plot curves
    pdf_path = os.path.join(args.outdir, "loss_curve.pdf")
    plot_curves(history, pdf_path)

    # Test evaluation
    print(" Evaluating test set ...")
    res = model.evaluate(ds_te, return_dict=True, verbose=1)
    print(f" Test_loss={res['loss']:.6f}, Test_R={res['masked_PearsonR']:.4f}")

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_peaks": int(X.shape[0]),
        "n_tissues": len(tissue_cols),
        "latent_dim": args.latent_dim,
        "width": args.width,
        "depth": args.depth,
        "mask_rate": args.mask_rate,
        "dropout": args.dropout,
        "best_val_loss": float(np.min(history["val_loss"])),
        "best_val_masked_PearsonR": float(np.max(history["val_masked_PearsonR"])),
        "test_loss": float(res["loss"]),
        "test_masked_PearsonR": float(res["masked_PearsonR"])
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(" Meta saved.")

# ============================================================
# 6. CLI
# ============================================================
def build_parser():
    p = argparse.ArgumentParser(description="Stage 1 Final MAE Trainer")
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="stage1_out_final")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--mask_rate", type=float, default=0.25)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--xla", action="store_true")
    p.add_argument("--no_mixed", action="store_true")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)


