#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Seq→ATAC (Real-peak-length, Chr-split, BPNet Dilated CNN)

import os
import argparse
import json
import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L
import matplotlib.pyplot as plt


def setup_runtime(seed=42, use_amp=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # GPU setup
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f" GPUs detected: {len(gpus)}")
        else:
            print(" No GPU detected, using CPU.")
    except Exception as e:
        print(f" GPU setup warning: {e}")

    if use_amp:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(" Mixed precision: ON")
        except Exception as e:
            print(f" Mixed precision setup failed: {e}")
    else:
        print(" Mixed precision: OFF")


# ==============================
# FASTA read
# ==============================

def load_fasta_as_dict(path):
    seqs = {}
    name = None
    buf = []

    opener = open
    if str(path).endswith(".gz"):
        import gzip
        opener = gzip.open

    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            seqs[name] = "".join(buf).upper()

    if not seqs:
        raise RuntimeError(f"Empty FASTA: {path}")

    print(f" Genome loaded: {len(seqs)} chromosomes from {path}")
    return seqs


# ==============================
# one-hot encoding
# ==============================

BASE2IDX = np.zeros((256,), dtype=np.int32)
BASE2IDX[:] = -1
BASE2IDX[ord("A")] = 0
BASE2IDX[ord("C")] = 1
BASE2IDX[ord("G")] = 2
BASE2IDX[ord("T")] = 3

def one_hot_vectorized(seq: str) -> np.ndarray:
    """ACGTN → (L,4) one-hot, dtype float16 (N is all 0)"""
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = BASE2IDX[arr]  # (L,)
    Ls = len(idx)
    out = np.zeros((Ls, 4), dtype=np.float16)
    mask = idx >= 0
    out[np.arange(Ls)[mask], idx[mask]] = 1.0
    return out


# ==============================
# Extract sequence (peak + flank)
# ==============================

def extract_window_center_flank(chr_seq: str,
                                start: int,
                                end: int,
                                flank: int,
                                seq_len: int) -> str:

    center = (start + end) // 2
    half = seq_len // 2
    left = center - half
    right = center + half

    if left < 0:
        left_pad = -left
        left = 0
    else:
        left_pad = 0
    if right > len(chr_seq):
        right_pad = right - len(chr_seq)
        right = len(chr_seq)
    else:
        right_pad = 0

    window = chr_seq[left:right]
    if left_pad > 0:
        window = "N" * left_pad + window
    if right_pad > 0:
        window = window + "N" * right_pad

    if len(window) != seq_len:
        if len(window) > seq_len:
            window = window[:seq_len]
        else:
            window = window + "N" * (seq_len - len(window))
    return window




META_COLS = ["peak_id", "chr", "start", "end"]

def load_atac_matrix(path):
    df = pd.read_csv(path, sep="\t")
    for c in META_COLS:
        if c not in df.columns:
            raise ValueError(f" ATAC error: Missing column {c}")

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    tissue_cols = [c for c in df.columns if c not in META_COLS]
    Y = df[tissue_cols].to_numpy(np.float32)

    print(f" ATAC matrix loaded: {Y.shape[0]} peaks, {len(tissue_cols)} tissues")
    return df, tissue_cols, Y


def standardize_targets(Y, log1p=True, eps=1e-6):
    # Clip extreme outliers
    q = np.percentile(Y, 99.9, axis=0, keepdims=True)
    Y = np.clip(Y, 0, q)

    if log1p:
        Y = np.log1p(Y, dtype=np.float32)

    mean = Y.mean(axis=0, keepdims=True)
    std = Y.std(axis=0, keepdims=True) + eps
    Yz = (Y - mean) / std
    Yz = np.nan_to_num(Yz).astype(np.float32)

    print(f" Target scaled: min={Yz.min():.3f}, max={Yz.max():.3f}")
    return Yz, {"mean": mean, "std": std}




_GLOBAL_GENOME = None
_GLOBAL_SEQ_LEN = None
_GLOBAL_FLANK = None

def _init_worker(genome_dict, seq_len, flank):
    global _GLOBAL_GENOME, _GLOBAL_SEQ_LEN, _GLOBAL_FLANK
    _GLOBAL_GENOME = genome_dict
    _GLOBAL_SEQ_LEN = seq_len
    _GLOBAL_FLANK = flank

def _worker_one_hot(args):
    idx, chrom, start, end = args
    chr_seq = _GLOBAL_GENOME.get(str(chrom))
    if chr_seq is None:
        raise RuntimeError(f"Chrom {chrom} not found in genome (peak idx={idx})")
    window = extract_window_center_flank(chr_seq,
                                         int(start),
                                         int(end),
                                         flank=_GLOBAL_FLANK,
                                         seq_len=_GLOBAL_SEQ_LEN)
    arr = one_hot_vectorized(window)
    return idx, arr

def build_seq_array(genome_dict,
                    df_peaks,
                    seq_len,
                    flank=200,
                    n_workers=None):
    n = len(df_peaks)
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f" Building sequence array: N={n}, seq_len={seq_len}, flank={flank}, workers={n_workers}")
    X = np.zeros((n, seq_len, 4), dtype=np.float16)

    tasks = [
        (i,
         df_peaks.iloc[i]["chr"],
         df_peaks.iloc[i]["start"],
         df_peaks.iloc[i]["end"])
        for i in range(n)
    ]

    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(genome_dict, seq_len, flank)) as pool:
        for k, arr in pool.imap_unordered(_worker_one_hot, tasks, chunksize=256):
            X[k] = arr
            if (k + 1) % 20000 == 0:
                print(f"  built {k+1}/{n}")

    print(f" Sequence array built: {X.shape}, dtype={X.dtype}")
    return X


# ==============================
# TF Dataset
# ==============================

def make_dataset(X, Y, batch=256, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 200_000))
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================
# BPNet Dilated CNN Model
# ==============================

def bpnet_block(x, filters, dilation, name_prefix):
    h = L.Conv1D(
        filters,
        kernel_size=7,
        padding="same",
        dilation_rate=dilation,
        activation="gelu",
        name=f"{name_prefix}_conv1",
    )(x)
    h = L.Conv1D(
        filters,
        kernel_size=7,
        padding="same",
        dilation_rate=dilation,
        activation="gelu",
        name=f"{name_prefix}_conv2",
    )(h)

    # Residual connection projection if dimensions mismatch
    if int(x.shape[-1]) != filters:
        x = L.Conv1D(filters, kernel_size=1, padding="same",
                     activation=None,
                     name=f"{name_prefix}_proj")(x)

    out = L.Add(name=f"{name_prefix}_add")([x, h])
    out = L.LayerNormalization(name=f"{name_prefix}_ln")(out)
    return out


def build_bpnet_seq2atac_model(seq_len,
                               n_tissues,
                               filters=128,
                               n_blocks=8,
                               dropout=0.15):

    inp = L.Input(shape=(seq_len, 4), dtype="float16", name="seq")

    x = L.Conv1D(
        filters,
        kernel_size=15,
        padding="same",
        activation="gelu",
        name="stem_conv",
    )(inp)

    for i in range(n_blocks):
        d = 2 ** i
        x = bpnet_block(x, filters=filters, dilation=d,
                        name_prefix=f"dblock_{i}_d{d}")

    if dropout > 0:
        x = L.Dropout(dropout, name="global_dropout")(x)

    x = L.GlobalAveragePooling1D(name="gap")(x)

    x = L.Dense(256, activation="gelu", name="fc1")(x)
    if dropout > 0:
        x = L.Dropout(dropout, name="dropout1")(x)
    x = L.Dense(128, activation="gelu", name="fc2")(x)

    out = L.Dense(
        n_tissues,
        activation=None,
        dtype="float32",  # Ensure output is float32 for stability
        name="atac_pred",
    )(x)

    model = tf.keras.Model(inp, out, name="Seq2ATAC_RealPeak_BPNet")
    return model


# ==============================
# PearsonR metric
# ==============================

@tf.function
def batch_pearson(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = y_true - tf.reduce_mean(y_true, axis=1, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)

    cov = tf.reduce_sum(y_true * y_pred, axis=1)
    var_t = tf.reduce_sum(tf.square(y_true), axis=1)
    var_p = tf.reduce_sum(tf.square(y_pred), axis=1)

    r = cov / (tf.sqrt(var_t * var_p) + 1e-6)
    r = tf.clip_by_value(r, -1.0, 1.0)
    mask = tf.math.is_finite(r)
    r = tf.boolean_mask(r, mask)
    return tf.reduce_mean(r)


class PearsonMetric(tf.keras.metrics.Metric):
    def __init__(self, name="pearsonR", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_r = self.add_weight("sum_r", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        r = batch_pearson(y_true, y_pred)
        self.sum_r.assign_add(r)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.sum_r, self.count)

    def reset_states(self):
        self.sum_r.assign(0.0)
        self.count.assign(0.0)


# ==============================
# Plotting
# ==============================

def plot_curves(history, out_pdf):
    plt.figure(figsize=(7, 5))
    if "loss" in history:
        plt.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss")
    if "pearsonR" in history:
        plt.plot(history["pearsonR"], label="Train R")
    if "val_pearsonR" in history:
        plt.plot(history["val_pearsonR"], label="Val R")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Seq→ATAC (Real-peak, BPNet, Chr-split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"📊 Curve saved: {out_pdf}")


# ==============================
# Training Pipeline
# ==============================

def train(args):
    setup_runtime(args.seed, use_amp=not args.no_amp)

    # 1) Load Genome
    genome = load_fasta_as_dict(args.fasta)

    # 2) Load ATAC matrix
    df_peaks, tissue_cols, Y = load_atac_matrix(args.atac_matrix)
    n_tissues = len(tissue_cols)

    # 3) Standardize targets
    Yz, y_stats = standardize_targets(Y, log1p=not args.no_log1p)

    # 4) Determine sequence length
    peak_len = (df_peaks["end"] - df_peaks["start"]).to_numpy()
    max_len = peak_len.max() + 2 * args.flank
    seq_len_auto = int(np.ceil(max_len / 32.0) * 32)

    if args.seq_len is not None:
        seq_len = int(args.seq_len)
        if seq_len < max_len:
            print(f" Warning: seq_len={seq_len} < max(L+2*flank)={max_len}, "
                  f" sequences may be truncated.")
    else:
        seq_len = seq_len_auto

    print(f"📏 max_peak_len={peak_len.max()}, "
          f"max_len_with_flank={max_len}, using seq_len={seq_len}")

    # 5) Build all peak sequences
    X = build_seq_array(genome_dict=genome,
                        df_peaks=df_peaks,
                        seq_len=seq_len,
                        flank=args.flank,
                        n_workers=args.n_workers)

    # Save scaler stats
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "target_scaler_stats.npz"), **y_stats)

    # 6) Split by Chromosome (Train / Val / Test)
    chr_series = df_peaks["chr"].astype(str)
    is_val = chr_series.isin(["Chr9", "9", "chr9"])
    is_test = chr_series.isin(["Chr10", "10", "chr10"])
    is_train = ~(is_val | is_test)

    tr_idx = np.where(is_train)[0]
    val_idx = np.where(is_val)[0]
    te_idx = np.where(is_test)[0]

    print("📦 Split by chromosome:")
    print(f"  Train: {len(tr_idx)} peaks (chr != 9,10)")
    print(f"  Val  : {len(val_idx)} peaks (Chr9)")
    print(f"  Test : {len(te_idx)} peaks (Chr10)")

    X_tr, Y_tr = X[tr_idx], Yz[tr_idx]
    X_val, Y_val = X[val_idx], Yz[val_idx]
    X_te, Y_te = X[te_idx], Yz[te_idx]

    ds_tr = make_dataset(X_tr, Y_tr, batch=args.batch_size, shuffle=True)
    ds_val = make_dataset(X_val, Y_val, batch=args.batch_size, shuffle=False)
    ds_te = make_dataset(X_te, Y_te, batch=args.batch_size, shuffle=False)

    # 7) Build Model
    model = build_bpnet_seq2atac_model(
        seq_len=seq_len,
        n_tissues=n_tissues,
        filters=args.filters,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    )
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[PearsonMetric(name="pearsonR")]
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 8) Train
    print("🚀 Start training Seq→ATAC (Real-peak, BPNet, Chr-split) ...")
    hist = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 9) Save Model & Logs
    model_path = os.path.join(args.outdir, "bpnet_realpeak_seq2atac_model.keras")
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")

    history = hist.history
    log_df = pd.DataFrame(history)
    log_path = os.path.join(args.outdir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"📝 Training log saved: {log_path}")

    pdf_path = os.path.join(args.outdir, "loss_curve.pdf")
    plot_curves(history, pdf_path)

    # 10) Evaluate on TEST set (Chr10)
    print("🧪 Evaluating TEST set (Chr10) ...")
    res = model.evaluate(ds_te, return_dict=True, verbose=1)
    print(f"✅ Test (Chr10): loss={res['loss']:.6f}, R={res['pearsonR']:.4f}")

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_peaks": int(Y.shape[0]),
        "n_tissues": int(n_tissues),
        "seq_len": int(seq_len),
        "auto_seq_len": int(seq_len_auto),
        "max_peak_len": int(peak_len.max()),
        "flank": int(args.flank),
        "filters": int(args.filters),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "best_val_loss": float(np.min(history["val_loss"])),
        "best_val_pearsonR": float(np.max(history["val_pearsonR"])),
        "test_loss": float(res["loss"]),
        "test_pearsonR": float(res["pearsonR"]),
        "tissues": tissue_cols,
        "chr_split": {
            "train": len(tr_idx),
            "val": len(val_idx),
            "test": len(te_idx)
        }
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("🧾 Meta saved.")


# ==============================
# CLI Arguments
# ==============================

def build_parser():
    p = argparse.ArgumentParser(
        description="Seq→ATAC (Real-peak-length, Chr9 val, Chr10 test, BPNet Dilated CNN)"
    )
    p.add_argument("--fasta", required=True,
                   help="Reference Genome FASTA, e.g., Os.fa")
    p.add_argument("--atac_matrix", required=True,
                   help="ATAC Peak x Tissue Matrix TSV (peak_id, chr, start, end + tissues)")
    p.add_argument("--outdir", default="C00_bpnet_seq2atac_out",
                   help="Output directory")

    p.add_argument("--seq_len", type=int, default=None,
                   help="Optional: Manually specify sequence length; Default auto = ceil(max(L+2*flank)/32)*32")
    p.add_argument("--flank", type=int, default=200,
                   help="Flank length on both sides of the peak")

    p.add_argument("--filters", type=int, default=128,
                   help="Number of filters for BPNet convolution blocks")
    p.add_argument("--n_blocks", type=int, default=8,
                   help="Number of dilated blocks")
    p.add_argument("--dropout", type=float, default=0.15,
                   help="Global dropout rate")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=None,
                   help="Number of CPU workers for sequence building; Default = CPU cores - 1")

    p.add_argument("--no_log1p", action="store_true",
                   help="Disable log1p transformation for ATAC intensity (generally not recommended)")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable mixed precision training (use if environment issues occur)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)