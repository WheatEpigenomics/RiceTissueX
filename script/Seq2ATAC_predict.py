#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A01 — BPNet(A00) → Stage1 z256 
----------------------------------------------------------------

input：
  --fasta        WT.fa
  --alt_dir      alt_genomes/
  --peaks        peaks.tsv（peak_id chr start end）
  --bpnet        bpnet_realpeak_seq2atac_model.keras
  --encoder      stage1_encoder.keras

output：
  WT_z.tsv
  ALT1_z.tsv ...
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================================
# Stage1 ：ChannelMask
# =========================================================
@tf.keras.saving.register_keras_serializable(package="stage1")
class ChannelMask(tf.keras.layers.Layer):
    def __init__(self, mask_rate=0.25, **kw):
        super().__init__(**kw)
        self.mask_rate = mask_rate

    def call(self, x, training=None):
        if training:
            b = tf.shape(x)[0]
            c = x.shape[-1]
            m = tf.cast(tf.random.uniform((b, c)) > self.mask_rate, x.dtype)
            return x * m, m
        return x, tf.ones_like(x)

    def get_config(self):
        return {"mask_rate": self.mask_rate}


# =========================================================
# FASTA + one-hot
# =========================================================
def load_fasta(path):
    seqs = {}
    name = None
    buf = []
    op = open
    if path.endswith(".gz"):
        import gzip
        op = gzip.open

    with op(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name:
            seqs[name] = "".join(buf).upper()
    return seqs


BASE2IDX = np.zeros(256, dtype=np.int32)
BASE2IDX[:] = -1
BASE2IDX[ord("A")] = 0
BASE2IDX[ord("C")] = 1
BASE2IDX[ord("G")] = 2
BASE2IDX[ord("T")] = 3

def one_hot(seq):
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = BASE2IDX[arr]
    out = np.zeros((len(idx), 4), np.float32)
    m = idx >= 0
    out[np.arange(len(idx))[m], idx[m]] = 1
    return out


def extract_window(seq, start, end, L):
    center = (start + end) // 2
    half = L // 2
    left = center - half
    right = center + half

    if left < 0:
        padL = -left
        left = 0
    else:
        padL = 0
    if right > len(seq):
        padR = right - len(seq)
        right = len(seq)
    else:
        padR = 0

    w = seq[left:right]
    if padL > 0:
        w = "N" * padL + w
    if padR > 0:
        w = w + "N" * padR

    if len(w) != L:
        if len(w) > L:
            w = w[:L]
        else:
            w = w + "N" * (L - len(w))
    return w


# =========================================================
# predict
# =========================================================
def stream_predict(genome, peaks_df, bpnet, encoder,
                   seq_len, batch_size):

    N = len(peaks_df)
    rows = []

    for i0 in range(0, N, batch_size):
        sub = peaks_df.iloc[i0:i0 + batch_size]

        X = np.zeros((len(sub), seq_len, 4), np.float32)
        for j, (_, r) in enumerate(sub.iterrows()):
            seq = genome[r["chr"]]
            w = extract_window(seq, int(r.start), int(r.end), seq_len)
            X[j] = one_hot(w)

        atac = bpnet.predict(X, verbose=0)
        z = encoder.predict(atac, verbose=0)

        for k in range(len(sub)):
            rows.append([sub.iloc[k]["peak_id"]] + list(z[k]))

        print(f"  batch {i0}/{N} done", flush=True)

    return pd.DataFrame(rows,
                        columns=["peak_id"] + [f"z{i+1}" for i in range(z.shape[1])])


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--alt_dir", required=True)
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--bpnet", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--chr", default=None,
                    help="Chr10 / 10 / chr10;")
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------------- WT genome ----------------
    print("Load WT genome...")
    wt = load_fasta(args.fasta)

    print("Load peaks.tsv ...")
    peaks = pd.read_csv(args.peaks, sep="\t")

    # ---------- Filter by chromosome (optional) ----------
    if args.chr:
        tag = str(args.chr).replace("chr", "").replace("Chr", "")
        target_chr = f"Chr{tag}"
        print(f" Only predict chromosome {target_chr}")

        peaks = peaks[peaks["chr"] == target_chr].reset_index(drop=True)
        print(f" Number of peaks to predict: {len(peaks)}")
    else:
        print(" --chr not set, predicting all peaks")

    # -------------- Load BPNet ----------------
    print(" Load BPNet(A00)...")
    bpnet = tf.keras.models.load_model(args.bpnet, compile=False)

    # Auto seq_len
    seq_len = args.seq_len if args.seq_len else bpnet.input_shape[1]
    print(f" Using seq_len = {seq_len}")

    # ------------- Load Stage1 encoder ------------
    print(" Load Stage1 encoder...")
    encoder = tf.keras.models.load_model(
        args.encoder,
        compile=False,
        custom_objects={"ChannelMask": ChannelMask}
    )

    # ---------------- WT ----------------
    print("\n Predict WT ...")
    df_wt = stream_predict(
        wt, peaks, bpnet, encoder,
        seq_len=seq_len,
        batch_size=args.batch_size,
    )
    df_wt.to_csv(f"{args.outdir}/WT_z.tsv", sep="\t", index=False)
    print(" WT done")

    # ---------------- ALT1–ALT10 ----------------
    for i in range(1, 10 + 1):
        fa = os.path.join(args.alt_dir, f"ALT{i}.fa")
        if not os.path.exists(fa):
            print(f"⚠ ALT{i}.fa not found, skip")
            continue

        print(f"\n Predict ALT{i}...")
        alt = load_fasta(fa)

        df_alt = stream_predict(
            alt, peaks, bpnet, encoder,
            seq_len=seq_len,
            batch_size=args.batch_size,
        )
        df_alt.to_csv(f"{args.outdir}/ALT{i}_z.tsv",
                      sep="\t", index=False)
        print(f"ALT{i} done")


if __name__ == "__main__":
    main()