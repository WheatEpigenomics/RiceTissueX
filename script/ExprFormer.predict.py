#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A04 — Stage2++ v4.2 StableVal
Only predict gene expression (y_pred), no explanation outputs
----------------------------------------------------------------
Compatible with your training model:
 - ClusterPoolLayer
 - SignedContribution
 - TissueMoEReadout
 - Internal Lambda (safe_mode=False)
----------------------------------------------------------------
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L
from tqdm import tqdm


# ============================================================
# 1. Register custom layers (must match your training code!)
# ============================================================

@tf.keras.saving.register_keras_serializable(package="stage2pp")
class ClusterPoolLayer(L.Layer):
    def __init__(self, topk, z_dim, d_pool=128, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.z_dim = z_dim
        self.d_pool = d_pool

    def build(self, input_shape):
        self.q = self.add_weight(
            "cluster_q",
            shape=(1, 4, 1, self.d_pool),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )
        self.proj = L.TimeDistributed(
            L.TimeDistributed(L.Dense(self.d_pool, activation="gelu"))
        )

    def call(self, z_in):
        h = self.proj(z_in)                       # (B,4,K,d)
        att = tf.nn.softmax(tf.reduce_sum(h * self.q, axis=-1), axis=2)
        v = tf.reduce_sum(h * att[..., None], axis=2)
        return v, att


@tf.keras.saving.register_keras_serializable(package="stage2pp")
class SignedContribution(L.Layer):
    def __init__(self, d_seq=256, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = L.Dense(256, activation="gelu")
        self.fc2 = L.Dense(1)
        self.concat = L.Concatenate()

    def call(self, v_c, g_seq, sign_in):
        outs = []
        for c in range(4):
            h = self.concat([g_seq, v_c[:, c, :]])
            h = self.fc1(h)
            out = tf.tanh(self.fc2(h)) * sign_in[:, c:c+1]
            outs.append(out)
        return tf.concat(outs, axis=1)


@tf.keras.saving.register_keras_serializable(package="stage2pp")
class TissueMoEReadout(L.Layer):
    def __init__(self, n_tissues, d_token, d_hidden=128, **kwargs):
        super().__init__(**kwargs)
        self.n_tissues = int(n_tissues)
        self.d_token = int(d_token)
        self.d_hidden = int(d_hidden)
        self.mlp1 = L.Dense(self.d_hidden, activation="gelu")
        self.mlp2 = L.Dense(1)

    def build(self, input_shape):
        self.E = self.add_weight(
            "tissue_embed",
            shape=(self.n_tissues, self.d_token),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )
        self.bias = self.add_weight(
            "tissue_bias",
            shape=(self.n_tissues,),
            initializer="zeros",
        )

    def call(self, tokens):
        logits = tf.einsum("bcd,nd->bcn", tokens, self.E)
        alpha = tf.nn.softmax(logits, axis=1)
        u = tf.einsum("bcd,bcn->bnd", tokens, alpha)
        h = self.mlp1(u)
        y = self.mlp2(h)[..., 0] + self.bias[None, :]
        return y, alpha


# ============================================================
# 2. Constants (same as training)
# ============================================================

TSS_UP, TSS_DOWN = 2000, 1500
TTS_UP, TTS_DOWN = 500, 1500
SEQ_LEN = TSS_UP + TSS_DOWN + TTS_UP + TTS_DOWN
TOPK = 16


# ============================================================
# 3. FASTA / GFF / PCC / z_peak utilities
# ============================================================

def load_fasta(path):
    seqs = {}
    name, buf = None, []
    for line in open(path):
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


def revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def one_hot(seq):
    arr = np.zeros((len(seq), 4), np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, c in enumerate(seq):
        if c in m:
            arr[i, m[c]] = 1
    return arr


def get_window(row, fasta_dict):
    chr_seq = fasta_dict.get(row.chr)
    if chr_seq is None:
        return None
    if row.strand == "+":
        TSS = row.start
        TTS = row.end
        rev = False
    else:
        TSS = row.end
        TTS = row.start
        rev = True

    t1 = chr_seq[max(0, TSS - TSS_UP): TSS + TSS_DOWN]
    t2 = chr_seq[max(0, TTS - TTS_UP): TTS + TTS_DOWN]
    seq = t1 + t2
    if len(seq) != SEQ_LEN:
        return None
    if rev:
        seq = revcomp(seq)
    return seq


def build_topk(gid, pcc_dict, z_dict):
    z_dim = len(next(iter(z_dict.values())))
    Z = np.zeros((4, TOPK, z_dim), np.float32)
    sign = np.ones(4, np.float32)
    mag = np.zeros(4, np.float32)

    if gid not in pcc_dict:
        return Z, sign, mag

    df = pcc_dict[gid]
    df["abs"] = df["PCC"].abs()
    for c in range(4):
        sub = df[df["cluster_idx"] == c]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("abs", ascending=False).head(TOPK)
        sign[c] = 1 if sub["PCC"].mean() >= 0 else -1
        mag[c] = sub["abs"].mean()
        for i, pk in enumerate(sub["peak_id"].values[:TOPK]):
            if pk in z_dict:
                Z[c, i] = z_dict[pk]

    if mag.max() > 0:
        mag /= (mag.max() + 1e-6)
    return Z, sign, mag


# ============================================================
# 4. Prediction main
# ============================================================

def main(args):

    print("📌 Loading model...")
    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        safe_mode=False,      # required because Lambda exists
    )

    fasta = load_fasta(args.fasta)
    expr = pd.read_csv(args.expr, sep="\t")
    tissues = [c for c in expr.columns if c != "gene_id"]

    # PCC
    pcc_df = pd.read_csv(args.pcc, sep="\t")
    if "cluster_idx" not in pcc_df.columns:
        cmap = {"Cluster1": 0, "Cluster2": 1, "Cluster3": 2, "Cluster4": 3}
        pcc_df["cluster_idx"] = pcc_df["cluster"].map(cmap)
    pcc_dict = {gid: df for gid, df in pcc_df.groupby("gene_id")}

    # z_peak
    zdf = pd.read_csv(args.z_peak, sep="\t")
    zcols = [c for c in zdf.columns if c != "peak_id"]
    z_dict = {r["peak_id"]: r[zcols].values.astype(np.float32)
              for _, r in zdf.iterrows()}

    # gene list
    target_genes = [g.strip() for g in open(args.genes)]

    # parse GFF
    print("📌 Loading genes from GFF...")
    rows = []
    for line in open(args.gff):
        if line.startswith("#"):
            continue
        arr = line.split("\t")
        if len(arr) < 9 or arr[2] != "gene":
            continue
        chrom, _, _, s, e, _, strand, _, info = arr
        gid = None
        for kv in info.split(";"):
            if kv.startswith("ID="):
                gid = kv.split("=")[1].split(":")[-1]
        if gid in target_genes:
            rows.append((gid, chrom, int(s), int(e), strand))
    gff_df = pd.DataFrame(rows, columns=["gene_id", "chr", "start", "end", "strand"])

    # build inputs
    X_seq, X_z, X_sign, X_mag, GIDs = [], [], [], [], []

    print("📌 Building inputs...")
    for _, row in tqdm(gff_df.iterrows(), total=len(gff_df)):
        seq = get_window(row, fasta)
        if seq is None:
            continue
        seq_oh = one_hot(seq)

        Z, sign, mag = build_topk(row.gene_id, pcc_dict, z_dict)

        X_seq.append(seq_oh)
        X_z.append(Z)
        X_sign.append(sign)
        X_mag.append(mag)
        GIDs.append(row.gene_id)

    X_seq = np.array(X_seq, np.float32)
    X_z = np.array(X_z, np.float32)
    X_sign = np.array(X_sign, np.float32)
    X_mag = np.array(X_mag, np.float32)

    # predict
    print("📌 Predicting...")
    ds = tf.data.Dataset.from_tensor_slices((X_seq, X_z, X_sign, X_mag)).batch(args.batch)

    preds = []
    for a, b, c, d in ds:
        y = model([a, b, c, d], training=False)[0]  # index 0 = y_pred
        preds.append(y.numpy())
    preds = np.concatenate(preds, axis=0)

    # output
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, args.prefix + "_pred.tsv")

    df = pd.DataFrame(preds, columns=tissues)
    df.insert(0, "gene_id", GIDs)
    df.to_csv(out, sep="\t", index=False)

    print(f"✅ Done. Output saved at: {out}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--expr", required=True)
    ap.add_argument("--pcc", required=True)
    ap.add_argument("--z_peak", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default="predict_expr_out")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    main(args)

