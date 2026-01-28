#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers as L, regularizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error


# ============================
# Constants
# ============================
TSS_UP, TSS_DOWN = 2000, 1500
TTS_UP, TTS_DOWN = 500, 1500
SEQ_LEN_EXPECTED = (TSS_UP + TSS_DOWN) + (TTS_UP + TTS_DOWN)  # 5500

TOPK_PER_CLUSTER = 16
L2_WEIGHT = 1e-5
GAUSS_NOISE = 0.05

LAMBDA_MAG = 0.05
LAMBDA_PCC = 0.02
LAMBDA_KO = 0.01
LAMBDA_IMP = 0.02


# ============================
# Runtime
# ============================
def setup_runtime(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print(" Runtime ready (GPU memory growth enabled)")


# ============================
# Warmup + Cosine LR
# ============================
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_base, warmup_steps, total_steps):
        super().__init__()
        self.lr_base = tf.constant(lr_base, tf.float32)
        self.warmup_steps = tf.constant(max(1, int(warmup_steps)), tf.float32)
        self.total_steps = tf.constant(max(2, int(total_steps)), tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.lr_base * (step / self.warmup_steps)
        prog = tf.clip_by_value(
            (step - self.warmup_steps) /
            tf.maximum(self.total_steps - self.warmup_steps, 1.0),
            0.0, 1.0
        )
        cosine = 0.5 * self.lr_base * (1.0 + tf.cos(np.pi * prog))
        return tf.where(step < self.warmup_steps, warm, cosine)


# ============================
# FASTA / GFF
# ============================
def load_fasta(path):
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf).upper()
                buf = []
                name = line[1:].strip().split()[0]
            else:
                buf.append(line.strip())
        if name:
            seqs[name] = "".join(buf).upper()
    return seqs


def revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def parse_genes_from_gff(gff, allowed_genes):
    genes = []
    allowed = set(allowed_genes)
    with open(gff) as f:
        for line in f:
            if line.startswith("#"):
                continue
            arr = line.strip().split("\t")
            if len(arr) < 9:
                continue
            if arr[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attr = arr
            gid = None
            for kv in attr.split(";"):
                if kv.startswith("ID="):
                    gid = kv.split("=")[1].split(":")[-1]
                    break
            if gid and gid in allowed:
                genes.append((gid, chrom, int(start), int(end), strand))
    return pd.DataFrame(genes, columns=["gene_id", "chr", "start", "end", "strand"])


# ============================
# Sequence window + segments
# ============================
def get_gene_window_and_seq(row, fasta_dict):
    """
    Returns:
      seq: final 5500bp sequence (5'->3' unified)
      chr_name: chromosome
      segments: list of (label, win_l_1, win_r_1, pre_offset) where pre_offset is in seq_raw coords
      need_rev: True if '-' strand (final seq = revcomp(seq_raw))
    """
    chr_seq = fasta_dict.get(row.chr, "")
    if not chr_seq:
        return None, None, None, None

    chr_len = len(chr_seq)
    start1 = int(row.start)
    end1 = int(row.end)
    strand = row.strand

    if strand == "+":
        TSS1 = start1
        TTS1 = end1
        need_rev = False
    else:
        TSS1 = end1
        TTS1 = start1
        need_rev = True

    TSS0 = TSS1 - 1
    TTS0 = TTS1 - 1

    if not need_rev:
        tss_up_start0   = TSS0 - TSS_UP
        tss_up_end0     = TSS0
        tss_down_start0 = TSS0
        tss_down_end0   = TSS0 + TSS_DOWN

        tts_up_start0   = TTS0 - TTS_UP
        tts_up_end0     = TTS0
        tts_down_start0 = TTS0
        tts_down_end0   = TTS0 + TTS_DOWN

        for s0, e0 in [
            (tss_up_start0, tss_up_end0),
            (tss_down_start0, tss_down_end0),
            (tts_up_start0, tts_up_end0),
            (tts_down_start0, tts_down_end0),
        ]:
            if s0 < 0 or e0 > chr_len or s0 >= e0:
                return None, None, None, None

        seq = (
            chr_seq[tss_up_start0:tss_up_end0] +
            chr_seq[tss_down_start0:tss_down_end0] +
            chr_seq[tts_up_start0:tts_up_end0] +
            chr_seq[tts_down_start0:tts_down_end0]
        )

        segments = [
            ("TSS_up",   tss_up_start0+1,   tss_up_end0+1,     0),
            ("TSS_down", tss_down_start0+1, tss_down_end0+1,   2000),
            ("TTS_up",   tts_up_start0+1,   tts_up_end0+1,     3500),
            ("TTS_down", tts_down_start0+1, tts_down_end0+1,   4000),
        ]

    else:
        tss_up_start0   = TSS0 + 1
        tss_up_end0     = TSS0 + 1 + TSS_UP

        tss_down_start0 = TSS0 - TSS_DOWN
        tss_down_end0   = TSS0

        tts_up_start0   = TTS0
        tts_up_end0     = TTS0 + TTS_UP

        tts_down_start0 = TTS0 - TTS_DOWN
        tts_down_end0   = TTS0

        for s0, e0 in [
            (tss_up_start0, tss_up_end0),
            (tss_down_start0, tss_down_end0),
            (tts_up_start0, tts_up_end0),
            (tts_down_start0, tts_down_end0),
        ]:
            if s0 < 0 or e0 > chr_len or s0 >= e0:
                return None, None, None, None

        seq_tss_up   = chr_seq[tss_up_start0:tss_up_end0]
        seq_tss_down = chr_seq[tss_down_start0:tss_down_end0]
        seq_tts_up   = chr_seq[tts_up_start0:tts_up_end0]
        seq_tts_down = chr_seq[tts_down_start0:tts_down_end0]

        seq_raw = seq_tts_down + seq_tts_up + seq_tss_down + seq_tss_up
        seq = revcomp(seq_raw)

        segments = [
            ("TTS_down", tts_down_start0+1, tts_down_end0+1,   0),
            ("TTS_up",   tts_up_start0+1,   tts_up_end0+1,     1500),
            ("TSS_down", tss_down_start0+1, tss_down_end0+1,   2000),
            ("TSS_up",   tss_up_start0+1,   tss_up_end0+1,     3500),
        ]

    if len(seq) != SEQ_LEN_EXPECTED:
        return None, None, None, None

    return seq, row.chr, segments, need_rev


def one_hot(seq):
    arr = np.zeros((len(seq), 4), np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq.upper()):
        if ch in m:
            arr[i, m[ch]] = 1.0
    return arr


# ============================
# DynamicMask (fixed)
# ============================
def build_dynamic_masks_for_gene(gene_chr, segments, need_revcomp, pcc_gene_df):
    L = SEQ_LEN_EXPECTED
    if pcc_gene_df is None or len(pcc_gene_df) == 0:
        return np.zeros(L, np.float32), np.zeros(L, np.float32)

    required_cols = {"chr", "start", "end", "PCC"}
    if not required_cols.issubset(pcc_gene_df.columns):
        raise ValueError(f"[DynamicMask] pcc_tsv must include {required_cols}")

    sum_pos = np.zeros(L, np.float32)
    sum_neg = np.zeros(L, np.float32)

    def add_interval(gen_s, gen_e, win_l, win_r, pre_off, v_pos, v_neg):
        inter_l = max(gen_s, win_l)
        inter_r = min(gen_e, win_r)
        if inter_r <= inter_l:
            return

        pre_s = pre_off + (inter_l - win_l)
        pre_e = pre_off + (inter_r - win_l)

        if not need_revcomp:
            s, e = pre_s, pre_e
        else:
            s, e = L - pre_e, L - pre_s

        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            return

        if v_pos != 0:
            sum_pos[s:e] += v_pos
        if v_neg != 0:
            sum_neg[s:e] += v_neg

    for _, r in pcc_gene_df.iterrows():
        if r["chr"] != gene_chr:
            continue

        p_s, p_e = int(r["start"]), int(r["end"])
        if p_e <= p_s:
            continue

        p = float(r["PCC"])
        if p == 0:
            continue

        abs_w = abs(p)
        pos_val = max(p, 0.0) * abs_w
        neg_val = max(-p, 0.0) * abs_w

        for (_, win_l, win_r, pre_off) in segments:
            add_interval(p_s, p_e, win_l, win_r, pre_off, pos_val, neg_val)

    kernel = np.array([0.25, 0.5, 0.25], np.float32)

    def smooth(x):
        out = x.copy()
        for _ in range(2):
            out = np.convolve(out, kernel, mode="same")
        return out

    pos = smooth(sum_pos)
    neg = smooth(sum_neg)

    if pos.max() > 0:
        pos = pos / (pos.max() + 1e-6)
    if neg.max() > 0:
        neg = neg / (neg.max() + 1e-6)

    return pos.astype(np.float32), neg.astype(np.float32)


# ============================
# Data preparation
# ============================
def prepare_samples(
    fasta,
    gff,
    expr_tsv,
    z_peak_tsv,
    pcc_tsv,
    cluster_map_tsv,
    limit=None,
    threads=1
):
    expr = pd.read_csv(expr_tsv, sep="\t")
    if "gene_id" not in expr.columns:
        raise ValueError("expr TSV must contain 'gene_id' column")
    allowed = set(expr["gene_id"])
    tissues = [c for c in expr.columns if c != "gene_id"]

    cmap = pd.read_csv(cluster_map_tsv, sep="\t", header=None, names=["tissue", "cluster"])
    c2i = {"Cluster1": 0, "Cluster2": 1, "Cluster3": 2, "Cluster4": 3}
    cluster_of_tissue = np.array([c2i[cmap.set_index("tissue").loc[t, "cluster"]] for t in tissues], np.int32)

    fasta_dict = load_fasta(fasta)
    genes = parse_genes_from_gff(gff, allowed)
    print(f"Loaded {len(genes)} genes from GFF (intersect with expr)")

    zdf = pd.read_csv(z_peak_tsv, sep="\t")
    z_cols = [c for c in zdf.columns if c != "peak_id"]
    z_dict = {r["peak_id"]: r[z_cols].values.astype(np.float32) for _, r in zdf.iterrows()}
    z_dim = len(z_cols)
    print(f"Loaded z_peak embeddings: {len(z_dict)} peaks, dim={z_dim}")

    pcc = pd.read_csv(pcc_tsv, sep="\t")
    if "cluster_idx" not in pcc.columns:
        c2i_p = {"Cluster1": 0, "Cluster2": 1, "Cluster3": 2, "Cluster4": 3}
        if "cluster" in pcc.columns:
            if pcc["cluster"].dtype.kind in "iu":
                pcc["cluster_idx"] = pcc["cluster"].astype(np.int32)
            else:
                pcc["cluster_idx"] = pcc["cluster"].map(c2i_p).astype(np.int32)
        else:
            raise ValueError("PCC table must have 'cluster' or 'cluster_idx'")
    pcc["abs_pcc"] = pcc["PCC"].abs()
    pcc_dict = {gid: df for gid, df in pcc.groupby("gene_id")}
    print(f"PCC indexed for {len(pcc_dict)} genes")

    def topk_for_gene(gid):
        z_stack = np.zeros((4, TOPK_PER_CLUSTER, z_dim), np.float32)
        pid_stack = np.full((4, TOPK_PER_CLUSTER), "", dtype=object)
        sign_c = np.ones(4, np.float32)
        mag_c = np.zeros(4, np.float32)

        if gid not in pcc_dict:
            return z_stack, sign_c, mag_c, pid_stack

        sub_df = pcc_dict[gid]
        for c in range(4):
            sub = sub_df[sub_df["cluster_idx"] == c]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("abs_pcc", ascending=False).head(TOPK_PER_CLUSTER)
            sign_c[c] = 1.0 if sub["PCC"].mean() >= 0 else -1.0
            mag_c[c] = sub["abs_pcc"].mean()
            pks = sub["peak_id"].values[:TOPK_PER_CLUSTER]
            for i, pk in enumerate(pks):
                pid_stack[c, i] = pk
                if pk in z_dict:
                    z_stack[c, i] = z_dict[pk]
        if mag_c.max() > 0:
            mag_c /= (mag_c.max() + 1e-6)
        return z_stack, sign_c, mag_c, pid_stack

    results = []
    chroms = []

    for _, row in tqdm(genes.iterrows(), total=len(genes), desc="Build samples"):
        seq, gchr, segments, need_rev = get_gene_window_and_seq(row, fasta_dict)
        if seq is None:
            continue

        z_stack, sign_c, mag_c, pid_stack = topk_for_gene(row.gene_id)

        y = expr[expr["gene_id"] == row.gene_id][tissues].values[0].astype(np.float32)
        y = np.log1p(y)

        pcc_gene_df = pcc_dict.get(row.gene_id, None)
        act_mask_1d, rep_mask_1d = build_dynamic_masks_for_gene(
            gchr, segments, need_rev, pcc_gene_df
        )

        results.append((
            row.gene_id, row.chr, seq, z_stack, sign_c, mag_c, y, pid_stack, act_mask_1d, rep_mask_1d
        ))
        chroms.append(row.chr)

        if limit is not None and len(results) >= limit:
            break

    N = len(results)
    X_seq = np.zeros((N, SEQ_LEN_EXPECTED, 4), np.float32)
    X_z = np.zeros((N, 4, TOPK_PER_CLUSTER, z_dim), np.float32)
    X_sign = np.zeros((N, 4), np.float32)
    X_pcc = np.zeros((N, 4), np.float32)
    X_act = np.zeros((N, SEQ_LEN_EXPECTED, 1), np.float32)
    X_rep = np.zeros((N, SEQ_LEN_EXPECTED, 1), np.float32)
    Y = np.zeros((N, len(tissues)), np.float32)

    gene_ids = []
    Chr = []
    PID = np.empty((N, 4, TOPK_PER_CLUSTER), dtype=object)

    for i, (gid, chrom, seq, zs, sg, mg, y, pids, act_1d, rep_1d) in enumerate(results):
        X_seq[i] = one_hot(seq)
        X_z[i] = zs
        X_sign[i] = sg
        X_pcc[i] = mg
        X_act[i, :, 0] = act_1d
        X_rep[i, :, 0] = rep_1d
        Y[i] = y
        gene_ids.append(gid)
        Chr.append(chrom)
        PID[i] = pids

    print(f"Built {N} samples")

    return dict(
        X_seq=X_seq,
        X_z=X_z,
        X_sign=X_sign,
        X_pcc=X_pcc,
        X_act_mask=X_act,
        X_rep_mask=X_rep,
        Y=Y,
        gene_ids=np.array(gene_ids),
        chroms=np.array(Chr),
        tissues=tissues,
        cluster_of_tissue=cluster_of_tissue,
        z_dim=z_dim,
        peak_ids_stack=PID,
    )


# ============================
# Model blocks
# ============================
def ConvBlock(x, f, k, drop):
    x = L.Conv1D(
        f, k, padding="same", activation="gelu",
        kernel_regularizer=regularizers.l2(L2_WEIGHT),
    )(x)
    x = L.BatchNormalization()(x)
    if drop > 0:
        x = L.Dropout(drop)(x)
    return x


def UNet_backbone(seq_len, base=64, d_out=256):
    inp = L.Input((seq_len, 4), name="seq_onehot")

    def add_region_embed(x):
        B = tf.shape(x)[0]
        pos = tf.range(seq_len)[None, :, None]
        pos = tf.tile(pos, [B, 1, 1])

        promoter = tf.cast(pos < TSS_UP, tf.float32)
        body = tf.cast(
            (pos >= TSS_UP) &
            (pos < (TSS_UP + TSS_DOWN + TTS_UP)),
            tf.float32
        )
        tts = tf.cast(pos >= (TSS_UP + TSS_DOWN + TTS_UP), tf.float32)
        region = tf.concat([promoter, body, tts], axis=-1)
        return tf.concat([x, region], axis=-1)

    x = L.Lambda(add_region_embed, name="add_region_embed")(inp)
    x = L.GaussianNoise(GAUSS_NOISE)(x)

    c1 = ConvBlock(x, base, 9, 0.15)
    p1 = L.MaxPool1D(2)(c1)

    c2 = ConvBlock(p1, base * 2, 7, 0.15)
    p2 = L.MaxPool1D(2)(c2)

    c3 = ConvBlock(p2, base * 4, 5, 0.20)
    p3 = L.MaxPool1D(5)(c3)

    b = ConvBlock(p3, base * 8, 3, 0.25)
    att = L.MultiHeadAttention(num_heads=4, key_dim=64)(b, b)
    b = L.Add()([b, att])
    b = L.LayerNormalization()(b)

    u2 = L.UpSampling1D(5)(b)
    u2 = L.Concatenate()([u2, c3])
    u2 = ConvBlock(u2, base * 4, 5, 0.10)

    u1 = L.UpSampling1D(2)(u2)
    u1 = L.Concatenate()([u1, c2])
    u1 = ConvBlock(u1, base * 2, 7, 0.05)

    u0 = L.UpSampling1D(2)(u1)
    u0 = L.Concatenate()([u0, c1])
    u0 = ConvBlock(u0, base, 9, 0.05)

    feat = L.Conv1D(d_out, 1, activation="gelu", name="seq_feat")(u0)

    att_logits = L.Dense(1, name="seq_att_logits")(feat)
    att_weights = L.Softmax(axis=1, name="seq_att_weights")(att_logits)
    g_seq_weighted = L.Multiply(name="seq_att_weighted")([feat, att_weights])
    g_seq = L.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="seq_att_pool")(g_seq_weighted)

    return tf.keras.Model(inp, [feat, g_seq], name="UNet_backbone_v4_2")


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
        self.proj = L.TimeDistributed(L.TimeDistributed(L.Dense(self.d_pool, activation="gelu")))

    def call(self, z_in):
        h = self.proj(z_in)
        att = tf.nn.softmax(tf.reduce_sum(h * self.q, axis=-1), axis=2)
        v = tf.reduce_sum(h * att[..., None], axis=2)
        return v, att


@tf.keras.saving.register_keras_serializable(package="stage2pp")
class SignedContribution(L.Layer):
    def __init__(self, d_seq, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = L.Dense(256, activation="gelu")
        self.fc2 = L.Dense(1)
        self.concat = L.Concatenate()
        self.d_seq = d_seq

    def call(self, v_c, g_seq, sign_in):
        s_list = []
        for c in range(4):
            vc = v_c[:, c, :]
            h = self.concat([g_seq, vc])
            h = self.fc1(h)
            r = self.fc2(h)
            s = tf.tanh(r) * sign_in[:, c:c+1]
            s_list.append(s)
        return tf.concat(s_list, axis=1)


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
        self.bias = self.add_weight("tissue_bias", shape=(self.n_tissues,), initializer="zeros")

    def call(self, tokens):
        logits = tf.einsum("bcd,nd->bcn", tokens, self.E)
        alpha = tf.nn.softmax(logits, axis=1)
        u = tf.einsum("bcd,bcn->bnd", tokens, alpha)
        h = self.mlp1(u)
        y = self.mlp2(h)[..., 0] + self.bias[None, :]
        return y, alpha


def build_stage2pp_v4_2(seq_len, z_dim, n_tissues, d_seq=256, d_pool=128, d_head=128):
    seq_in = L.Input((seq_len, 4), name="seq_in")
    z_in = L.Input((4, TOPK_PER_CLUSTER, z_dim), name="z_in")
    sign_in = L.Input((4,), name="sign_in")
    pcc_in = L.Input((4,), name="pcc_in")

    feat, g_seq = UNet_backbone(seq_len, 64, d_seq)(seq_in)

    base_imp = L.Conv1D(1, 1, padding="same", activation="tanh", name="base_importance")(feat)

    v_c, att_cluster = ClusterPoolLayer(TOPK_PER_CLUSTER, z_dim, d_pool)(z_in)

    h_seq = L.Dense(d_head, activation="gelu", name="seq_head_dense")(g_seq)
    y_seq = L.Dense(n_tissues, name="seq_head_out")(h_seq)

    v_proj = L.Dense(d_head, activation="gelu", name="atac_token_proj")(v_c)
    y_atac, alpha = TissueMoEReadout(n_tissues, d_head, d_hidden=128, name="atac_moe")(v_proj)

    y_pred = L.Add(name="y_pred_sum")([y_seq, y_atac])

    S = SignedContribution(d_seq, name="signed_contrib")(v_c, g_seq, sign_in)

    model = tf.keras.Model(
        [seq_in, z_in, sign_in, pcc_in],
        [y_pred, S, att_cluster, base_imp, alpha],
        name="Stage2PP_v4_2_StableVal_strandfix_explainSampling"
    )
    return model


# ============================
# Trainer
# ============================
def batch_pcc(y_true, y_pred):
    y_true = y_true - tf.reduce_mean(y_true, axis=0, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, axis=0, keepdims=True)
    num = tf.reduce_sum(y_true * y_pred, axis=0)
    den = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=0) * tf.reduce_sum(y_pred ** 2, axis=0) + 1e-6)
    return tf.reduce_mean(num / den)


class Trainer(tf.keras.Model):
    def __init__(self, base, lm=LAMBDA_MAG, lp=LAMBDA_PCC, lk=LAMBDA_KO, li=LAMBDA_IMP):
        super().__init__()
        self.base = base
        self.lm = lm
        self.lp = lp
        self.lk = lk
        self.li = li

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.total_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.mag_tracker = tf.keras.metrics.Mean(name="mag_loss")
        self.pcc_tracker = tf.keras.metrics.Mean(name="pcc_loss")
        self.ko_tracker = tf.keras.metrics.Mean(name="ko_loss")
        self.imp_tracker = tf.keras.metrics.Mean(name="imp_loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.total_tracker,
            self.mag_tracker,
            self.pcc_tracker,
            self.ko_tracker,
            self.imp_tracker,
        ]

    def _importance_coherence_loss(self, base_imp, act_mask, rep_mask):
        imp = base_imp[..., 0]
        act = act_mask[..., 0]
        rep = rep_mask[..., 0]

        eps = 1e-6
        act_sum = tf.reduce_sum(act, axis=1) + eps
        rep_sum = tf.reduce_sum(rep, axis=1) + eps

        imp_pos = tf.reduce_sum(imp * act, axis=1) / act_sum
        imp_neg = tf.reduce_sum(imp * rep, axis=1) / rep_sum

        pos_penalty = tf.nn.relu(-imp_pos)
        neg_penalty = tf.nn.relu(imp_neg)
        return tf.reduce_mean(pos_penalty + neg_penalty)

    def _loss_components(self, y, outputs, p_guid, sign_in,
                         act_mask=None, rep_mask=None,
                         y_ko_neg=None, y_ko_pos=None):
        y_hat, S, att_cluster, base_imp, alpha = outputs

        expr_loss = tf.reduce_mean(tf.keras.losses.huber(y, y_hat))
        pcc_loss = 1.0 - batch_pcc(y, y_hat)

        S_abs = tf.abs(S)
        S_n = S_abs / (tf.reduce_max(S_abs, axis=1, keepdims=True) + 1e-6)
        P_max = tf.reduce_max(p_guid, axis=1, keepdims=True)
        valid = tf.cast(P_max > 1e-8, tf.float32)
        P_n = tf.where(valid > 0, p_guid / (P_max + 1e-6), tf.zeros_like(p_guid))
        mag_per = tf.reduce_mean(tf.square(S_n - P_n), axis=1, keepdims=True)
        mag_loss = tf.reduce_sum(mag_per * valid) / (tf.reduce_sum(valid) + 1e-6)

        ko_loss = tf.constant(0.0, tf.float32)
        if self.lk > 0 and (y_ko_neg is not None) and (y_ko_pos is not None):
            pos_mask = tf.cast(sign_in > 0, tf.float32)
            neg_mask = tf.cast(sign_in < 0, tf.float32)
            pos_w = tf.reduce_sum(pos_mask * p_guid, axis=1) / (tf.reduce_sum(pos_mask, axis=1) + 1e-6)
            neg_w = tf.reduce_sum(neg_mask * p_guid, axis=1) / (tf.reduce_sum(neg_mask, axis=1) + 1e-6)

            ko_neg = tf.nn.relu(y_hat - y_ko_neg)
            ko_pos = tf.nn.relu(y_ko_pos - y_hat)

            ko_neg_sample = tf.reduce_mean(ko_neg, axis=1) * neg_w
            ko_pos_sample = tf.reduce_mean(ko_pos, axis=1) * pos_w

            ko_loss = tf.reduce_mean(ko_neg_sample + ko_pos_sample)

        imp_loss = tf.constant(0.0, tf.float32)
        if self.li > 0 and (act_mask is not None) and (rep_mask is not None):
            imp_loss = self._importance_coherence_loss(base_imp, act_mask, rep_mask)

        total_loss = (
            expr_loss +
            self.lm * mag_loss +
            self.lp * pcc_loss +
            self.lk * ko_loss +
            self.li * imp_loss
        )
        return total_loss, expr_loss, mag_loss, pcc_loss, ko_loss, imp_loss

    def train_step(self, data):
        (seq, z, s, p, act_mask, rep_mask), y = data
        with tf.GradientTape() as t:
            outputs = self.base([seq, z, s, p], training=True)

            if self.lk > 0:
                seq_ko_neg = seq * (1.0 - rep_mask)
                seq_ko_pos = seq * (1.0 - act_mask)
                out_ko_neg = self.base([seq_ko_neg, z, s, p], training=True)
                out_ko_pos = self.base([seq_ko_pos, z, s, p], training=True)
                y_ko_neg = out_ko_neg[0]
                y_ko_pos = out_ko_pos[0]
            else:
                y_ko_neg = None
                y_ko_pos = None

            total_loss, expr_loss, mag_loss, pcc_loss, ko_loss, imp_loss = self._loss_components(
                y, outputs, p_guid=p, sign_in=s,
                act_mask=act_mask, rep_mask=rep_mask,
                y_ko_neg=y_ko_neg, y_ko_pos=y_ko_pos
            )

        grads = t.gradient(total_loss, self.base.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_weights))

        self.loss_tracker.update_state(expr_loss)
        self.total_tracker.update_state(total_loss)
        self.mag_tracker.update_state(mag_loss)
        self.pcc_tracker.update_state(pcc_loss)
        self.ko_tracker.update_state(ko_loss)
        self.imp_tracker.update_state(imp_loss)

        return {
            "loss": self.loss_tracker.result(),
            "total_loss": self.total_tracker.result(),
            "mag_loss": self.mag_tracker.result(),
            "pcc_loss": self.pcc_tracker.result(),
            "ko_loss": self.ko_tracker.result(),
            "imp_loss": self.imp_tracker.result(),
        }

    def test_step(self, data):
        (seq, z, s, p, act_mask, rep_mask), y = data
        outputs = self.base([seq, z, s, p], training=False)
        y_hat, S, att_cluster, base_imp, alpha = outputs

        expr_loss = tf.reduce_mean(tf.keras.losses.huber(y, y_hat))
        pcc_loss = 1.0 - batch_pcc(y, y_hat)
        total_loss = expr_loss + self.lp * pcc_loss

        self.loss_tracker.update_state(expr_loss)
        self.total_tracker.update_state(total_loss)
        self.mag_tracker.update_state(tf.constant(0.0, tf.float32))
        self.pcc_tracker.update_state(pcc_loss)
        self.ko_tracker.update_state(tf.constant(0.0, tf.float32))
        self.imp_tracker.update_state(tf.constant(0.0, tf.float32))

        return {
            "loss": self.loss_tracker.result(),
            "total_loss": self.total_tracker.result(),
            "mag_loss": self.mag_tracker.result(),
            "pcc_loss": self.pcc_tracker.result(),
            "ko_loss": self.ko_tracker.result(),
            "imp_loss": self.imp_tracker.result(),
        }


# ============================
# Eval & plots
# ============================
def compute_metrics_per_tissue(y_true, y_pred, tissue_names):
    rows = []
    for i, t in enumerate(tissue_names):
        yt, yp = y_true[:, i], y_pred[:, i]
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            PCC = np.nan
            SPR = np.nan
            R2 = np.nan
            RMSE = np.nan
        else:
            PCC = pearsonr(yt, yp)[0]
            SPR = spearmanr(yt, yp)[0]
            R2 = r2_score(yt, yp)
            RMSE = np.sqrt(mean_squared_error(yt, yp))
        rows.append(dict(tissue=t, PCC=PCC, Spearman=SPR, R2=R2, RMSE=RMSE))
    df = pd.DataFrame(rows)
    macro = dict(
        tissue="__macro__",
        PCC=np.nanmean(df["PCC"]),
        Spearman=np.nanmean(df["Spearman"]),
        R2=np.nanmean(df["R2"]),
        RMSE=np.nanmean(df["RMSE"]),
    )
    return df, macro


def plot_loss(history, out_pdf, detailed_pdf=None):
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="Train expr_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val expr_loss")
    plt.xlabel("Epoch")
    plt.ylabel("expr_loss (Huber)")
    plt.title("Training / Validation expr_loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

    if detailed_pdf:
        plt.figure(figsize=(7, 4))
        if "total_loss" in history:
            plt.plot(history["total_loss"], label="Train total_loss")
        if "val_total_loss" in history:
            plt.plot(history["val_total_loss"], label="Val total_loss")
        for k in ["mag_loss", "pcc_loss", "ko_loss", "imp_loss"]:
            if k in history:
                plt.plot(history[k], label=f"Train {k}")
            vk = "val_" + k
            if vk in history:
                plt.plot(history[vk], "--", label=f"Val {k}")
        plt.xlabel("Epoch")
        plt.ylabel("Component Loss")
        plt.title("Loss components (train / val)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(detailed_pdf)
        plt.close()


def plot_test_heatmap(test_df, out_pdf):
    df = test_df[test_df["tissue"] != "__macro__"].copy().set_index("tissue")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, metric in enumerate(["PCC", "R2"]):
        im = axes[i].imshow(df[metric][None, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[i].set_xticks(range(len(df.index)))
        axes[i].set_xticklabels(df.index, rotation=90, fontsize=8)
        axes[i].set_yticks([])
        axes[i].set_title(metric)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.suptitle("Test Set Performance per Tissue", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_pdf)
    plt.close()


def plot_pcc_bar(test_df, out_pdf):
    df = test_df[test_df["tissue"] != "__macro__"].copy()
    df = df.sort_values("PCC", ascending=False)
    plt.figure(figsize=(10, 4))
    plt.bar(df["tissue"], df["PCC"], alpha=0.85)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("PCC")
    plt.title("Test set PCC per tissue (sorted)")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


# ============================
# Explain exports
# ============================
def export_S_csv_and_heatmap(S_all, gene_ids, outdir, page_rows=300):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(S_all, columns=[f"Cluster{c+1}" for c in range(4)])
    df.insert(0, "gene_id", gene_ids)
    csv_path = os.path.join(outdir, "explain_signed_contrib_S.tsv")
    df.to_csv(csv_path, sep="\t", index=False)

    from matplotlib.backends import backend_pdf
    pdf_path = os.path.join(outdir, "explain_signed_contrib_S_heatmap.pdf")
    n = len(df)
    pages = max(1, int(np.ceil(n / page_rows)))
    with backend_pdf.PdfPages(pdf_path) as pdf:
        for p in range(pages):
            sl = slice(p * page_rows, min((p + 1) * page_rows, n))
            sub = df.iloc[sl, 1:].copy()
            fig = plt.figure(figsize=(4, max(2, sub.shape[0] * 0.12)))
            plt.imshow(sub.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.yticks(range(sub.shape[0]), df.iloc[sl, 0].values, fontsize=6)
            plt.xticks(range(4), sub.columns, fontsize=8, rotation=0)
            plt.title(f"SignedContribution S (rows {sl.start}–{sl.stop-1})")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f" Exported S CSV: {csv_path}")
    print(f"  Exported S heatmap PDF: {pdf_path}")


def export_att_csv(att_all, peak_ids_stack, gene_ids, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    N = att_all.shape[0]
    for i in range(N):
        gid = gene_ids[i]
        for c in range(4):
            for k in range(TOPK_PER_CLUSTER):
                pk = peak_ids_stack[i, c, k]
                wt = float(att_all[i, c, k])
                if (pk == "" or pd.isna(pk)) and wt <= 0:
                    continue
                rows.append((gid, f"Cluster{c+1}", k + 1, pk, wt))
    df = pd.DataFrame(rows, columns=["gene_id", "cluster", "rank", "peak_id", "att_weight"])
    outp = os.path.join(outdir, "explain_topk_peak_attention.tsv")
    df.to_csv(outp, sep="\t", index=False)
    print(f" Exported TopK peak attention CSV: {outp}")


def export_base_importance_and_masks(base_imp_all, act_mask_all, rep_mask_all, gene_ids, outdir):
    os.makedirs(outdir, exist_ok=True)
    base_imp = base_imp_all[..., 0]
    act = act_mask_all[..., 0]
    rep = rep_mask_all[..., 0]
    np.savez_compressed(
        os.path.join(outdir, "base_importance_and_masks_raw.npz"),
        gene_ids=gene_ids,
        base_imp=base_imp,
        act_mask=act,
        rep_mask=rep,
    )
    print(" Exported raw base importance & masks to npz.")


# ============================
# Split
# ============================
def split_indices_chr10_fixed(chroms_all, seed=42, val_ratio=0.12):
    idx_all = np.arange(len(chroms_all))
    mask_test = (chroms_all == "Chr12")
    mask_chr10 = (chroms_all == "Chr10")
    mask_val_candidates = (~mask_test) & (~mask_chr10)

    cand_idx = np.where(mask_val_candidates)[0]
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(val_ratio * len(cand_idx))))
    val_idx = rng.choice(cand_idx, size=n_val, replace=False)

    is_val = np.zeros_like(idx_all, dtype=bool)
    is_val[val_idx] = True

    is_test = mask_test
    is_train = ~is_test & ~is_val

    tr_idx = np.where(is_train)[0]
    te_idx = np.where(is_test)[0]
    va_idx = np.where(is_val)[0]

    print(" Dataset split (Chr10 fixed as train):")
    print(f"   Train: {len(tr_idx)} genes")
    print(f"   Val  : {len(va_idx)} genes (random 12% )")
    print(f"   Test : {len(te_idx)} genes (Chr12)")
    return tr_idx, va_idx, te_idx


# ============================
# Main
# ============================
def main(args):
    setup_runtime(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    data = prepare_samples(
        args.fasta,
        args.gff,
        args.expr,
        args.z_peak,
        args.pcc,
        args.cluster_map,
        args.limit_genes,
        args.threads,
    )
    X_seq = data["X_seq"]
    X_z = data["X_z"]
    X_sign = data["X_sign"]
    X_pcc = data["X_pcc"]
    X_act = data["X_act_mask"]
    X_rep = data["X_rep_mask"]
    Y = data["Y"]
    tissues = data["tissues"]
    z_dim = data["z_dim"]
    gene_ids_all = data["gene_ids"]
    chroms_all = data["chroms"]
    peak_ids_stack_all = data["peak_ids_stack"]
    n_tissues = len(tissues)

    tr_idx, val_idx, te_idx = split_indices_chr10_fixed(chroms_all, seed=args.seed, val_ratio=0.12)

    def make_ds(ix, shuffle=False):
        d = tf.data.Dataset.from_tensor_slices((
            (X_seq[ix], X_z[ix], X_sign[ix], X_pcc[ix], X_act[ix], X_rep[ix]),
            Y[ix],
        ))
        if shuffle:
            d = d.shuffle(len(ix), reshuffle_each_iteration=True)
        d = d.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        return d

    ds_tr = make_ds(tr_idx, shuffle=True)
    ds_val = make_ds(val_idx, shuffle=False)
    ds_te = make_ds(te_idx, shuffle=False)

    base = build_stage2pp_v4_2(
        SEQ_LEN_EXPECTED, z_dim, n_tissues,
        d_seq=args.d_seq, d_pool=args.d_pool, d_head=args.d_head
    )

    steps_per_epoch = max(1, int(np.ceil(len(tr_idx) / args.batch_size)))
    total_steps = steps_per_epoch * args.epochs
    warm_steps = max(10, int(0.05 * total_steps))

    lr_sched = WarmUpCosine(args.lr, warm_steps, total_steps)
    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr_sched,
        weight_decay=1e-4,
        clipnorm=1.0,
    )

    model = Trainer(base, lm=args.lambda_mag, lp=args.lambda_pcc, lk=args.lambda_ko, li=args.lambda_imp)
    model.compile(optimizer=opt)

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
        )
    ]

    print(" Start Stage2++ v4.2 (StableVal, explain-sampling aligned)...")
    hist = model.fit(ds_tr, validation_data=ds_val, epochs=args.epochs, callbacks=cbs, verbose=1)

    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(args.outdir, "training_log.csv"), sep="\t", index=False)
    plot_loss(
        hist.history,
        os.path.join(args.outdir, "loss_curve_expr_only.pdf"),
        detailed_pdf=os.path.join(args.outdir, "loss_curve_components.pdf"),
    )

    base.save(os.path.join(args.outdir, "stage2pp_v4_2_stableval_explainSampling_model.keras"))

    if "val_loss" in hist.history:
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        best_val = float(np.min(hist.history["val_loss"]))
    else:
        best_epoch = -1
        best_val = float("nan")

    print(f" Model saved. Best epoch = {best_epoch}, val_expr_loss = {best_val:.4f}")

    y_true_all, y_pred_all = [], []
    S_all, att_all, base_imp_all = [], [], []
    act_mask_te, rep_mask_te = [], []

    for (seq, z, sg, pc, act_m, rep_m), y in ds_te:
        y_hat, S, att, bimp, alpha = base([seq, z, sg, pc], training=False)
        y_true_all.append(y.numpy())
        y_pred_all.append(y_hat.numpy())
        S_all.append(S.numpy())
        att_all.append(att.numpy())
        base_imp_all.append(bimp.numpy())
        act_mask_te.append(act_m.numpy())
        rep_mask_te.append(rep_m.numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    S_all = np.concatenate(S_all, axis=0)
    att_all = np.concatenate(att_all, axis=0)
    base_imp_all = np.concatenate(base_imp_all, axis=0)
    act_mask_te = np.concatenate(act_mask_te, axis=0)
    rep_mask_te = np.concatenate(rep_mask_te, axis=0)

    gene_ids_te = gene_ids_all[te_idx]
    peak_ids_stack_te = peak_ids_stack_all[te_idx]

    test_df, test_macro = compute_metrics_per_tissue(y_true_all, y_pred_all, tissues)
    test_df = pd.concat([test_df, pd.DataFrame([test_macro])], axis=0)
    test_df.to_csv(os.path.join(args.outdir, "test_metrics.tsv"), sep="\t", index=False)

    plot_pcc_bar(test_df, os.path.join(args.outdir, "test_PCC_bar_sorted.pdf"))
    plot_test_heatmap(test_df, os.path.join(args.outdir, "test_heatmap_R2_PCC.pdf"))

    explain_dir = os.path.join(args.outdir, "explain")
    export_S_csv_and_heatmap(S_all, gene_ids_te, explain_dir, page_rows=300)
    export_att_csv(att_all, peak_ids_stack_te, gene_ids_te, explain_dir)
    export_base_importance_and_masks(base_imp_all, act_mask_te, rep_mask_te, gene_ids_te, explain_dir)

    print("\n [Test Set Detailed Performance by Tissue]")
    df_display = test_df[test_df["tissue"] != "__macro__"].copy().sort_values("R2", ascending=False)
    for _, r in df_display.iterrows():
        print(
            f"  {r.tissue:<20s}  "
            f"R²={r.R2:6.3f}  PCC={r.PCC:6.3f}  "
            f"Spearman={r.Spearman:6.3f}  RMSE={r.RMSE:6.3f}"
        )

    print(
        f"\n Best tissue: {df_display.iloc[0]['tissue']} "
        f"(R²={df_display.iloc[0]['R2']:.3f}, PCC={df_display.iloc[0]['PCC']:.3f})"
    )
    print(
        f"Worst tissue: {df_display.iloc[-1]['tissue']} "
        f"(R²={df_display.iloc[-1]['R2']:.3f}, PCC={df_display.iloc[-1]['PCC']:.3f})"
    )

    print("\n Macro Test Summary:")
    print(f"    R²_mean       = {test_macro['R2']:.3f}")
    print(f"    PCC_mean      = {test_macro['PCC']:.3f}")
    print(f"    Spearman_mean = {test_macro['Spearman']:.3f}")
    print(f"    RMSE_mean     = {test_macro['RMSE']:.3f}")
    print(" Done. Outputs at:", args.outdir)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_genes": int(Y.shape[0]),
        "n_tissues": int(n_tissues),
        "d_seq": int(args.d_seq),
        "d_pool": int(args.d_pool),
        "d_head": int(args.d_head),
        "lambda_mag": float(args.lambda_mag),
        "lambda_pcc": float(args.lambda_pcc),
        "lambda_ko": float(args.lambda_ko),
        "lambda_imp": float(args.lambda_imp),
        "best_val_expr_loss": best_val,
        "macro_R2": float(test_macro["R2"]),
        "macro_PCC": float(test_macro["PCC"]),
        "macro_Spearman": float(test_macro["Spearman"]),
        "macro_RMSE": float(test_macro["RMSE"]),
        "split_rule": "Chr12=test; Chr10=train_only; others random 12% val",
        "model_name": "Stage2PP_v4_2_StableVal_explainSamplingAligned",
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Meta saved.")


# ============================
# CLI
# ============================
def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Stage2++ v4.2 StableVal — sampling aligned to A03_Explain_V1-14 "
            "(Chr12=test, Chr10 only in train, others random 12% for val)"
        )
    )
    p.add_argument("--fasta", required=True)
    p.add_argument("--gff", required=True)
    p.add_argument("--expr", required=True, help="TPM matrix, gene_id + tissues")
    p.add_argument("--pcc", required=True, help="PCC table: gene_id, peak_id, PCC, cluster/cluster_idx, chr, start, end")
    p.add_argument("--z_peak", required=True, help="Stage1 z_peak_embeddings.tsv")
    p.add_argument("--cluster_map", required=True, help="Two cols: tissue<TAB>cluster (Cluster1~4)")
    p.add_argument("--outdir", default="stage2pp_v4_2_stableval_explainSampling_out")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_mag", type=float, default=LAMBDA_MAG)
    p.add_argument("--lambda_pcc", type=float, default=LAMBDA_PCC)
    p.add_argument("--lambda_ko", type=float, default=LAMBDA_KO)
    p.add_argument("--lambda_imp", type=float, default=LAMBDA_IMP)
    p.add_argument("--d_seq", type=int, default=256)
    p.add_argument("--d_pool", type=int, default=128)
    p.add_argument("--d_head", type=int, default=128)
    p.add_argument("--limit_genes", type=int, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)

