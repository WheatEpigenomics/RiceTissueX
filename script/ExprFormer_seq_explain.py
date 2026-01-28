#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

TSS_UP, TSS_DOWN = 2000, 1500
TTS_UP, TTS_DOWN = 500, 1500
SEQ_LEN_EXPECTED = (TSS_UP + TSS_DOWN) + (TTS_UP + TTS_DOWN)  # 5500
TOPK_PER_CLUSTER = 16

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {
    "A": "#1f77b4",  # Blue
    "C": "#2ca02c",  # Green
    "G": "#ff7f0e",  # Orange
    "T": "#d62728",  # Red
}

# ============================================================
# Custom Layers: Consistent with Stage2++ v4.2 training code
# ============================================================
from tensorflow.keras import layers as L

@tf.keras.saving.register_keras_serializable(package="stage2pp")
class ClusterPoolLayer(L.Layer):
    """Layer used in Stage2++ for cluster pooling from z_peak TopK embeddings."""
    def __init__(self, topk, z_dim, d_pool=128, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.z_dim = z_dim
        self.d_pool = d_pool

    def build(self, input_shape):
        # input: (B,4,TopK,z_dim)
        self.q = self.add_weight(
            "cluster_q",
            shape=(1, 4, 1, self.d_pool),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )
        # TimeDistributed(TimeDistributed(Dense))
        self.proj = L.TimeDistributed(
            L.TimeDistributed(L.Dense(self.d_pool, activation="gelu"))
        )

    def call(self, z_in):
        # z_in: (B,4,TopK,z_dim)
        h = self.proj(z_in)  # (B,4,TopK,d_pool)
        att = tf.nn.softmax(tf.reduce_sum(h * self.q, axis=-1), axis=2)  # (B,4,TopK)
        v = tf.reduce_sum(h * att[..., None], axis=2)  # (B,4,d_pool)
        return v, att

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "topk": self.topk,
            "z_dim": self.z_dim,
            "d_pool": self.d_pool,
        })
        return cfg


@tf.keras.saving.register_keras_serializable(package="stage2pp")
class SignedContribution(L.Layer):
    """Cluster signed contribution head in Stage2++."""
    def __init__(self, d_seq, **kwargs):
        super().__init__(**kwargs)
        self.d_seq = d_seq
        self.fc1 = L.Dense(256, activation="gelu")
        self.fc2 = L.Dense(1)
        self.concat = L.Concatenate()

    def call(self, v_c, g_seq, sign_in):
        # v_c:   (B,4,d_pool)
        # g_seq: (B,d_seq)
        # sign_in: (B,4)
        s_list = []
        for c in range(4):
            vc = v_c[:, c, :]
            h = self.concat([g_seq, vc])
            h = self.fc1(h)
            r = self.fc2(h)
            s = tf.tanh(r) * sign_in[:, c:c+1]
            s_list.append(s)
        return tf.concat(s_list, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_seq": self.d_seq})
        return cfg


@tf.keras.saving.register_keras_serializable(package="stage2pp")
class TissueMoEReadout(L.Layer):
    """Pure ATAC MoE readout head for Stage2++."""
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
        # tokens: (B,4,d_token)
        logits = tf.einsum("bcd,nd->bcn", tokens, self.E)
        alpha = tf.nn.softmax(logits, axis=1)   # (B,4,n)
        u = tf.einsum("bcd,bcn->bnd", tokens, alpha)  # (B,n,d)
        h = self.mlp1(u)
        y = self.mlp2(h)[..., 0] + self.bias[None, :]
        return y, alpha

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "n_tissues": self.n_tissues,
            "d_token": self.d_token,
            "d_hidden": self.d_hidden,
        })
        return cfg


# ============================================================
# Runtime Setup
# ============================================================
def setup_runtime(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("✅ Runtime ready (GPU memory growth enabled)")


# ============================================================
# Basic Utilities
# ============================================================
def load_fasta(path):
    d = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    d[name] = "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name:
        d[name] = "".join(buf).upper()
    return d


def revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def one_hot(seq):
    arr = np.zeros((len(seq), 4), np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq.upper()):
        if ch in m:
            arr[i, m[ch]] = 1.0
    return arr


def load_gff(gff, target_genes):
    rows = []
    tg = set(target_genes)
    with open(gff) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            arr = ln.strip().split("\t")
            if len(arr) < 9:
                continue
            if arr[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, info = arr
            gid = None
            for kv in info.split(";"):
                if kv.startswith("ID="):
                    gid = kv.split("=")[1].split(":")[-1]
                    break
            if gid and gid in tg:
                rows.append((gid, chrom, int(start), int(end), strand))
    if not rows:
        print("⚠️  No genes from gene_list matched in GFF")
    return pd.DataFrame(
        rows,
        columns=["gene_id", "chr", "start", "end", "strand"]
    )


# ============================================================
# Build input for a single gene (Sequence + ATAC TopK + sign/mag + coordinate mapping)
#  —— 4 fixed segments concatenated model (Unified for +/- strands)
# ============================================================
def build_inputs_for_gene(
    gene_id,
    tissues,
    expr_df,
    pcc_df,
    zdf,
    fasta_dict,
    gff_df,
    topk=TOPK_PER_CLUSTER,
):

    # ---------- Gene Coordinates ----------
    sub_gff = gff_df[gff_df["gene_id"] == gene_id]
    if len(sub_gff) == 0:
        raise ValueError(f"Gene not found in GFF: {gene_id}")
    row = sub_gff.iloc[0]
    chr_name = row["chr"]
    chr_seq = fasta_dict.get(chr_name)
    if chr_seq is None:
        raise ValueError(f"Chromosome {chr_name} not found in FASTA (gene {gene_id})")
    chr_len = len(chr_seq)

    strand = row["strand"]
    start1 = row["start"]  # GFF: 1-based, inclusive
    end1 = row["end"]      # GFF: 1-based, inclusive

    if strand == "+":
        TSS1 = start1
        TTS1 = end1
    else:
        # Negative strand: TSS at end, TTS at start
        TSS1 = end1
        TTS1 = start1

    # Convert to 0-based index (for slicing), 0-based = 1-based - 1
    TSS0 = TSS1 - 1
    TTS0 = TTS1 - 1

    if strand == "+":
        tss_up_start0   = TSS0 - TSS_UP
        tss_up_end0     = TSS0
        tss_down_start0 = TSS0
        tss_down_end0   = TSS0 + TSS_DOWN

        tts_up_start0   = TTS0 - TTS_UP
        tts_up_end0     = TTS0
        tts_down_start0 = TTS0
        tts_down_end0   = TTS0 + TTS_DOWN

        # Check bounds
        for s0, e0, name in [
            (tss_up_start0, tss_up_end0, "TSS_up"),
            (tss_down_start0, tss_down_end0, "TSS_down"),
            (tts_up_start0, tts_up_end0, "TTS_up"),
            (tts_down_start0, tts_down_end0, "TTS_down"),
        ]:
            if s0 < 0 or e0 > chr_len or s0 >= e0:
                raise ValueError(
                    f"Gene {gene_id} (+ strand) region {name} out of chromosome bounds. "
                    f"Check TSS_UP/TSS_DOWN/TTS_UP/TTS_DOWN parameters or try another gene."
                )

        # 4 segments order matches model sequence order (5'->3')
        seq_tss_up   = chr_seq[tss_up_start0:tss_up_end0]
        seq_tss_down = chr_seq[tss_down_start0:tss_down_end0]
        seq_tts_up   = chr_seq[tts_up_start0:tts_up_end0]
        seq_tts_down = chr_seq[tts_down_start0:tts_down_end0]

        pos_tss_up   = np.arange(tss_up_start0,   tss_up_end0,   dtype=np.int64) + 1
        pos_tss_down = np.arange(tss_down_start0, tss_down_end0, dtype=np.int64) + 1
        pos_tts_up   = np.arange(tts_up_start0,   tts_up_end0,   dtype=np.int64) + 1
        pos_tts_down = np.arange(tts_down_start0, tts_down_end0, dtype=np.int64) + 1

        seq_concat = seq_tss_up + seq_tss_down + seq_tts_up + seq_tts_down
        concat_pos1 = np.concatenate(
            [pos_tss_up, pos_tss_down, pos_tts_up, pos_tts_down],
            axis=0
        )

    else:
        # Negative strand: upstream is larger coordinate
        tss_up_start0   = TSS0 + 1
        tss_up_end0     = TSS0 + 1 + TSS_UP

        tss_down_start0 = TSS0 - TSS_DOWN
        tss_down_end0   = TSS0

        tts_up_start0   = TTS0
        tts_up_end0     = TTS0 + TTS_UP

        tts_down_start0 = TTS0 - TTS_DOWN
        tts_down_end0   = TTS0

        for s0, e0, name in [
            (tss_up_start0, tss_up_end0, "TSS_up"),
            (tss_down_start0, tss_down_end0, "TSS_down"),
            (tts_up_start0, tts_up_end0, "TTS_up"),
            (tts_down_start0, tts_down_end0, "TTS_down"),
        ]:
            if s0 < 0 or e0 > chr_len or s0 >= e0:
                raise ValueError(
                    f"Gene {gene_id} (- strand) region {name} out of chromosome bounds. "
                    f"Check TSS_UP/TSS_DOWN/TTS_UP/TTS_DOWN parameters or try another gene."
                )

        # Extract in genomic coordinates (small -> large) first
        seq_tss_up   = chr_seq[tss_up_start0:tss_up_end0]     # Length TSS_UP
        seq_tss_down = chr_seq[tss_down_start0:tss_down_end0] # Length TSS_DOWN
        seq_tts_up   = chr_seq[tts_up_start0:tts_up_end0]     # Length TTS_UP
        seq_tts_down = chr_seq[tts_down_start0:tts_down_end0] # Length TTS_DOWN

        pos_tss_up   = np.arange(tss_up_start0,   tss_up_end0,   dtype=np.int64) + 1
        pos_tss_down = np.arange(tss_down_start0, tss_down_end0, dtype=np.int64) + 1
        pos_tts_up   = np.arange(tts_up_start0,   tts_up_end0,   dtype=np.int64) + 1
        pos_tts_down = np.arange(tts_down_start0, tts_down_end0, dtype=np.int64) + 1

        # Raw concatenation order on genomic + strand
        seq_raw = seq_tts_down + seq_tts_up + seq_tss_down + seq_tss_up
        pos_raw = np.concatenate(
            [pos_tts_down, pos_tts_up, pos_tss_down, pos_tss_up],
            axis=0
        )

        # Reverse complement for negative strand model input
        seq_concat = revcomp(seq_raw)
        concat_pos1 = pos_raw[::-1].copy()


    if len(seq_concat) != SEQ_LEN_EXPECTED:
        raise ValueError(
            f"Gene {gene_id} concatenated sequence length {len(seq_concat)} != {SEQ_LEN_EXPECTED}. Check window settings."
        )
    if concat_pos1.shape[0] != SEQ_LEN_EXPECTED:
        raise ValueError(
            f"Gene {gene_id} concat_pos1 length {concat_pos1.shape[0]} != {SEQ_LEN_EXPECTED}"
        )

    # Region tags
    region = np.empty(SEQ_LEN_EXPECTED, dtype="<U10")
    region[:TSS_UP] = "TSS_up"
    region[TSS_UP:TSS_UP + TSS_DOWN] = "TSS_down"
    region[TSS_UP + TSS_DOWN:TSS_UP + TSS_DOWN + TTS_UP] = "TTS_up"
    region[TSS_UP + TSS_DOWN + TTS_UP:] = "TTS_down"

    seq_1hot = one_hot(seq_concat)

    # ---------- z_peak + PCC ----------
    z_cols = [c for c in zdf.columns if c != "peak_id"]
    z_dim = len(z_cols)
    z_stack = np.zeros((4, topk, z_dim), np.float32)
    sign_c = np.ones(4, np.float32)
    mag_c = np.zeros(4, np.float32)

    sub_pcc = pcc_df[pcc_df["gene_id"] == gene_id].copy()
    if len(sub_pcc) == 0:
        meta = {
            "chr": chr_name,
            "strand": strand,
            "concat_pos1": concat_pos1,
            "region": region,
        }
        return seq_1hot, z_stack, sign_c, mag_c, meta

    if "cluster_idx" not in sub_pcc.columns:
        if "cluster" in sub_pcc.columns:
            cmap = {"Cluster1": 0, "Cluster2": 1, "Cluster3": 2, "Cluster4": 3}
            sub_pcc["cluster_idx"] = sub_pcc["cluster"].map(cmap).astype(int)
        else:
            raise ValueError("PCC table requires 'cluster' or 'cluster_idx' column")

    sub_pcc["abs_pcc"] = sub_pcc["PCC"].abs()
    # Take TopK peaks for each cluster
    for c in range(4):
        sc = sub_pcc[sub_pcc["cluster_idx"] == c]
        if len(sc) == 0:
            continue
        sc = sc.sort_values("abs_pcc", ascending=False).head(topk)
        sign_c[c] = 1.0 if sc["PCC"].mean() >= 0 else -1.0
        mag_c[c] = sc["abs_pcc"].mean()

        for i, (_, r) in enumerate(sc.iterrows()):
            pk = r["peak_id"]
            zrow = zdf[zdf["peak_id"] == pk]
            if len(zrow) == 0:
                continue
            z_stack[c, i] = zrow[z_cols].values[0]

    if mag_c.max() > 0:
        mag_c /= (mag_c.max() + 1e-6)

    meta = {
        "chr": chr_name,
        "strand": strand,
        "concat_pos1": concat_pos1,
        "region": region,
    }
    return seq_1hot, z_stack, sign_c, mag_c, meta


# ============================================================
# Debug TSV Output
# ============================================================
def init_debug_tsv(path):

    if path is None:
        return
    d = os.path.dirname(path)
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                "gene_id\t"
                "tissue\t"
                "chr\t"
                "strand\t"
                "concat_index_0based\t"
                "in_window\t"
                "window_index_0based\t"
                "genome_pos_1based\t"
                "region\t"
                "base\n"
            )


def dump_debug_coords(
    debug_path,
    gene_id,
    tissue,
    meta,
    seq_1hot,
    window,
):

    if debug_path is None:
        return

    concat_pos1 = meta["concat_pos1"]
    region = meta["region"]
    chr_name = meta["chr"]
    strand = meta["strand"]
    L0, L1 = window

    L_total = seq_1hot.shape[0]
    assert len(concat_pos1) == L_total

    with open(debug_path, "a") as f:
        for idx in range(L_total):
            genome_pos = int(concat_pos1[idx])
            reg = region[idx]
            # one-hot -> base
            if seq_1hot[idx].sum() > 0:
                b_idx = int(np.argmax(seq_1hot[idx]))
                base = BASES[b_idx]
            else:
                base = "N"

            if L0 <= idx < L1:
                in_window = 1
                w_idx = idx - L0
            else:
                in_window = 0
                w_idx = -1

            f.write(
                f"{gene_id}\t"
                f"{tissue}\t"
                f"{chr_name}\t"
                f"{strand}\t"
                f"{idx}\t"
                f"{in_window}\t"
                f"{w_idx}\t"
                f"{genome_pos}\t"
                f"{reg}\t"
                f"{base}\n"
            )


# ============================================================
# NEW: per-base attribution Output (Full length / Window)
# ============================================================
def dump_per_base_attribution(
    out_tsv,
    gene_id,
    tissue,
    meta,
    seq_1hot,
    ig_vals,
    gxi_vals,
    window=None,
):

    chr_name = meta["chr"]
    strand = meta["strand"]
    concat_pos1 = meta["concat_pos1"]
    region = meta["region"]

    L = seq_1hot.shape[0]
    assert ig_vals.shape == (L, 4), f"IG shape mismatch: {ig_vals.shape}"
    assert gxi_vals.shape == (L, 4), f"GxI shape mismatch: {gxi_vals.shape}"

    if window is None:
        idx_range = range(L)
    else:
        L0, L1 = window
        idx_range = range(L0, L1)

    with open(out_tsv, "w") as f:
        f.write(
            "gene_id\t"
            "tissue\t"
            "chr\t"
            "strand\t"
            "concat_index\t"
            "genome_pos\t"
            "region\t"
            "base\t"
            "IG_signed\t"
            "IG_pos\t"
            "IG_neg\t"
            "GradXInput_signed\t"
            "GradXInput_pos\t"
            "GradXInput_neg\n"
        )

        for i in idx_range:
            if seq_1hot[i].sum() == 0:
                continue

            b = int(np.argmax(seq_1hot[i]))
            base = BASES[b]

            ig = float(ig_vals[i, b])
            gxi = float(gxi_vals[i, b])

            f.write(
                f"{gene_id}\t"
                f"{tissue}\t"
                f"{chr_name}\t"
                f"{strand}\t"
                f"{i}\t"
                f"{int(concat_pos1[i])}\t"
                f"{region[i]}\t"
                f"{base}\t"
                f"{ig:.6e}\t"
                f"{max(ig, 0.0):.6e}\t"
                f"{min(ig, 0.0):.6e}\t"
                f"{gxi:.6e}\t"
                f"{max(gxi, 0.0):.6e}\t"
                f"{min(gxi, 0.0):.6e}\n"
            )


# ============================================================
# Load Model (With Custom Layers & Lambda Safe)
# ============================================================
def load_stage2_model(model_path):
    print(f"📦 Loading model: {model_path}")
    custom_objects = {
        "ClusterPoolLayer": ClusterPoolLayer,
        "SignedContribution": SignedContribution,
        "TissueMoEReadout": TissueMoEReadout,
    }
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects=custom_objects,
    )
    print("✅ Model loaded.")
    return model


# ============================================================
# Integrated Gradients (Vectorized)
# ============================================================
def integrated_gradients_seq(model, seq_1hot, z, sign_c, mag_c, t_idx, steps=64):
    """
    IG on seq_1hot: baseline=0;
    Constructs (steps, L, 4) at once, batched forward + gradients.
    """
    seq_tf = tf.convert_to_tensor(seq_1hot, dtype=tf.float32)  # (L,4)
    baseline = tf.zeros_like(seq_tf)                           # (L,4)

    alphas = tf.linspace(0.0, 1.0, steps)[:, None, None]
    seqs = baseline[None, :, :] + alphas * (seq_tf[None, :, :] - baseline[None, :, :])
    # seqs: (steps, L, 4)

    # Tile other inputs
    z_tf = tf.convert_to_tensor(z[None, ...], dtype=tf.float32)
    z_tf = tf.repeat(z_tf, repeats=steps, axis=0)
    s_tf = tf.convert_to_tensor(sign_c[None, :], dtype=tf.float32)
    s_tf = tf.repeat(s_tf, repeats=steps, axis=0)
    p_tf = tf.convert_to_tensor(mag_c[None, :], dtype=tf.float32)
    p_tf = tf.repeat(p_tf, repeats=steps, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(seqs)
        pred, _, _, _, _ = model([seqs, z_tf, s_tf, p_tf], training=False)
        y = pred[:, t_idx]

    grads = tape.gradient(y, seqs)     # (steps,L,4)
    avg_grads = tf.reduce_mean(grads, axis=0)  # (L,4)

    ig = (seq_tf - baseline) * avg_grads
    return ig.numpy().astype(np.float32)


def grad_x_input_seq(model, seq_1hot, z, sign_c, mag_c, t_idx):
    """Grad×Input: Single forward + gradient."""
    seq_tf = tf.convert_to_tensor(seq_1hot[None, ...], dtype=tf.float32)
    z_tf = tf.convert_to_tensor(z[None, ...], dtype=tf.float32)
    s_tf = tf.convert_to_tensor(sign_c[None, :], dtype=tf.float32)
    p_tf = tf.convert_to_tensor(mag_c[None, :], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(seq_tf)
        pred, _, _, _, _ = model([seq_tf, z_tf, s_tf, p_tf], training=False)
        y = pred[:, t_idx]
    grads = tape.gradient(y, seq_tf)[0].numpy()  # (L,4)
    gxi = grads * seq_1hot
    return gxi.astype(np.float32)


# ============================================================
# Saturation Mutagenesis: Window only, Vectorized Batch Inference
# ============================================================
def saturation_mutagenesis_window(
    model,
    seq_1hot,
    z,
    sign_c,
    mag_c,
    t_idx,
    window,
    batch_size=512,
):
    """
    SM only in window=[L0,L1):
      - Single point mutation for each pos in window, each base b ∈ {A,C,G,T}
      - Constructs large batch at once, run in batches on GPU
    Returns: delta_scores, shape = (4, L_win), values = Δy = y(mut) - y(ref)
    """
    L0, L1 = window
    L_total = seq_1hot.shape[0]
    if not (0 <= L0 < L1 <= L_total):
        raise ValueError(f"window [{L0},{L1}) out of sequence range (0,{L_total})")

    L_win = L1 - L0
    num_mut = L_win * 4

    # Reference prediction
    ref_tf = tf.convert_to_tensor(seq_1hot[None, ...], dtype=tf.float32)
    z_tf_ref = tf.convert_to_tensor(z[None, ...], dtype=tf.float32)
    s_tf_ref = tf.convert_to_tensor(sign_c[None, :], dtype=tf.float32)
    p_tf_ref = tf.convert_to_tensor(mag_c[None, :], dtype=tf.float32)
    ref_pred, _, _, _, _ = model([ref_tf, z_tf_ref, s_tf_ref, p_tf_ref], training=False)
    ref_pred = ref_pred[0, t_idx].numpy().astype(np.float32)

    # Construct all mutated sequences
    base_seq = seq_1hot
    mutated_seqs = np.repeat(base_seq[None, ...], num_mut, axis=0)  # (N,L,4)

    for i_pos in range(L_win):
        gpos = L0 + i_pos
        for b in range(4):
            idx = i_pos * 4 + b
            mutated_seqs[idx, gpos, :] = 0.0
            mutated_seqs[idx, gpos, b] = 1.0

    # Batch prediction
    y_mut = np.zeros(num_mut, np.float32)
    start = 0
    while start < num_mut:
        end = min(start + batch_size, num_mut)
        m_batch = tf.convert_to_tensor(mutated_seqs[start:end], dtype=tf.float32)

        # Repeat z/sign/mag
        z_tf = tf.convert_to_tensor(z[None, ...], dtype=tf.float32)
        z_tf = tf.repeat(z_tf, repeats=(end - start), axis=0)
        s_tf = tf.convert_to_tensor(sign_c[None, :], dtype=tf.float32)
        s_tf = tf.repeat(s_tf, repeats=(end - start), axis=0)
        p_tf = tf.convert_to_tensor(mag_c[None, :], dtype=tf.float32)
        p_tf = tf.repeat(p_tf, repeats=(end - start), axis=0)

        preds_batch, _, _, _, _ = model([m_batch, z_tf, s_tf, p_tf], training=False)
        preds_batch = preds_batch[:, t_idx].numpy().astype(np.float32)
        y_mut[start:end] = preds_batch
        start = end

    scores = y_mut.reshape(L_win, 4).T  # (4,L_win)
    delta = scores - ref_pred  # (4,L_win)
    return delta


# ============================================================
# Basenji style single-base loss/gain logo
# ============================================================
def compute_loss_gain_from_sm(sm_sub):
    """
    sm_sub: (4,L_win) = Δy (mut - ref)

    Returns:
      loss_scores: (L_win,)  = -min(Δy_b)  >= 0
      gain_scores: (L_win,)  =  max(Δy_b)  >= 0
    """
    delta_t = sm_sub.T  # (L,4)
    loss = -np.min(delta_t, axis=1)  # Larger means "more disruption"
    gain = np.max(delta_t, axis=1)   # Larger means "more enhancement"
    loss[loss < 0] = 0.0
    gain[gain < 0] = 0.0
    return loss, gain


def plot_single_base_logo(
    ax,
    seq_sub,
    scores,
    title,
    max_height=1.5,
    fontsize=8,
):
    """
    Closer to Basenji logo:

      - Only draw the one base present in the reference sequence at each position
      - Use bar + top letter for height, fixed bar width, fixed letter size, height varies
      - scores: (L_win,) already >=0, rescaled to [0, max_height] internally
    """
    L_win = seq_sub.shape[0]
    if scores.shape[0] != L_win:
        raise ValueError("seq_sub and scores length mismatch")

    max_score = float(np.max(scores)) + 1e-8
    x_coords = np.arange(L_win) + 0.5

    for i in range(L_win):
        base_idx = int(np.argmax(seq_sub[i]))
        base = BASES[base_idx]
        score = scores[i]
        if score <= 0:
            continue

        h = (score / max_score) * max_height
        color = BASE_COLORS[base]

        # Vertical bar (fixed width)
        ax.vlines(x_coords[i], 0.0, h, color=color, linewidth=1.0)

        # Top letter (fixed font size)
        ax.text(
            x_coords[i],
            h,
            base,
            ha="center",
            va="bottom",
            color=color,
            fontsize=fontsize,
            fontweight="bold",
        )

    ax.set_xlim(0, L_win)
    ax.set_ylim(0, max_height * 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Base importance\n(Basenji-style)")
    ax.set_title(title, fontsize=9)


# ============================================================
# Plotting: One PDF, Two Pages (IG / Grad×Input)
# ============================================================
def plot_explain_two_pages(
    gene_id,
    tissue,
    seq_1hot,
    ig_vals,
    gxi_vals,
    sm_window_vals,
    window,
    out_pdf,
):
    L0, L1 = window
    L_total = seq_1hot.shape[0]
    if not (0 <= L0 < L1 <= L_total):
        raise ValueError(f"window [{L0},{L1}) out of sequence range(0,{L_total})")

    x = np.arange(L0, L1)
    seq_sub = seq_1hot[L0:L1]              # (L_win,4)
    ig_sub = ig_vals[L0:L1]                # (L_win,4)
    gxi_sub = gxi_vals[L0:L1]              # (L_win,4)
    sm_sub = sm_window_vals                # (4,L_win)

    def _summarize(pos_imp):
        pos = np.sum(np.maximum(pos_imp, 0.0), axis=1)
        neg = np.sum(np.minimum(pos_imp, 0.0), axis=1)
        return pos, neg

    ig_pos, ig_neg = _summarize(ig_sub)
    gxi_pos, gxi_neg = _summarize(gxi_sub)

    # Build loss / gain scores from SM
    loss_scores, gain_scores = compute_loss_gain_from_sm(sm_sub)

    with PdfPages(out_pdf) as pdf:
        # ---------------- Page 1: IG + loss logo ----------------
        fig1 = plt.figure(figsize=(12, 9))
        gs1 = fig1.add_gridspec(4, 1, height_ratios=[2, 2, 0.2, 2])

        ax1 = fig1.add_subplot(gs1[0, 0])
        ax1.plot(x, np.abs(ig_pos), color="red", label="IG (+)")
        ax1.plot(x, np.abs(ig_neg), color="blue", label="IG (-)")
        ax1.set_ylabel("Contribution (|IG|)")
        ax1.set_title(f"{gene_id} — {tissue} — Integrated Gradients  [{L0},{L1})")
        ax1.legend(loc="upper right", fontsize=9)

        ax2 = fig1.add_subplot(gs1[1, 0])
        vmax = np.max(np.abs(sm_sub)) + 1e-6
        im = ax2.imshow(
            sm_sub,
            cmap="bwr",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            extent=[L0, L1, -0.5, 3.5],
            origin="lower",
        )
        ax2.set_yticks(range(4))
        ax2.set_yticklabels(BASES)
        ax2.set_ylabel("Saturation\nmutagenesis")
        plt.colorbar(im, ax=ax2, fraction=0.025)

        ax_dummy = fig1.add_subplot(gs1[2, 0])
        ax_dummy.axis("off")

        ax3 = fig1.add_subplot(gs1[3, 0])
        plot_single_base_logo(
            ax3,
            seq_sub,
            loss_scores,
            title="Basenji-style loss logo (from SM, negative Δy)",
            max_height=1.5,
            fontsize=7,
        )

        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # ---------------- Page 2: Grad×Input + gain logo ----------------
        fig2 = plt.figure(figsize=(12, 9))
        gs2 = fig2.add_gridspec(4, 1, height_ratios=[2, 2, 0.2, 2])

        bx1 = fig2.add_subplot(gs2[0, 0])
        bx1.plot(x, np.abs(gxi_pos), color="red", label="Grad×Input (+)")
        bx1.plot(x, np.abs(gxi_neg), color="blue", label="Grad×Input (-)")
        bx1.set_ylabel("Contribution (|Grad×Input|)")
        bx1.set_title(f"{gene_id} — {tissue} — Grad×Input  [{L0},{L1})")
        bx1.legend(loc="upper right", fontsize=9)

        bx2 = fig2.add_subplot(gs2[1, 0])
        im2 = bx2.imshow(
            sm_sub,
            cmap="bwr",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            extent=[L0, L1, -0.5, 3.5],
            origin="lower",
        )
        bx2.set_yticks(range(4))
        bx2.set_yticklabels(BASES)
        bx2.set_ylabel("Saturation\nmutagenesis")
        plt.colorbar(im2, ax=bx2, fraction=0.025)

        bx_dummy = fig2.add_subplot(gs2[2, 0])
        bx_dummy.axis("off")

        bx3 = fig2.add_subplot(gs2[3, 0])
        plot_single_base_logo(
            bx3,
            seq_sub,
            gain_scores,
            title="Basenji-style gain logo (from SM, positive Δy)",
            max_height=1.5,
            fontsize=7,
        )

        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)


# ============================================================
# CLI & Main Process
# ============================================================
def build_parser():
    ap = argparse.ArgumentParser(
        description=(
            "Stage2++ v4.2 StableVal Sequence Interpretation Script "
            "(IG + Grad×Input + SM-window + Basenji-style loss/gain logo "
            "+ Full-length coord debug + NEW: per-base attribution TSV)"
        )
    )
    ap.add_argument("--model", required=True, help="stage2pp_v4_2_stableval_model.keras")
    ap.add_argument("--fasta", required=True, help="Genome FASTA")
    ap.add_argument("--gff", required=True, help="GFF3 with gene coordinates")
    ap.add_argument("--expr", required=True, help="TPM Matrix, first column gene_id, rest tissues")
    ap.add_argument("--pcc", required=True, help="PCC table, must contain gene_id, peak_id, PCC, cluster/cluster_idx")
    ap.add_argument("--z_peak", required=True, help="z_peak_embeddings.tsv")
    ap.add_argument("--gene_list", required=True, help="List of gene_ids to explain, one per line")
    ap.add_argument(
        "--tissues",
        required=True,
        help="Comma-separated tissue names, e.g., Leaf_4_5,YP2",
    )
    ap.add_argument(
        "--windows",
        required=True,
        help="Interpretation sequence window, e.g., 4600:4800",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--ig_steps",
        type=int,
        default=64,
        help="IG steps, default 64",
    )
    ap.add_argument(
        "--sm_batch",
        type=int,
        default=512,
        help="SM large batch size, default 512",
    )
    ap.add_argument(
        "--debug_coords_tsv",
        default=None,
        help="Path to output full 5500bp coord mapping TSV if needed (e.g. out/coords_debug.tsv)",
    )
    ap.add_argument(
        "--dump_attr_tsv",
        action="store_true",
        help="Whether to output per-base attribution TSV (full + window). Default off; add flag to enable.",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(args):
    setup_runtime(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Init debug TSV (if needed)
    init_debug_tsv(args.debug_coords_tsv)

    # ---------- Load Expression Matrix ----------
    expr_df = pd.read_csv(args.expr, sep="\t")
    if "gene_id" not in expr_df.columns:
        raise ValueError("expr TSV must contain 'gene_id' column")
    tissues_all = [c for c in expr_df.columns if c != "gene_id"]

    # Check if tissues exist in expr
    req_tissues = [t.strip() for t in args.tissues.split(",") if t.strip()]
    for t in req_tissues:
        if t not in tissues_all:
            raise ValueError(f"Tissue {t} not found in expression matrix columns")

    # ---------- z_peak / PCC ----------
    zdf = pd.read_csv(args.z_peak, sep="\t")
    pcc_df = pd.read_csv(args.pcc, sep="\t")

    # ---------- Gene List & GFF & FASTA ----------
    gene_list = [x.strip() for x in open(args.gene_list) if x.strip()]
    print(f"🧬 Preparing to explain {len(gene_list)} genes.")

    fasta_dict = load_fasta(args.fasta)
    gff_df = load_gff(args.gff, gene_list)
    print(f"🧬 Found coordinates for {len(gff_df)} genes in GFF.")

    # ---------- Model ----------
    model = load_stage2_model(args.model)

    # ---------- Window ----------
    L0_str, L1_str = args.windows.split(":")
    L0, L1 = int(L0_str), int(L1_str)
    window = (L0, L1)
    print(f"📏 Using window [{L0}, {L1}) for SM and local plotting")

    # ---------- Main Loop ----------
    for gid in tqdm(gene_list, desc="Explain genes"):
        try:
            seq_1hot, z_stack, sign_c, mag_c, meta = build_inputs_for_gene(
                gid,
                tissues_all,
                expr_df,
                pcc_df,
                zdf,
                fasta_dict,
                gff_df,
            )
        except Exception as e:
            print(f"⚠️  Skipping {gid}: {e}")
            continue

        first_tissue_for_debug = True

        for tissue in req_tissues:
            t_idx = tissues_all.index(tissue)

            # Dump coord debug for gene only once on first tissue (avoid duplication)
            if first_tissue_for_debug and args.debug_coords_tsv is not None:
                dump_debug_coords(
                    args.debug_coords_tsv,
                    gid,
                    tissue,
                    meta,
                    seq_1hot,
                    window,
                )
                first_tissue_for_debug = False

            # IG
            ig_vals = integrated_gradients_seq(
                model, seq_1hot, z_stack, sign_c, mag_c,
                t_idx, steps=args.ig_steps
            )
            # Grad×Input
            gxi_vals = grad_x_input_seq(
                model, seq_1hot, z_stack, sign_c, mag_c,
                t_idx
            )

            # NEW: Output per-base attribution TSV (Full + Window)
            if args.dump_attr_tsv:
                out_attr_full = os.path.join(
                    args.outdir,
                    f"{gid}_{tissue}_V1-15_attr_full.tsv"
                )
                dump_per_base_attribution(
                    out_attr_full,
                    gid,
                    tissue,
                    meta,
                    seq_1hot,
                    ig_vals,
                    gxi_vals,
                    window=None,
                )

                out_attr_win = os.path.join(
                    args.outdir,
                    f"{gid}_{tissue}_V1-15_attr_window_{L0}_{L1}.tsv"
                )
                dump_per_base_attribution(
                    out_attr_win,
                    gid,
                    tissue,
                    meta,
                    seq_1hot,
                    ig_vals,
                    gxi_vals,
                    window=window,
                )

            # Saturation Mutagenesis (Window only)
            sm_vals_window = saturation_mutagenesis_window(
                model,
                seq_1hot,
                z_stack,
                sign_c,
                mag_c,
                t_idx,
                window=window,
                batch_size=args.sm_batch,
            )

            out_pdf = os.path.join(
                args.outdir,
                f"{gid}_{tissue}_V1-15_explain_seq_logo.pdf"
            )
            plot_explain_two_pages(
                gid,
                tissue,
                seq_1hot,
                ig_vals,
                gxi_vals,
                sm_vals_window,
                window,
                out_pdf,
            )

    print("✅ All done.")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
