#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A01 — Export Stage1 encoder (z_peak) from atac_mae_model_final.keras
--------------------------------------------------------------------

输入:
  --model   Stage1 训练好的 MAE 模型 (如 atac_mae_model_final.keras)
  --out     输出 encoder 模型文件名 (默认: stage1_encoder.keras)

输出:
  - 一个轻量级 Encoder：输入 22 维 ATAC → 输出 256 维 z_peak embedding
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers as L


# ============================================================================
#  与 Stage1 一致的自定义层（必须保留，否则 load_model 会失败）
# ============================================================================
class ChannelMask(L.Layer):
    """Stage1 的随机 mask 层（仅用于模型加载，导出时不会调用）"""
    def __init__(self, mask_rate=0.25, **kw):
        super().__init__(**kw)
        self.mask_rate = mask_rate

    def call(self, x, training=None):
        # 预测/导出模式返回全通道可见
        return x, tf.ones_like(x)


class MAEModel(tf.keras.Model):
    """
    为了让 Keras 能反序列化 Stage1 MAE。
    导出 encoder 不需要训练功能，因此 train_step/test_step 可以留空。
    """
    def train_step(self, data):
        return {}

    def test_step(self, data):
        return {}


# ============================================================================
#  主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Export Stage1 z_peak encoder")
    parser.add_argument("--model", required=True,
                        help="Stage1 MAE 模型 .keras 文件")
    parser.add_argument("--out", default="stage1_encoder.keras",
                        help="输出 Encoder 文件名")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"❌ 模型文件不存在: {args.model}")

    print(f"🔁 正在加载 Stage1 模型: {args.model}")

    # 必须注册 custom_objects，否则无法成功加载
    custom_objs = {
        "ChannelMask": ChannelMask,
        "MAEModel": MAEModel,
    }

    full_model = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objs
    )

    print("✅ Stage1 模型加载成功。")

    # ------------------------------------------------------------------------------
    # 获取 z_peak 层（名称必须是 Stage1 训练脚本中指定的 name='z_peak'）
    # ------------------------------------------------------------------------------
    try:
        z_layer = full_model.get_layer("z_peak")
    except ValueError:
        raise RuntimeError(
            "❌ 未找到 z_peak 层。\n"
            "请确认 Stage1 中是否定义了 Dense(..., name='z_peak')"
        )

    # ------------------------------------------------------------------------------
    # 构建 Encoder: 输入 = 22维 ATAC, 输出 = 256维 z_peak embedding
    # ------------------------------------------------------------------------------
    encoder = tf.keras.Model(
        inputs=full_model.input,
        outputs=z_layer.output,
        name="Stage1_z_peak_encoder"
    )

    encoder.save(args.out)

    print(f"🎯 编码器已保存至: {args.out}")
    print(f"📌 输入维度 : {encoder.input_shape}")
    print(f"📌 输出维度 : {encoder.output_shape}  ← 应为 (None, 256)")
    print("✨ 完成。")


if __name__ == "__main__":
    main()
