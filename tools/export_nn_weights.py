#!/usr/bin/env python3
"""
NNモデル（models/nn_qc_model.keras）の重みを numpy 形式（models/nn_qc_weights.npz）に書き出す。

Webアプリは TensorFlow を使わず、この npz から numpy で推論する。
モデルを作り直したときだけ、TensorFlow がある環境で再実行する：

    python tools/export_nn_weights.py

書き出し後、元モデル（Keras）と numpy 推論の出力が一致するかを確認する。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "nn_qc_model.keras"
DST = ROOT / "models" / "nn_qc_weights.npz"


def main():
    model = load_model(SRC)

    arrays = {}
    activations = []
    for layer in model.layers:
        if layer.__class__.__name__ == "Dense":
            kernel, bias = layer.get_weights()
            i = len(activations)
            arrays[f"W{i}"] = kernel
            arrays[f"b{i}"] = bias
            activations.append(layer.get_config()["activation"])
        elif layer.__class__.__name__ not in ("Dropout", "InputLayer"):
            # 推論時に何もしない層以外が入っていたら、numpy推論では再現できない
            raise ValueError(f"Unsupported layer: {layer.__class__.__name__}")
    arrays["activations"] = np.array(activations)
    np.savez(DST, **arrays)
    print(f"wrote {DST} ({len(activations)} dense layers: {activations})")

    # 検証：Keras と numpy 推論の比較
    import sys

    sys.path.insert(0, str(ROOT))
    import joblib

    from qc_core import INPUT_COLUMNS, CATEGORICAL_COLUMNS, NumpyMLP

    pre = joblib.load(ROOT / "models" / "nn_qc_preprocess.joblib")
    df = pd.read_csv(ROOT / "example" / "input_test.csv")
    X = pd.get_dummies(df[INPUT_COLUMNS], columns=CATEGORICAL_COLUMNS, drop_first=True)
    X = pre["scaler"].transform(X.reindex(columns=pre["feature_columns"], fill_value=0))
    p_keras = model.predict(X, verbose=0).ravel()
    p_numpy = NumpyMLP(DST).predict_proba(X)
    diff = np.abs(p_keras - p_numpy).max()
    print(f"max |keras - numpy| = {diff:.2e}")
    assert diff < 1e-5


if __name__ == "__main__":
    main()
