#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import sys

########################################
# check parameter
########################################

if len(sys.argv) != 3:
    print("Usage: python predict_qc.py input.csv output.csv")
    sys.exit(1)

input_csv = sys.argv[1]
output_csv = sys.argv[2]

########################################
# load model
########################################

# GBM
gbm_load = joblib.load("./models/gbm_qc_pipeline.joblib")
gbm = gbm_load["model"]
label_encoders_gbm = gbm_load["label_encoders"]
y_encoder_gbm = gbm_load["y_encoder"]
features_gbm = gbm_load["features"]

# RF
rf_load = joblib.load("./models/rf_qc_pipeline.joblib")
rf = rf_load["model"]
label_encoders_rf = rf_load["label_encoders"]
y_encoder_rf = rf_load["y_encoder"]
features_rf = rf_load["features"]


# SVM
svm_load = joblib.load("./models/svm_qc_pipeline.joblib")
svm_model = svm_load["model"]
scaler_svm = svm_load["scaler"]
feature_columns_svm = svm_load["feature_columns"]
lb_svm = svm_load["label_binarizer"]


# NN
model = load_model("./models/nn_qc_model.keras")
nn_load = joblib.load("./models/nn_qc_preprocess.joblib")
scaler_nn = nn_load["scaler"]
label_encoder_nn = nn_load["label_encoder_qc"]
feature_columns_nn = nn_load["feature_columns"]

########################################
# preprocessing
########################################

def preprocess_gbm(df):
    df = df.copy()
    for col in ["CancerType", "OpeBx"]:
        df[col] = label_encoders_gbm[col].transform(df[col])
    #return df[["TN_HU", "CancerType", "OpeBx", "StorageTime"]]
    return df[features_gbm]

def preprocess_rf(df):
    df = df.copy()
    for col in ["CancerType", "OpeBx"]:
        df[col] = label_encoders_rf[col].transform(df[col])
    #return df[["TN_HU", "CancerType", "OpeBx", "StorageTime"]]
    return df[features_rf]


def preprocess_svm(df):
    X = pd.get_dummies(
        df[["TN_HU", "CancerType", "OpeBx", "StorageTime"]],
        columns=["CancerType", "OpeBx"],
        drop_first=True
    )
    X = X.reindex(columns=feature_columns_svm, fill_value=0)
    return scaler_svm.transform(X)


def preprocess_nn(df):
    X = pd.get_dummies(
        df[["TN_HU", "CancerType", "OpeBx", "StorageTime"]],
        columns=["CancerType", "OpeBx"],
        drop_first=True
    )
    X = X.reindex(columns=feature_columns_nn, fill_value=0)
    return scaler_nn.transform(X)

########################################
# predict
########################################

def predict_qc(df):
    results = {}

    # GBM
    X_gbm = preprocess_gbm(df)
    #results["GBM_prob"] = gbm.predict_proba(X_gbm)[:, 1]
    y_pred_gbm = gbm.predict(X_gbm)
    results["GBM_pred"] = y_encoder_gbm.inverse_transform(y_pred_gbm)

    # RF
    X_rf = preprocess_rf(df)
    #results["RF_prob"] = rf.predict_proba(X_rf)[:, 1]
    y_pred_rf = rf.predict(X_rf)
    results["RF_pred"] = y_encoder_rf.inverse_transform(y_pred_rf)


    # SVM
    X_svm = preprocess_svm(df)
    #results["SVM_prob"] = svm_model.predict_proba(X_svm)[:, 1]
    results["SVM_pred"] = svm_model.predict(X_svm)
    #results["SVM_pred"] = lb_svm.transform(y_pred_svm).ravel()


    # NN
    X_nn = preprocess_nn(df)
    y_prob_nn = model.predict(X_nn).ravel()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)
    y_label_nn = label_encoder_nn.inverse_transform(y_pred_nn)

    #results["NN_prob"] = y_prob_nn
    results["NN_pred"] = y_label_nn

    return pd.DataFrame(results)


########################################
# execution
########################################

def main():
    df = pd.read_csv(input_csv)

    required_cols = {"TN_HU", "CancerType", "OpeBx", "StorageTime"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pred = predict_qc(df)

    out = pd.concat([df, pred], axis=1)
    out.to_csv(output_csv, index=False)

    print(f"QC prediction finished: {output_csv}")

if __name__ == "__main__":
    main()
