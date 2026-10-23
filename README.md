# QC Prediction Using Machine Learning Models

This repository provides a Python script for predicting QC (Quality Control) status from tabular data using multiple pre-trained machine learning models.

Four models are supported and executed in parallel:

- Gradient Boosting Machine (GBM)
- Random Forest (RF)
- Support Vector Machine (SVM)
- Neural Network (TensorFlow / Keras)

Each model outputs the **predicted QC status**.

---

## Features

- Load previously trained and saved models
- CSV file as input
- Apply the same preprocessing used during training
- Predict QC labels and probabilities for each model
- Executable as a standalone Python script

---

> Model and file names may be adjusted to match your environment.

---

## Software Environment
- pandas        2.3.2
- numpy         2.1.3
- scikit-learn  1.7.2 (the saved models require this exact version)
- joblib        1.4.2
- tensorflow    2.19.0

## Input Data Format

Input data must contain the following columns:

| Column Name   | Description                         |
|---------------|-------------------------------------|
| TN_HU         | Tumor CT value                      |
| CancerType    | Cancer type (categorical)           |
| OpeBx         | Operation or biopsy (categorical)   |
| StorageTime   | Storage time                        |

Example (`sample_input.csv`):

```csv
TN_HU,CancerType,OpeBx,StorageTime
35,Pancreas,Bx,12
| TN_HU | CancerType | OpeBx | StorageTime |
|-------|------------|-------|-------------|
|  15   |  Pancreas  |   Bx  |     550     |
```

## Allowed Values
**TN_HU**
Percent tumor nuclei measured by pathologist

**CancerType**
* Adrenal_Gland
* Ampulla_of_Vater
* Biliary_Tract
* Bladder_Urinary_Tract
* Bone
* Bowel
* Breast
* Cervix
* CNS_Brain
* Esophagus_Stomach
* Eye
* Head_and_Neck
* Kidney
* Liver
* Lung
* Other
* Ovary_Fallopian_Tube
* Pancreas
* Penis
* Peripheral_Nervous_System
* Peritoneum
* Pleura
* Prostate
* Skin
* Soft_Tissue
* Testis
* Thymus
* Thyroid
* Uterus

**OpeBx**
* Ope
* Bx

**StorageTime**
FFPE storage period

## Usage

### Command-Line Execution

The prediction script can be executed directly from the command line by providing CSVs file as input and output.

```bash
git clone https://github.com/hnakahara/QCpredictor.git
cd QCpredictor
python predict_qc.py <input_csv> <output_csv>
```

### Web App

A browser-based version is available (Japanese / English): **https://qcpredictor.streamlit.app**
Enter the five items below and the four models' predictions are shown.
The storage time is calculated automatically as *submission date − collection date* (in days).

| Input | Model column |
|-------|--------------|
| Tumor nuclei (%) (0–100) | TN_HU |
| Cancer type | CancerType |
| Surgical / Biopsy | OpeBx |
| Collection date | — (used to calculate StorageTime) |
| Submission date | — (used to calculate StorageTime) |

Each model's prediction (Pass / Qualified) and the probability of *Qualified* are shown,
in the order GBM, RF, NN, SVM.
The SVM probability is not shown because the SVM was trained with `class_weight='balanced'`,
so its Platt-scaled probability does not agree with its prediction.

- **Pass**: likely to pass the QC check
- **Qualified**: detection sensitivity may be reduced

The storage time of the training data ranged from 0 to 4,714 days; a warning is shown for longer storage times.

The web app does not need TensorFlow: the neural network is evaluated with numpy using the weights in
`models/nn_qc_weights.npz`, exported from `models/nn_qc_model.keras` by `tools/export_nn_weights.py`
(re-run it only if the NN model is retrained; it requires TensorFlow).

#### Batch prediction (CSV)

Up to 1,000 samples can be predicted at once by uploading a CSV file (UTF-8 or Shift-JIS).
The first 5 results are shown on screen and all results can be downloaded as a CSV.

```csv
TN_HU,CancerType,OpeBx,CollectionDate,SubmissionDate
15,Adrenal_Gland,Bx,2024/01/01,2025/04/05
```

A `StorageTime` column (days) can be used instead of the two date columns, so `example/input_test.csv` can be uploaded as is.
Rows with invalid values are not predicted; the reason is written in the `Message` column of the result CSV.
The result CSV adds `StorageTime`, `<model>_pred` and `<model>_prob_Qualified` (0–1; not for SVM) columns.

Run locally (Python 3.12–3.14):

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app is designed to be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) (free).

## Citation

This tool was developed to evaluate the quality of comprehensive genomic profiling tests using tissue specimens.
If you use this tool in your research, please cite:

> Nakahara H, et al. npj Genomic Med. 2026.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC-By-NC) license. For more details, refer to the LICENSE file.