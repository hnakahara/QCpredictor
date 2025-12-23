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

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC-By-NC) license. For more details, refer to the LICENSE file.