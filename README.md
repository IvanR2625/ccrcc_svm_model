# ccrcc_svm_model

A Support Vector Machine (SVM) classifier that predicts whether a kidney tissue sample is **cancerous (clear cell renal cell carcinoma, ccRCC)** or **normal**, using the expression level of the microRNA **hsa-let-7b**.

The model takes a single number, the let-7b expression in **reads per million miRNA mapped (RPM)**, and returns a *Cancer* / *Normal* prediction with a probability.

The data comes from the **TCGA-KIRC** (Kidney Renal Clear Cell Carcinoma) project, obtained through the [NCI Genomic Data Commons (GDC) portal](https://portal.gdc.cancer.gov/).

---

## Repository contents

| File | Description |
| --- | --- |
| `svm_model.py` | Main script. Trains the SVM (if no saved model exists), saves it, and then runs an interactive prompt that classifies RPM values you type in. |
| `box_plot.py` | Draws a box plot of let-7b RPM across tissue stages (Normal, Stage-1 to Stage-4). |
| `best_svm_model.joblib` | The trained SVM model (best estimator from the grid search). |
| `scaler.joblib` | The fitted `StandardScaler` used to normalize RPM input. |
| `model_info` | Plain-text evaluation report written during training (accuracy, precision, recall, ROC-AUC, confusion matrix, classification report). |
| `Python_Data.xlsx` | Training data. Each sample's `Stage` is labeled `Normal` or `Cancer`. |
| `KIRC_Project_Data_ALL.xlsx` | The same samples with the full staging (`Normal`, `Stage-1` … `Stage-4`). `box_plot.py` reads this file. |

### Data format

Both workbooks have one worksheet per let-7 family miRNA:
`hsa-let-7a-1`, `hsa-let-7a-2`, `hsa-let-7a-3`, `hsa-let-7a MEAN`, `hsa-let-7b`, `hsa-let-7c`, `hsa-let-7d`, `hsa-let-7e`, `hsa-let-7f-1`, `hsa-let-7f-2`, `hsa-let-7g`, `hsa-let-7i`.

Each sheet has these columns:

| Column | Meaning |
| --- | --- |
| `Stage` | Tissue class (`Normal` / `Cancer` in `Python_Data.xlsx`; `Normal`, `Stage-1` … `Stage-4` in `KIRC_Project_Data_ALL.xlsx`) |
| `CaseID` | TCGA case barcode (e.g. `TCGA-DV-5565`) |
| `FileUUID`, `FileURL`, `FileName` | GDC file identifiers and links to the source data |
| `read_Count` | Raw read count for the miRNA |
| `read_per_million_miRNAMapped` | Normalized expression (RPM); **this is the model's input feature** |
| `Cross_mapped` | Whether the reads are cross-mapped (`Y` / `N`) |
| `MEAN_STAGE` | Mean RPM for the stage group |

The current model uses only the **`hsa-let-7b`** sheet.

---

## How the model works

`svm_model.py` runs these steps:

1. **Load data:** reads the `hsa-let-7b` sheet from `Python_Data.xlsx`.
2. **Label:** `Cancer` → `1`, everything else (`Normal`) → `0`.
3. **Scale:** standardizes `read_per_million_miRNAMapped` with `StandardScaler`.
4. **Balance classes:** cancer samples far outnumber normal ones, so SMOTE (`imblearn`) oversamples the minority class.
5. **Split:** 80% train / 20% test (`random_state=42`).
6. **Tune:** `GridSearchCV` (5-fold, scored on accuracy) over:
   - `C`: `0.1, 1, 10, 100`
   - `kernel`: `linear, rbf`
   - `gamma`: `scale, auto`
7. **Evaluate and save:** writes the metrics to `model_info` and saves the model and scaler as `.joblib` files.

If `best_svm_model.joblib` and `scaler.joblib` already exist, the script skips training and loads them.

---

## Results

From `model_info` (215 test samples):

| Metric | Value |
| --- | --- |
| Accuracy | 0.907 |
| Precision | 0.887 |
| Recall | 0.936 |
| ROC-AUC | 0.931 |

**Confusion matrix** (rows = actual, columns = predicted):

|  | Pred. Normal | Pred. Cancer |
| --- | --- | --- |
| **Normal** | 93 | 13 |
| **Cancer** | 7 | 102 |

---

## Getting started

### Requirements

- Python 3.9+
- `pandas`, `openpyxl` (to read `.xlsx`)
- `scikit-learn`, `imbalanced-learn`, `joblib`
- `matplotlib`, `seaborn` (≥ 0.13, for `box_plot.py`)

### Installation

```bash
git clone https://github.com/IvanR2625/ccrcc_svm_model.git
cd ccrcc_svm_model
pip install pandas openpyxl scikit-learn imbalanced-learn joblib matplotlib seaborn
```

> **Note:** `.joblib` files are tied to the scikit-learn version that created them. If loading the saved model fails or prints version warnings, delete `best_svm_model.joblib` and `scaler.joblib` and run the script again to retrain.

### Usage

Run the scripts from the repository root, because they use relative file paths.

#### Classify an RPM value

```bash
python svm_model.py
```

Example session:

```
Loading, finished importing modules...
Model and scaler already exist at best_svm_model.joblib and scaler.joblib.
Do you want the info on the model loaded? Y for Yes, N for No
N
What is the given RPM for let-7b?
25000
Predicted: Cancer, XX.X% chance
Do you want to test again? Y for Yes, N for No.
N
```

(The input value is only an example. The prediction and probability depend on the RPM you enter.)

#### Retrain the model

Delete the saved artifacts, then run the script again:

```bash
rm best_svm_model.joblib scaler.joblib
python svm_model.py
```

#### Plot let-7b expression by stage

```bash
python box_plot.py
```

This opens a box plot of let-7b RPM for Normal tissue and Stages 1–4.

---

## Limitations

- **Single feature.** The model uses let-7b expression only. Other let-7 family members are included in the spreadsheets but are not used.
- **Optimistic evaluation.** SMOTE runs *before* the train/test split, so the test set includes synthetic samples, and some of them are interpolated from training points. The reported metrics are therefore likely higher than performance on new, real data. For a stricter evaluation, split the data first and apply SMOTE only to the training set (e.g. with an `imblearn.pipeline.Pipeline` inside cross-validation).
- **Interactive input only.** The script reads RPM values from the console. It does not validate input, and typing `Y`/`N` is case-sensitive.
- **Research use only.** This is an educational/research project and **is not a diagnostic tool**.

---

## Data source and acknowledgements

The data was generated by the TCGA Research Network (https://www.cancer.gov/tcga) and accessed through the NCI Genomic Data Commons. The `FileURL` column in each spreadsheet links to the original GDC file for each sample.
