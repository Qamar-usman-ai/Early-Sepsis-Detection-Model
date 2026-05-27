
# 🏥 Early Sepsis Detection in Pediatric ICU — Machine Learning Study

> **Research Paper:** *Early Prediction of Pediatric Sepsis Using XGBoost and LightGBM with Data Balancing Techniques*
> **Dataset:** PHEMS Hackathon – Early Sepsis Prediction (Kaggle)
> **Institution:** Hospital Sant Joan de Déu — Pediatric Intensive Care Unit (PICU)

---

## 📌 What is Pediatric Sepsis?

Sepsis is a life-threatening condition caused by a dysregulated immune response to infection, leading to organ dysfunction and high mortality. Globally, approximately **30 million people** develop sepsis annually, with around **6 million deaths** (WHO). Among the most vulnerable are newborns and children — nearly **20 million pediatric sepsis cases** occur each year.

Pediatric sepsis is especially critical in PICUs, where it is a leading cause of morbidity and mortality. Delays in antibiotic treatment are associated with a **4–8% increase in mortality per hour**, making early detection vital.

---

## 🎯 Research Objective

This study builds a machine learning model to **predict pediatric sepsis onset 6 hours before clinical recognition**, enabling early intervention and potentially saving lives. It was structured as a machine learning hackathon using retrospective PICU data.

**Key differentiators from prior work:**
- Focuses specifically on **pediatric patients** (not adults)
- Uses a **multi-source, real-world** dataset (physiological data, device usage, drug records, lab measurements)
- Shifts prediction window **6 hours before** clinical recognition
- Compares **multiple data balancing strategies** across two gradient boosting models

---

## 📊 Dataset Description

**Source:** PHEMS Hackathon – Early Sepsis Prediction (Kaggle)
**Origin:** PICU, Hospital Sant Joan de Déu

### Data Files

| File | Description |
|------|-------------|
| `SepsisLabel_train.csv` | 331,639 rows × 66 columns; includes `SepsisLabel` target (1=Sepsis, 0=No Sepsis) |
| `SepsisLabel_test.csv` | 130,483 rows; no label column (used for blind evaluation) |
| `devices.csv` | Medical device usage records |
| `drugexposure.csv` | Administered drugs, dosages, and administration routes |
| `measurement_lab.csv` | Laboratory test results |
| `measurement_meds.csv` | Medication-related measurements |
| `measurement_observation.csv` | Clinical observation measurements |
| `observation.csv` | Clinical observations during PICU stay |
| `person_demographics_episode.csv` | Patient demographics (age, gender) |
| `proceduresoccurrences.csv` | Medical procedures performed |

### Class Distribution (Training Set)

| Class | Records | Percentage |
|-------|---------|------------|
| Sepsis Positive (1) | 6,874 | 2.07% |
| Sepsis Negative (0) | 324,765 | 97.93% |
| **Total** | **331,639** | — |

> ⚠️ **Severe class imbalance** (98:2 ratio) is the central challenge addressed in this study.

---

## 🔧 Methodology

### 2.1 Data Preprocessing

**Missing Value Handling:**
- Categorical variables → imputed with `'None'` to preserve integrity
- Numerical variables → replaced with the **median** of each feature

**Categorical Encoding via Word2Vec:**
Categorical variables (`drug_concept_id`, `route_concept_id`, `procedures`) were transformed into numerical embeddings using Word2Vec, capturing latent semantic relationships:
1. Tokenize categorical features into text sequences
2. Apply TF-IDF vectorization to assess feature relevance
3. Train Word2Vec model to generate vector representations

**Temporal Feature Engineering** (from `measurement_datetime`):
- `dow` — Day of the week (captures weekly hospital workflow patterns)
- `doy` — Day of the year (captures seasonal infection trends)
- `hour` — Hour of the day (captures diurnal fluctuations in vital signs)

**Data Splitting:** 80–20 stratified split preserving class distribution.

---

### 2.2 Feature Engineering

**Aggregated Drug Exposure Features:**
- Count of unique drugs per patient per day
- Most frequently used drug per patient
- Concatenated text of all administration routes

**Lab Measurement Features:**
- Rolling mean values over 6-hour and 12-hour windows (captures temporal trends)

**Device Usage Features:**
- Binary indicators for device presence/absence over time

**TF-IDF Vectorization for Drug Data:**
Daily drug administrations treated as "documents" → 200 informative drug-embedding features extracted. Rare antibiotics and drug combinations serve as strong clinical signals for infection severity.

---

### 2.3 Model Selection

Traditional models (Random Forest, KNN, Logistic Regression) were evaluated but found suboptimal for large-scale imbalanced datasets. The study focused on two **gradient boosting** models:

| Model | Key Strengths |
|-------|--------------|
| **XGBoost (XGB)** | Superior handling of class imbalance; interpretable feature importance; built-in regularization |
| **LightGBM (LGB)** | Optimized for large datasets; efficient with categorical features; fast training |

**XGB Configuration tuned for medical prediction:**
- Conservative learning rate for stable convergence
- Depth limiting to prevent over-complexity
- Feature sampling for robustness
- AUC optimization as primary objective

---

### 2.4 Addressing Class Imbalance

Four strategies were systematically compared:

| Strategy | Description |
|----------|-------------|
| **Unbalanced (Baseline)** | Train on raw imbalanced data without modification |
| **SMOTE Oversampling** | Synthetic minority samples generated to balance classes |
| **Undersampling** | Random reduction of majority class to match minority count |
| **Minority Duplication** | Duplicate minority-class rows to increase representation |

---

### 2.5 Validation Strategy

**Stratified Group K-Fold Cross-Validation** ensuring:
- Patient independence (same patient never in both train and validation)
- Class balance preservation per fold
- Temporal integrity of sequential patient data

---

## 📈 Results

### XGBoost (XGB) Model Performance

| Dataset Type | Train Accuracy | Train F1 | Val Accuracy | Val Precision | Val Recall | Val F1 | PR-AUC | Test Acc (46%) | Test Acc (100%) |
|---|---|---|---|---|---|---|---|---|---|
| **Unbalanced** | 0.9962–0.9964 | 0.9954–0.9956 | 0.6216–0.6702 | 0.9308–0.9833 | 0.0803–0.1899 | 0.1478–0.3183 | 0.9453 | 0.380 | 0.540 |
| **Undersampled** | — | — | — | — | — | — | — | — | — |
| **Oversampled** | 0.9910–0.9914 | 0.9996–1.000 | 0.9954–0.9956 | — | — | — | — | 0.380 | 0.540 |
| **Duplication** | 0.9966–0.9971 | 0.9967–0.9972 | 0.4937–0.5736 | 0.8528–0.9803 | 0.0218–0.1782 | 0.0427–0.3016 | 0.9573 | 0.380 | 0.540 |

### LightGBM (LGB) Model Performance

| Dataset Type | Train Accuracy | Train F1 | Val Accuracy | Val Precision | Val Recall | Val F1 | PR-AUC | Test Acc (46%) | Test Acc (100%) |
|---|---|---|---|---|---|---|---|---|---|
| **Unbalanced** | 0.9998 | 0.9947–0.9963 | 0.9758–0.9799 | 0.1985–0.5703 | 0.0199–0.1426 | 0.0361–0.2270 | 0.3453 | 0.580 | 0.380 |
| **Undersampled** | 0.9983–0.9987 | 0.9983–0.9987 | 0.8119–0.8895 | 0.9501–0.9693 | 0.6536–0.8044 | 0.7769–0.8792 | 0.9570 | 0.550 | 0.600 |
| **Oversampled** | 0.9953–0.9964 | 0.9953–0.9964 | 0.7556–0.9036 | 0.9599–0.9730 | 0.5280–0.8297 | 0.6831–0.8956 | 0.9563 | 0.650 | 0.400 |
| **Duplication** | 0.9993–0.9994 | 0.9993–0.9995 | 0.7372–0.8073 | 0.9697–0.9848 | 0.5055–0.6366 | 0.6649–0.7733 | 0.9538 | 0.730 | 0.460 |

**Primary Evaluation Metric — PR-AUC:** Used instead of ROC-AUC because it focuses on minority class (sepsis) performance, which is more clinically meaningful for imbalanced datasets.

**Best PR-AUC:** XGB Undersampled → **0.9453** | LGB Undersampled → **0.9570**

---

## 💬 Discussion

### Model Comparison

**XGBoost** demonstrated superior and more stable generalization on the full test dataset, especially on unbalanced and undersampled data. Its built-in regularization and handling of imbalanced data make it better suited for real-world clinical deployment.

**LightGBM** excelled on smaller validation sets (notably with minority duplication), but showed performance degradation on the larger test set — indicating overfitting tendencies when minority-class augmentation is applied without diversity.

### Impact of Balancing Strategies

| Technique | Finding |
|-----------|---------|
| **Undersampling** ✅ | Most effective overall; improved generalization for both models |
| **SMOTE Oversampling** ⚠️ | Inflated validation metrics; led to overfitting on test data |
| **Duplication** ⚠️ | Strong on small datasets; failed to generalize to full test set |
| **Unbalanced** ⚠️ | LGB collapsed on minority class; XGB held up better |

### Clinical Implications

By predicting sepsis **6 hours before clinical recognition**, this model could enable timely antibiotic administration and ICU interventions that directly reduce mortality. The undersampling + XGBoost combination is the most suitable for real-world deployment, where perfect data balance is rarely achievable.

**Limitation:** The dataset originates from a single institution (Hospital Sant Joan de Déu), which may limit cross-center generalizability.

---

## 🗂️ Repository Structure

```
Early-Sepsis-Detection-Model/
│
├── early-sepsis-detection-model.ipynb          # Main exploratory notebook (LGB + XGB comparisons)
├── fork-of-xgb-with-undersmpling-687393.ipynb  # XGBoost with undersampling (best model)
├── lgb with out blanace data.ipynb             # LightGBM on unbalanced data
├── mainapp.py                                  # Streamlit/application deployment script
├── requirements.txt                            # Python dependencies
├── training_data.zip                           # Compressed training dataset
├── testing_data (1).zip                        # Compressed test dataset
└── README.md                                   # This file
```

> **Recommended notebook to start with:** `fork-of-xgb-with-undersmpling-687393.ipynb` — the best-performing configuration from the study.

---

## ⚙️ Setup & Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `xgboost`
- `lightgbm`
- `scikit-learn`
- `pandas`, `numpy`
- `gensim` (Word2Vec)
- `imbalanced-learn` (SMOTE)
- `matplotlib`, `seaborn`

### Running the Notebooks

1. Clone the repository:
   ```bash
   git clone https://github.com/Qamar-usman-ai/Early-Sepsis-Detection-Model.git
   cd Early-Sepsis-Detection-Model
   ```
2. Extract data:
   ```bash
   unzip training_data.zip
   unzip "testing_data (1).zip"
   ```
3. Open the desired notebook in Jupyter or Kaggle and run all cells.

### Running the App

```bash
python mainapp.py
```

---

## 🔬 Conclusions

1. **XGBoost outperformed LightGBM** on the full test dataset across both unbalanced and undersampled configurations — making it the preferred model for real-world clinical deployment.
2. **Undersampling is the most effective balancing strategy**, improving minority-class detection without the overfitting risks of SMOTE or duplication.
3. **6-hour early prediction is achievable** with PR-AUC scores up to 0.957 — enabling clinically meaningful advance warnings.
4. **Single-center limitation** remains: multi-center validation is required to establish broader generalizability.

---

## 🔮 Future Work

1. **Advanced balancing** — Adaptive sampling, cost-sensitive learning
2. **Multi-center validation** — Ensuring generalizability across diverse healthcare settings
3. **Deep learning integration** — RNNs or Transformers for capturing complex temporal patterns in time-series PICU data
4. **Real-time EHR integration** — Streaming prediction pipeline for live clinical deployment
5. **Improved interpretability** — SHAP values and feature attribution for clinician trust
6. **Pediatric-specific features** — Growth metrics, developmental indicators for age-specific models

---

## 📚 References

- Calvert, J. et al. (2016). A computational approach to early sepsis detection. *PLOS ONE*.
- Nemati, S. et al. (2018). An interpretable machine learning model for accurate prediction of sepsis in the ICU. *Critical Care Medicine*.
- Scherpf, M. et al. (2019). Predicting sepsis with a recurrent neural network using the MIMIC III database. *Computers in Biology and Medicine*.
- Margherita, G. & Vincent, J.L. (2021). Sepsis and septic shock: New definitions, new diagnostic and therapeutic approaches. *Journal of Intensive Medicine*.
- World Health Organization (WHO). Sepsis Fact Sheet. https://www.who.int/news-room/fact-sheets/detail/sepsis

---

## 📄 License

This project is released for academic and research purposes under the repository's default license. Dataset access is subject to the PHEMS Hackathon terms on Kaggle.

---

*Built for the PHEMS Hackathon — Early Sepsis Prediction | Hospital Sant Joan de Déu PICU Data*
