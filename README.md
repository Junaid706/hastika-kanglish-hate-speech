# HASTIKA @ ICON-2026: Kanglish Hate Speech Detection

**Hate Speech and Target Category Identification in Kannada-English Code-Mixed Text**

Submission to the [HASTIKA shared task](https://www.codabench.org/competitions/17784/) at ICON-2026.

## Authors

- Junaid Abedin Rafi
- Rifah Tasnim Jui
- Tasfia Tabassum Toma

## Tasks

- **Task A - Binary Hate Speech Detection:** classify each comment as `Hate` or `Non-Hate`.
- **Task B - Fine-Grained Classification:** classify each hate comment into one of six categories: `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others`.

## Approach

- **Preprocessing:** HTML entity and URL cleanup, whitespace normalisation. No spelling normalisation, since spelling variation in transliterated Kanglish text is informative signal.
- **Features:** TF-IDF over word n-grams (1-2) and character n-grams (2-5, word-boundary aware). Character n-grams help generalise across inconsistent transliteration spellings.
- **Model:** Logistic Regression (`class_weight='balanced'`, `C=5`) to handle class imbalance, especially for the minority Task B categories (Geo-political, Violence).
- **Validation:** 85/15 stratified split to estimate Macro-F1; the final model is retrained on 100% of the labelled data before predicting the test set.

## Results

Official test set (gold labels released by the organisers after the evaluation):

| Task | Macro-F1 | Accuracy |
|------|----------|----------|
| Task A - Binary | 0.8027 | 0.8027 |
| Task B - Fine-Grained | 0.6296 | 0.7020 |

Development phase (CodaBench validation leaderboard): Task A 0.80 / 0.80, Task B 0.60 / 0.71 (Macro-F1 / Accuracy).

## Files

| File | Description |
|------|-------------|
| `train_baseline.py` | Training and evaluation pipeline (development phase) |
| `predict_test.py` | Final prediction script run on the official test set |
| `test_predictions_taskA.csv` | Final Task A predictions (`id,label`) |
| `test_predictions_taskB.csv` | Final Task B predictions (`id,label`) |

## Reproducing

The HASTIKA data is not redistributed here (see the task terms). Download it from the
[official repository](https://github.com/shankarb14/Hastika-ICON2026) and place these files next to the scripts:

- `binary_train.csv` and `multiclass_train.csv`
- `hastika_binary_test.csv` and `hastika_multiclass_test.csv` (needed by `predict_test.py`)
- `hastika_binary_validation.csv` and `hastika_multiclass_validation.csv` (needed by `train_baseline.py`)

Then run:

```
pip install pandas scikit-learn scipy
python predict_test.py      # writes test_predictions_taskA.csv and test_predictions_taskB.csv
python train_baseline.py    # internal 85/15 check and validation-set predictions
```

`predict_test.py` reproduces the two submitted prediction files in this repository exactly.

## Future Work

- Fine-tune multilingual transformers (MuRIL, IndicBERT, XLM-RoBERTa) for stronger contextual representations.
- Address Task B class imbalance with focal loss or targeted oversampling.
- Ensemble TF-IDF-based and transformer-based models.

## Citation

> Kavatagi, S., Rachh, R. (2025). HASTIKA: hate speech and target identification in Kannada-English code-mixed text. *Language Resources and Evaluation*, 59(3), 2811-2856.

---
Shared task organised by Manipal Academy of Higher Education (MAHE). Official starting kit: [shankarb14/Hastika-ICON2026](https://github.com/shankarb14/Hastika-ICON2026)
