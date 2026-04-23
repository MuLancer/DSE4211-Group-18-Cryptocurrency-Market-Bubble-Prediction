# DSE4211 Group 18 — Cryptocurrency Market Bubble Prediction

## Project Structure

```
.
├── data/                              # Raw and processed datasets
│   └── outputs/                       # Generated plots and CSVs
├── models/                            # Saved LASSO model weights (.pkl)
├── src/                               # Model implementations
│   ├── lasso_model.py
│   ├── lstm_model.py
│   └── xgboost_model.py
├── xgboost_stats/                     # XGBoost evaluation outputs
├── old/                               # Legacy standalone model notebooks
│   ├── lasso.ipynb
│   ├── lstm.ipynb
│   └── xgboost.ipynb
└── *.ipynb                            # Pipeline notebooks (see below)
```

## Notebooks

### Feature Engineering Pipeline
| Step | Notebook |
|------|----------|
| 1. Data collection | `data_collection.ipynb` |
| 2. Data cleaning | `data_cleaning.ipynb` |
| 3. Feature engineering | `feature_engineering.ipynb` |
| 4. Data standardisation | `data_standardisation.ipynb` |

### Bubble Labelling
| Step | Notebook |
|------|----------|
| 1. Labelling | `bubble_labelling.ipynb` |
| 2. Merge labels with features | `combined_features_labelling.ipynb` |
| 3. Merge labels with integrated data | `combined_labelling.ipynb` |
| 4. Co-occurrence evaluation | `bubbles_cooccurrence.ipynb` |

### Dimensionality Reduction
| Step | Notebook |
|------|----------|
| PCA | `pca_dimensionality_reduction.ipynb` |

### Models & Comparisons
All three models (LASSO, XGBoost, LSTM) and inter-model comparisons are run from:

**`main.ipynb`** — loads model classes from `src/` and produces outputs in `data/outputs/` and `xgboost_stats/`.

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

Requires **Python 3.11+**.

## Overall Results

| Model   | Per-Coin F1 | Global F1 | Delta (Per − Global) |
|---------|-------------|-----------|----------------------|
| LASSO   | 0.4673      | 0.3294    | +0.1379              |
| LSTM    | 0.3404      | 0.3862    | −0.0458              |
| XGBoost | 0.4861      | 0.4275    | +0.0587              |

## Replication

1. Install dependencies: `pip install -r requirements.txt`
2. Run the feature engineering pipeline notebooks in order (steps 1–4 above).
3. Run `bubble_labelling.ipynb`, then `combined_features_labelling.ipynb` / `combined_labelling.ipynb`.
4. Run `pca_dimensionality_reduction.ipynb` for PCA.
5. Run `main.ipynb` to train all models and generate comparison outputs.
