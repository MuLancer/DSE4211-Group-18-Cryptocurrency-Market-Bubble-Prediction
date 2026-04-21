# DSE4211-Group-18-Cryptocurrency-Market-Bubble-Prediction

## Detailed Notebooks
- Feature engineering
  1. Data collection: `data_collection.ipynb`
  2. Data cleaning: `data_cleaning.ipynb`
  3. Feature engineering: `feature_engineering.ipynb`
  4. Data standardisation: `data_standardisation.ipynb`
 
- Bubble labelling
  1. Labelling: `bubble_labelling.ipynb`
     - Combining labels with df_features: `combined_features_labelling.ipynb`
     - Combining labels with integrated_data: `combined_labelling.ipynb`
  3. Evaluation: `bubbles_cooccurrence.ipynb`
  2. Co-occurrence plot: data/outputs/
     
- PCA
  1. Model: `pca_dimensionality_reduction.ipynb`
  2. Evaluation: data/outputs/
     
- LASSO model
  1. Model: `lasso.ipynb`
  2. Saved models in pkl file: models folder
  3. Evaluation: data/outputs/
     
- XGBoost model
  1. Model: `xgboost.ipynb`
  2. Evaluation: xgboost_stats folder

- LSTM model
  1. Model: `lstm.ipynb`

- Inter-model comparisons
  1. `main.ipynb` 

## Dependencies
- Install the relative dependecies using `pip install -r requirements.txt`

## Overall Results

| Model    | per_coin | global | delta (per-global) |
|----------|----------|--------|--------------------|
| LASSO    | 0.4673   | 0.3294 | 0.1379             |
| LSTM     | 0.3404   | 0.3862 | -0.0458            |
| XGBoost  | 0.4861   | 0.4275 | 0.0587             |


## Replication
1. Ensure that the relevant dependencies have been installed using the requirements.txt file
2. Run the specific notebooks desired. 



