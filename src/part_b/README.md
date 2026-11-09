# Part B — Customer Segmentation and Lapse Prediction

## Overview

Part B uses the cleaned Part A customer and order data to analyse purchasing behaviour, create customer segments and predict customer lapse risk. The implementation combines RFM analysis, K-Means clustering, PCA visualization and supervised machine learning.

## Source files

| File | Responsibility |
|---|---|
| `rfm_analysis.py` | Calculates customer Recency, Frequency and Monetary features |
| `customer_segmentation.py` | Evaluates and fits K-Means customer clusters and creates segment profiles |
| `export_customer_segments.py` | Exports the customer segment mapping used by later workflows |
| `pca_analysis.py` | Produces PCA coordinates, explained-variance data and a segment visualization |
| `lapse_prediction.py` | Builds the lapse cohort, preprocessing pipeline, model search/evaluation, feature interpretation and saved model/retention outputs |
| `retention_list.py` | Loads and validates/summarizes the saved retention-list output |

## Inputs

Part B depends on Part A's cleaned data and SQLite database. Its primary customer-behaviour analysis uses completed order history. The exact input paths, feature definitions and lapse cutoff are defined in the source modules.

## Workflow

### 1. RFM analysis

RFM represents:
- **Recency:** how recently the customer completed a purchase;
- **Frequency:** the number of completed purchases;
- **Monetary:** the amount spent on completed purchases.

The supplied run included 1,175 customers in the RFM analysis.

### 2. K-Means segmentation

K-Means was evaluated for cluster counts from K=2 through K=8. Four clusters were selected for the final segmentation. Segment profiles are generated from the RFM characteristics and are visualized using PCA.

| Cluster | Customers | Segment profile |
|---|---:|---|
| 0 | 436 | Occasional Customers |
| 1 | 138 | High-Value Premium Customers |
| 2 | 419 | Loyal Frequent Customers |
| 3 | 182 | At-Risk Dormant Customers |

### 3. Lapse prediction

A separate cohort of 1,156 customers was used for lapse prediction: 443 lapsed customers and 713 retained customers. The train/test split contains 924 training records and 232 test records.

Numerical features are scaled and categorical features are encoded. Preprocessing is fitted on the training split only and then applied to the test split to reduce data leakage. Logistic Regression and Random Forest are tuned with GridSearchCV and evaluated using classification metrics.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7284 | 0.6444 | 0.6517 | 0.6480 | 0.8011 |
| Random Forest | 0.7414 | 0.6667 | 0.6517 | 0.6591 | 0.7816 |

Random Forest had the higher F1-score in this recorded run and was selected as the final model. The project also evaluates oversampling on the training data. The oversampled Logistic Regression run achieved recall 0.8539 and F1-score 0.6972.

### 4. Model persistence and retention list

The selected model is saved with Joblib, reloaded and checked by comparing its predictions with the original predictions. A retention list of 10 Premium customers from the test set is generated.

**Interpretation limitation:** The Premium segment flag is a post-hoc segment label based on the whole observation window, including September. It is illustrative and must not be interpreted as a leakage-free predictive feature. Predictive model features are constructed using information available before the lapse cutoff.

## Generated outputs

Project-level outputs:
- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

Part B analysis outputs in `outputs/part_b/`:
- `rfm_analysis.csv`
- `customer_clusters.csv`
- `customer_cluster_profiles.csv`
- `kmeans_evaluation.csv`
- `pca_coordinates.csv`
- `pca_explained_variance.csv`
- `pca_customer_segments.png`
- `lapse_metrics.json`
- `lapse_test_predictions.csv`
- `lapse_feature_importance.csv`
- `rf_top_features.csv`
- `lr_coefficients.csv`

## Run commands

Run from the repository root after Part A:

```bash
python -m src.part_b.rfm_analysis
python -m src.part_b.customer_segmentation
python -m src.part_b.export_customer_segments
python -m src.part_b.pca_analysis
python -m src.part_b.lapse_prediction
python -m src.part_b.retention_list
```

## Verification scope

The project workflow checks RFM calculations, cluster creation/profiles, PCA outputs, lapse cohort construction, train/test preprocessing, model evaluation, oversampling, feature importance, model persistence/reload and retention-list generation. Metrics above describe the supplied synthetic dataset and recorded run; they are not production performance guarantees.
