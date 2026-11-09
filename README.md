# Instamart Customer Intelligence & Support Copilot

<<<<<<< HEAD
An end-to-end customer intelligence and AI-powered support automation capstone built using Python, SQL, Machine Learning, Retrieval-Augmented Generation (RAG), and tool-calling agents.

---

## 1. Project Overview

The **Instamart Customer Intelligence & Support Copilot** is a reproducible, offline capstone project designed to demonstrate how data analytics, machine learning and AI-assisted automation can improve customer understanding and support operations.

The project is divided into four major parts:

- **Part A – Data Foundation:** Data auditing, cleaning, database creation, SQL analysis and exploratory data analysis.
- **Part B – Customer Segmentation and Lapse Prediction:** RFM analysis, K-Means clustering, customer profiling and machine learning models for lapse prediction.
- **Part C – Support Ticket Triage with Guardrails:** PII redaction, zero-shot and few-shot classification, retry handling, output validation, token budgeting and intelligent ticket routing.
- **Part D – Policy RAG Assistant and Refund Agent:** Document chunking, TF-IDF retrieval, grounded answers, refund tools, ReAct agent orchestration, safe stopping and checkpoint recovery.

The project uses synthetic Instamart customer, product, order and support-ticket datasets.

### Part Documentation

Each implementation folder contains its own README with module-specific details, execution steps, outputs and validation notes:

- [Part A – Data Foundation](src/part_a/README.md)
- [Part B – Customer Segmentation and Lapse Prediction](src/part_b/README.md)
- [Part C – Support Ticket Triage with Guardrails](src/part_c/README.md)
- [Part D – Policy RAG Assistant and Refund Agent](src/part_d/README.md)

### Key Characteristics

- Fully local and reproducible implementation.
- Uses the supplied `generate_data.py` and `mock_llm.py` without modification.
- No external API keys or hosted language models required.
- Simulated refund operations; no real payments are performed.
- Includes evaluation metrics, validation checks and generated outputs.
- Uses INR (Indian Rupees) for monetary values.

---

## 2. Project Objectives

1. Build a reliable analytical database from raw customer, product and order data.
2. Identify customer purchasing behaviour using RFM analysis and clustering.
3. Predict customer lapse risk using supervised machine learning.
4. Classify support tickets while protecting sensitive customer information.
5. Implement controlled model retries, output validation and token-budget enforcement.
6. Retrieve relevant company policies and generate grounded responses.
7. Automate refund-support workflows using tool-calling agents.
8. Implement safety mechanisms, human escalation and checkpoint-based recovery.

---

## 3. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Database | SQLite |
| Query language | SQL |
| Data visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Clustering | K-Means |
| Classification | Logistic Regression, Random Forest |
| Model evaluation | GridSearchCV, classification metrics |
| Text processing | Python Regex |
| Retrieval | TF-IDF, Cosine Similarity |
| AI simulation | Supplied `mock_llm.py` |
| Agent architecture | ReAct, function/tool calling |
| Model persistence | Joblib |
| Output formats | CSV, JSON, JSONL, TXT |

---

## 4. Project Structure

```text
Instamart-Customer-Copilot/
│
├── data/
│   ├── raw/
│   │   ├── agent_scenarios.csv
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   ├── policy_eval.csv
│   │   ├── products.csv
│   │   └── tickets.csv
│   │
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   └── products.csv
│   │
│   ├── policies/
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── refund_policy.txt
│   │
│   └── instamart.db
│
├── src/
│   ├── part_a/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── analyze_sql.py
│   │   ├── pipeline.py
│   │   └── validate_data.py
│   │
│   ├── part_b/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── rfm_analysis.py
│   │   ├── customer_segmentation.py
│   │   ├── export_customer_segments.py
│   │   ├── pca_analysis.py
│   │   ├── lapse_prediction.py
│   │   └── retention_list.py
│   │
│   ├── part_c/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── ticket_preprocessing.py
│   │   ├── pii_redaction.py
│   │   ├── ticket_classification.py
│   │   ├── retry_validation.py
│   │   ├── token_budget.py
│   │   └── segment_routing.py
│   │
│   ├── part_d/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── document_chunking.py
│   │   ├── retrieval.py
│   │   ├── rag_evaluation.py
│   │   ├── refund_tools.py
│   │   ├── react_agent.py
│   │   ├── checkpoint_resume.py
│   │   └── agent_comparison.py
│   │
│   └── __init__.py
│
├── sql/
│   └── queries.sql
│
├── outputs/
│   ├── lapse_model.joblib
│   ├── part_b/
│   │   ├── customer_cluster_profiles.csv
│   │   ├── customer_clusters.csv
│   │   ├── kmeans_evaluation.csv
│   │   ├── lapse_feature_importance.csv
│   │   ├── lapse_metrics.json
│   │   ├── lapse_test_predictions.csv
│   │   ├── lr_coefficients.csv
│   │   ├── pca_coordinates.csv
│   │   ├── pca_customer_segments.png
│   │   ├── pca_explained_variance.csv
│   │   ├── rf_top_features.csv
│   │   └── rfm_analysis.csv
│   │
│   ├── part_c/
│   │   ├── few_shot_results.csv
│   │   ├── few_shot_retry_log.csv
│   │   ├── final_retry_log.csv
│   │   ├── retry_log.csv
│   │   ├── budget_demo_retry_log.csv
│   │   ├── ticket_guardrail_audit.csv
│   │   ├── tickets_redacted.csv
│   │   ├── triage_budget_demo.csv
│   │   ├── triage_results.csv
│   │   ├── zero_shot_results.csv
│   │   └── zero_shot_retry_log.csv
│   │
│   ├── part_d/
│   │   ├── agent_comparison.txt
│   │   ├── agent_runs.jsonl
│   │   ├── checkpoint_resume_result.json
│   │   ├── policy_chunks.csv
│   │   ├── policy_chunks_no_overlap.csv
│   │   ├── rag_evaluation_details.csv
│   │   ├── rag_evaluation_summary.csv
│   │   └── safe_stopping_demo.json
│   │
│   ├── customer_segments.csv
│   ├── lapse_model.joblib
│   ├── part_a_analysis.json
│   ├── raw_audit.json
│   ├── refund_ledger.jsonl
│   └── retention_list.csv
│
├── state/
│   ├── .gitkeep
│   └── S1_checkpoint_demo.json
│
├── generate_data.py
├── mock_llm.py
├── run_part_a.py
├── requirements.txt
├── README.md
└── .gitignore
```

The structure above shows the main project organization. Individual modules and generated output files are maintained within their respective directories.

---

# PART A — DATA FOUNDATION

## 5. Objective

Part A establishes a reliable data foundation by auditing and cleaning raw datasets, storing the cleaned data in SQLite, performing SQL analysis and validating results using Pandas.

### 5.1 Raw Dataset Overview

| Dataset | Rows | Columns |
|---|---:|---:|
| Customers | 1,200 | 5 |
| Products | 40 | 4 |
| Orders | 13,084 | 9 |
| Support tickets | 164 | 5 |

### 5.2 Raw Data Audit

The raw data was examined for missing values, duplicates, invalid values, inconsistent categories and data-quality problems.
=======
## Overview

A reproducible, offline capstone using synthetic data to build a trusted analytical database, customer segments and lapse predictions, guarded support-ticket triage, and a policy-grounded refund agent. All monetary amounts are in INR. The supplied `generate_data.py` and `mock_llm.py` are used without modification; no API key or hosted model is required.

## Reproduce

Run these commands from the repository root with Python 3.11 or newer. SQLite is bundled with Python.

1. `python -m venv .venv`
2. Windows PowerShell: `.venv\Scripts\Activate.ps1` (Windows cmd: `.venv\Scripts\activate.bat`; macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_part_a.py`
7. `python -m src.part_a.analyze_sql`
8. `python -m src.part_a.validate_data`

The generator's final line must be exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The pipeline creates clean CSVs, the SQLite database, and `outputs/raw_audit.json`. The analysis script executes all seven SQL queries and saves `outputs/part_a_analysis.json`. After Parts B–D are implemented, their commands will be added here in execution order.

## Part A

### Raw-data audit

Raw shapes and dtypes:

| Table | Shape (rows × columns) | Column dtypes |
|---|---:|---|
| customers | 1,200 × 5 | `customer_id` object; `city` object; `signup_date` object; `age` float64; `gender` object |
| products | 40 × 4 | `product_id` object; `product_name` object; `category` object; `price_inr` int64 |
| orders | 13,084 × 9 | `order_id` object; `customer_id` object; `product_id` object; `quantity` int64; `order_ts` object; `status` object; `discount_pct` int64; `order_amount` object; `delivery_minutes` float64 |
| tickets | 164 × 5 | all five columns object; `keep_default_na=False` preserves empty ticket text as an empty string |

Raw missing values (count and percentage for every column):

| Table | Column | Missing count | Missing % |
|---|---|---:|---:|
| customers | customer_id | 0 | 0.00% |
| customers | city | 0 | 0.00% |
| customers | signup_date | 0 | 0.00% |
| customers | age | 61 | 5.08% |
| customers | gender | 30 | 2.50% |
| products | product_id | 0 | 0.00% |
| products | product_name | 0 | 0.00% |
| products | category | 0 | 0.00% |
| products | price_inr | 0 | 0.00% |
| orders | order_id | 0 | 0.00% |
| orders | customer_id | 0 | 0.00% |
| orders | product_id | 0 | 0.00% |
| orders | quantity | 0 | 0.00% |
| orders | order_ts | 0 | 0.00% |
| orders | status | 0 | 0.00% |
| orders | discount_pct | 0 | 0.00% |
| orders | order_amount | 53 | 0.41% |
| orders | delivery_minutes | 2,541 | 19.42% |
| tickets | ticket_id | 0 | 0.00% |
| tickets | customer_id | 0 | 0.00% |
| tickets | order_id | 0 | 0.00% |
| tickets | text | 0 | 0.00% |
| tickets | true_category | 0 | 0.00% |

The order amount audit also identifies 120 values that cannot be parsed as numbers; this count includes values represented as blank/NA by pandas as well as text such as `N/A` and `error`.
>>>>>>> origin/main

| Audit measure | Observed value |
|---|---:|
| Raw order rows | 13,084 |
<<<<<<< HEAD
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
=======
| Exact duplicate rows | 119 |
| Pending rows | 656 |
>>>>>>> origin/main
| Distinct order IDs | 12,309 |
| Unparseable order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
<<<<<<< HEAD
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank gender spellings | 10 |
| Blank genders | 30 |

### 5.3 Customer Data Cleaning

The following cleaning operations were performed:

1. Removed unnecessary whitespace from city names.
2. Standardized city names using title case.
3. Normalized gender values into `male`, `female` and `unknown`.
4. Treated ages outside the valid range of 10–100 as missing.
5. Imputed missing ages using the median age of the corresponding city.
6. Converted signup dates into datetime values.

The cleaned customer dataset contains 1,200 records.

#### Cleaned City Distribution

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

#### Cleaned Gender Distribution

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

All cleaned ages are populated and fall between 18 and 60.

### 5.4 Order Data Cleaning

The order-cleaning process included:

1. Converting order timestamps into datetime values.
2. Removing 119 exact duplicate rows.
3. Sorting records by order ID and timestamp.
4. Retaining the latest record for each order ID.
5. Removing pending statuses through the latest-record resolution process.
6. Converting order amounts into numeric values.
7. Replacing unparseable amounts using product price, quantity and discount information.
8. Replacing negative delivery durations with missing values.
9. Imputing missing delivery durations for completed orders using the completed-order median for the corresponding product category.
10. Keeping cancelled-order delivery durations missing because no delivery occurred.

#### Cleaned Order Results

| Measure | Result |
|---|---:|
| Final orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders | 0 |
| Missing order amounts | 0 |
| Missing completed delivery durations | 0 |
| Missing cancelled delivery durations | 1,199 |

**Completed-order revenue: INR 2,998,668.80**

### 5.5 SQLite Database

The cleaned datasets are stored in:

`data/instamart.db`

The database contains three tables:

- `customers`
- `products`
- `orders`

### 5.6 SQL Analysis

Seven SQL queries were implemented and executed against the SQLite database.

| Query | Concept | Result |
|---|---|---|
| Q1 | WHERE and HAVING | 30 customers |
| Q2 | Three-table JOIN | 30 city/category rows |
| Q3 | Scalar subquery | 11 products above average catalogue price |
| Q4 | LEFT JOIN | 25 customers without completed orders |
| Q5 | Common Table Expressions | 550 customers above mean completed spend |
| Q6 | Ranking window functions | City-wise top customers |
| Q7 | Running total window function | 183 daily revenue rows |

Additional SQL concepts included aggregate functions, GROUP BY, ORDER BY, filtering, subqueries, CTEs, RANK(), ROW_NUMBER(), window functions and running totals.

All seven queries are maintained in `sql/queries.sql`.

### 5.7 Pandas Cross-Check and EDA

Pandas was used to independently validate SQL results and perform exploratory analysis.

| Statistical measure | Result |
|---|---:|
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| Skewness | 4.61 |
| First quartile | INR 64.80 |
| Third quartile | INR 297.00 |
| Upper outlier fence | INR 645.30 |
| Orders above upper fence | 897 |

The order amount distribution is strongly right-skewed, indicating that a relatively small number of large orders influence the mean.

### 5.8 Part A Outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `outputs/raw_audit.json`
- `outputs/part_a_analysis.json`
- `sql/queries.sql`

### 5.9 Part A Verification

The raw audit counts, cleaned customer distributions, cleaned order counts, completed revenue, SQLite tables, seven SQL queries and Pandas cross-checks were verified against the project checkpoints.

---

# PART B — CUSTOMER SEGMENTATION AND LAPSE PREDICTION

## 6. Objective

Part B uses customer purchase history to understand customer behaviour, identify meaningful customer segments and predict customers who may stop purchasing.

The implementation combines RFM analysis, K-Means clustering, exploratory analysis and supervised machine learning.

### 6.1 RFM Analysis

RFM stands for:

- **Recency:** How recently a customer made a completed purchase.
- **Frequency:** How many completed purchases the customer made.
- **Monetary:** How much the customer spent on completed orders.

A total of **1,175 customers** were included in the RFM analysis.

### 6.2 K-Means Clustering

K-Means clustering was evaluated for cluster counts from K=2 through K=8.

Four clusters were selected for the final segmentation.

### 6.3 Customer Segment Profiles

| Cluster | Customers | Recency | Frequency | Monetary (INR) | Segment |
|---|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Customers |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | High-Value Premium Customers |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Loyal Frequent Customers |
| 3 | 182 | 109.04 | 2.20 | 685.99 | At-Risk Dormant Customers |

PCA was used to visualize the resulting customer segments.

### 6.4 Lapse Prediction Dataset

A separate customer cohort was created for lapse prediction.

| Measure | Result |
|---|---:|
| Total customers in cohort | 1,156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training records | 924 |
| Testing records | 232 |

### 6.5 Data Preprocessing

The preprocessing pipeline included:

- Numerical feature scaling.
- Categorical feature encoding.
- Training-only fitting of preprocessing transformations.
- Consistent transformation of training and testing data.

| Dataset | Shape |
|---|---|
| Training | 924 × 19 |
| Testing | 232 × 19 |

Fitting preprocessing only on training data helps prevent information leakage from the test set.

### 6.6 Model Selection

Two supervised learning algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier

GridSearchCV was used to tune model hyperparameters.

#### Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7284 | 0.6444 | 0.6517 | 0.6480 | 0.8011 |
| Random Forest | 0.7414 | 0.6667 | 0.6517 | 0.6591 | 0.7816 |

Random Forest achieved the higher test F1-score and was selected as the final model.

#### Random Forest Confusion Matrix

| Actual / Predicted | Not Lapsed | Lapsed |
|---|---:|---:|
| Not Lapsed | 114 | 29 |
| Lapsed | 31 | 58 |

### 6.7 Oversampling

Oversampling was applied to the training data to address class imbalance.

The balanced training set contained:

| Class | Records |
|---|---:|
| Non-lapsed | 570 |
| Lapsed | 570 |

The oversampled Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.7155 |
| Precision | 0.5891 |
| Recall | 0.8539 |
| F1-score | 0.6972 |
| ROC-AUC | 0.7873 |

The oversampled model achieved higher recall, identifying more potentially lapsed customers, with a corresponding precision trade-off.

### 6.8 Feature Importance

The leading Random Forest features were:

| Feature | Importance |
|---|---:|
| Frequency | 0.2809 |
| Monetary | 0.2696 |
| Recency | 0.1137 |
| Average delivery minutes | 0.0617 |
| Cancellation rate | 0.0527 |

Frequency and monetary value were the most influential features in the selected model.

### 6.9 Model Persistence and Retention List

The selected Random Forest model was saved using Joblib.

The saved model was reloaded and its predictions were checked against the original predictions.

A retention list containing 10 Premium customers from the test set was generated. Two of these customers were actually lapsed.

**Evaluation limitation:** The Premium segment flag is based on the whole observation window, including September. It is an illustrative post-hoc segment label and not a leakage-free predictive feature. The model's predictive features were constructed using information available before the lapse cutoff.

### 6.10 Part B Outputs

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- Additional preprocessing, evaluation and model analysis files under `outputs/part_b/`.

### 6.11 Part B Verification

The following were verified:

- RFM customer calculations.
- K-Means clustering.
- Four customer segment profiles.
- Lapse cohort construction.
- Training and testing preprocessing.
- Logistic Regression and Random Forest evaluation.
- Oversampling.
- Feature importance.
- Model persistence and reload.
- Retention list generation and validation.

---

# PART C — SUPPORT TICKET TRIAGE WITH GUARDRAILS

## 7. Objective

Part C implements an automated support-ticket triage pipeline using a simulated language model.

The system classifies customer support tickets while applying privacy protection, input guardrails, retry handling, output validation and token-budget restrictions.

### 7.1 Input Dataset and Guardrails

The ticket dataset contains 164 support tickets.

Before classification, each ticket is checked for empty content and excessive estimated token length.

The maximum allowed input length is 400 estimated tokens.

| Status | Count |
|---|---:|
| Eligible | 156 |
| Skipped empty | 4 |
| Flagged long | 4 |
| Total | 164 |

The maximum estimated token count observed was 734.

### 7.2 PII Redaction

The system identifies and redacts three types of personally identifiable information:

1. Email addresses.
2. Phone numbers.
3. Order IDs.

| PII type | Redactions |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| Total | 278 |

The redaction patterns were applied again to the processed text to verify that the identified PII patterns were no longer present.

Only the redacted ticket text is passed as the classification message content.

### 7.3 Zero-Shot and Few-Shot Classification

The classifier supports four permitted ticket categories:

- Billing
- Delivery
- Technical
- Fraud

**Zero-shot classification:** The model receives classification instructions without labelled examples.

**Few-shot classification:** The model receives classification instructions together with labelled examples to guide its predictions.

Both prompts require JSON-only responses containing `category` and `reason`.

### 7.4 Retry Handling

Transient model errors are handled using retry logic with exponential backoff and jitter.

The implementation supports a maximum of three retries.

The mock model generated 19 transient errors across the 156 eligible ticket calls. The retry mechanism recovered these failures.

Retry information is recorded with ticket identifiers, retry numbers and waiting times.

### 7.5 Output Validation

The classification response is validated before being accepted.

The validation process supports:

- Valid JSON with an allowed category.
- Controlled keyword fallback when the output is not valid JSON.
- Rejection of invalid or unsupported categories.
- Human-review handling for invalid responses.

| Status | Count |
|---|---:|
| Classified JSON | 135 |
| Classified keyword | 12 |
| Flagged invalid output | 9 |
| Skipped empty | 4 |
| Flagged long | 4 |
| Total | 164 |

A total of 147 tickets were successfully classified.

### 7.6 Classification Evaluation

Zero-shot and few-shot classification were evaluated using the same eligible ticket dataset.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 21,256 | 37,324 |

Few-shot classification achieved higher measured accuracy on the supplied synthetic benchmark but consumed more tokens.

These results reflect the supplied mock model and synthetic evaluation dataset, not production model performance.

### 7.7 Token Budget Demonstration

A zero-shot classification run was performed with a token budget of 6,000.

Before each model call, the pipeline checks whether the accumulated token usage has reached the configured budget.

| Status | Count |
|---|---:|
| Budget exhausted | 112 |
| Classified JSON | 39 |
| Classified keyword | 3 |
| Flagged invalid output | 2 |
| Skipped empty | 4 |
| Flagged long | 4 |
| Total | 164 |

Actual accumulated usage reached 6,010 tokens.

### 7.8 Customer Segment-Based Routing

The final few-shot routing run used a token budget of 100,000.

Customer segments from Part B were used to determine whether a ticket belonged to a Premium customer.

#### Routing Rules

| Condition | Destination | Priority |
|---|---|---|
| Empty ticket | Human review | NORMAL |
| Long ticket | Human review | NORMAL |
| Invalid model output | Human review | NORMAL |
| Budget exhausted | Human review | NORMAL |
| Fraud category | Security team | HIGH |
| Premium customer | Premium desk | HIGH |
| Other valid ticket | Support queue | NORMAL |

If a customer has no matching segment record, the customer is not treated as Premium.

#### Final Routing Distribution

| Route | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |
| Total | 164 |

All 34 fraud tickets were correctly routed to the security team.

### 7.9 Part C Outputs

- `outputs/part_c/ticket_guardrail_audit.csv`
- `outputs/part_c/few_shot_results.csv`
- `outputs/part_c/triage_budget_demo.csv`
- `outputs/part_c/triage_results.csv`
- `outputs/part_c/final_retry_log.csv`

### 7.10 Part C Verification

The following were verified:

- Ticket input validation.
- Empty and long ticket handling.
- PII redaction.
- Zero-shot and few-shot prompts.
- Retry recovery.
- JSON and keyword output validation.
- Classification accuracy.
- Token-budget enforcement.
- Customer segment-based routing.
- Fraud routing to the security team.

---

# PART D — POLICY RAG ASSISTANT AND REFUND AGENT

## 8. Objective

Part D implements a policy-grounded Retrieval-Augmented Generation assistant and a simulated refund-support agent.

The system retrieves relevant company policy information, generates answers using retrieved context and uses controlled tools to handle refund-related requests.

The implementation uses local documents, TF-IDF retrieval, the supplied simulated language model and local refund tools.

No external API keys, hosted models or real payment services are required.

### 8.1 Policy Document Chunking

Five policy documents are processed:

1. Refund policy.
2. Delivery policy.
3. Escalation policy.
4. Payment policy.
5. Privacy policy.

Documents are divided into chunks of 40 words with a 10-word overlap.

| Policy document | Chunks |
|---|---:|
| Refund policy | 4 |
| Delivery policy | 3 |
| Escalation policy | 3 |
| Payment policy | 3 |
| Privacy policy | 2 |
| Total | 15 |

### 8.2 TF-IDF Retrieval

The retrieval system uses `TfidfVectorizer` to transform policy chunks into numerical vectors.

Cosine similarity is used to rank the chunks according to their relevance to a customer query.

For the query:

`refund policy late delivery`

The top three retrieved chunks came from `refund_policy.txt`.

**TF-IDF concepts:**

- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Reduces the influence of terms that occur frequently across many documents.

Cosine similarity compares the direction of the query vector with the document vectors to identify relevant chunks.

**Limitation:** TF-IDF primarily relies on lexical similarity. It may not identify semantically related wording when important terms do not overlap.

### 8.3 RAG Answer Generation

The RAG assistant follows these steps:

1. Accept a customer question.
2. Convert the question into a TF-IDF vector.
3. Retrieve the top three relevant policy chunks.
4. Build a context containing source filenames and retrieved text.
5. Send the context and question to the simulated language model.
6. Generate a policy-grounded answer.
7. Return the answer with its source filenames.

The assistant is instructed to answer only from the supplied policy context.

If the retrieved documents do not contain sufficient information, the assistant must use the fallback response beginning:

`I could not find`

### 8.4 Retrieval and Grounding Evaluation

The evaluation dataset contains 14 questions:

- 12 answerable questions.
- 2 unsupported questions.

#### Results with 10-Word Overlap

| Evaluation measure | Result |
|---|---:|
| Expected document among top three | 12/12 |
| Expected fact contained in answer | 12/12 |
| Unsupported questions receiving fallback | 2/2 |

#### Results Without Overlap

| Evaluation measure | Result |
|---|---:|
| Expected document among top three | 12/12 |
| Expected fact contained in answer | 10/12 |
| Unsupported questions receiving fallback | 2/2 |

Without overlap, two answerable questions failed the expected-fact substring check because relevant information was split across chunk boundaries.

This demonstrates why overlapping chunks can improve contextual retrieval and answer grounding.

### 8.5 Refund Tools

Five tools were registered for the simulated refund-support workflow.

| Tool | Purpose |
|---|---|
| `get_order` | Retrieves order information from the local database |
| `search_policy` | Retrieves relevant policy documents |
| `calculate_refund` | Calculates refund eligibility and amount |
| `issue_refund` | Records an approved simulated refund |
| `escalate_to_human` | Records a refund request requiring human approval |

#### Refund Rules

- Only completed orders qualify for refunds.
- Delivery exceeding 45 minutes qualifies for a full refund.
- Delivery of 45 minutes or less receives no late-delivery refund.
- Refunds above INR 1,000 require human approval.
- Simulated refund actions are recorded in a local JSONL ledger.

The refund tools do not initiate real payments.

### 8.6 ReAct Agent Orchestration

The refund-support agent uses the ReAct architecture.

ReAct combines reasoning and action. The agent observes the request, calls an appropriate tool, examines the tool observation and decides its next action.

The general workflow is:

```text
Customer Request
       |
       v
   Get Order
       |
       v
  Search Policy
       |
       v
 Calculate Refund
       |
       v
  Refund Decision
       |
       +--------------------+
       |                    |
       v                    v
 Automatic Refund     Human Escalation
       |
       v
 Final Response
```

The agent supports tool execution, observation handling and final response generation.

### 8.7 Agent Scenario Evaluation

Seven predefined scenarios were evaluated.

| Scenario | Expected outcome | Actual outcome | Model calls |
|---|---|---|---:|
| S1 | REFUND | REFUND – INR 128.25 | 5 |
| S2 | REFUND | REFUND – INR 135.00 | 5 |
| S3 | ESCALATE | ESCALATE – INR 1,999 | 5 |
| S4 | DENIED | DENIED | 4 |
| S5 | DENIED | DENIED | 3 |
| S6 | NOT_FOUND | NOT_FOUND | 2 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 |

#### Overall Agent Metrics

| Metric | Result |
|---|---:|
| Total scenarios | 7 |
| Total model calls | 30 |
| Average model calls | 4.29 |

Token usage is recorded per run in `agent_runs.jsonl`. It is intentionally not presented as a fixed project-wide constant because it can change when prompts, retrieved context or implementation details change.

### 8.8 Safe Stopping Mechanisms

Three safety mechanisms were implemented and tested.

| Safety mechanism | Configuration | Result |
|---|---|---|
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Timeout | 0 seconds | FALLBACK_TIMEOUT |
| Maximum iterations | 6 iterations | FALLBACK_MAX_ITERATIONS |

The token-limit test verified that when a model response exceeded the token budget, the tool requested by that response was not executed.

The timeout test stopped execution before making a model call.

The maximum-iteration test prevented the agent from continuing indefinitely.

### 8.9 Checkpointing and Resume

Checkpointing allows the agent to recover its state after an interruption.

The implementation saves:

- Run ID.
- Scenario ID.
- Complete conversation messages.
- Accumulated token usage.
- Completed step number.
- Executed tool names.
- Model call count.
- Current iteration.

The S1 scenario was used to demonstrate recovery.

The execution was interrupted after the second tool execution, after its observation had been saved. The agent was then resumed from the saved checkpoint.

#### Verified Execution Sequence

```text
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund
```

All four tools executed exactly once across the interrupted and resumed run.

Final result:

**REFUND: INR 128.25 issued.**

### 8.10 ReAct versus Plan-and-Execute

ReAct combines reasoning and actions, allowing an agent to decide its next action based on actual tool observations.

Plan-and-Execute first creates an action sequence and then executes the plan.

ReAct was selected for this project because refund decisions depend on order status, delivery time, policy retrieval and refund calculations.

Unexpected observations such as a cancelled order or a missing order require the agent to adapt its next action safely.

The generated `outputs/part_d/agent_comparison.txt` is a qualitative architecture comparison. No Plan-and-Execute execution benchmark was implemented, so comparative performance savings are not claimed.

### 8.11 Part D Outputs

- `outputs/part_d/rag_evaluation_details.csv`
- `outputs/part_d/rag_evaluation_summary.csv`
- `outputs/part_d/agent_runs.jsonl`
- `outputs/part_d/safe_stopping_demo.json`
- `outputs/part_d/checkpoint_resume_result.json`
- `outputs/part_d/agent_comparison.txt`
- `outputs/refund_ledger.jsonl`
- `state/S1_checkpoint_demo.json`

The refund ledger is append-only in the current simulation. Re-running successful refund or escalation scenarios can add repeated demonstration records; these are not real payment transactions.

### 8.12 Part D Verification

The following were verified:

- Policy document chunking.
- TF-IDF vectorization.
- Cosine similarity retrieval.
- RAG answer generation.
- Unsupported-question fallback.
- Overlap versus no-overlap evaluation.
- Refund tool registration and execution.
- Seven agent scenarios.
- Token-limit enforcement.
- Timeout handling.
- Maximum-iteration handling.
- Checkpoint creation.
- Agent resume without repeating completed tools.
- ReAct versus Plan-and-Execute analysis.

---

# 9. Installation and Reproduction

## 9.1 Prerequisites

- Python 3.11 or newer.
- Git.
- Visual Studio Code or another Python-compatible editor.
- Windows, macOS or Linux.

## 9.2 Clone the Repository

```bash
git clone https://github.com/SANJAIKARTHIKEYAN/Instamart-Customer-Copilot.git
cd Instamart-Customer-Copilot
```

## 9.3 Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 9.4 Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9.5 Run Part A

```bash
python generate_data.py
python run_part_a.py
python -m src.part_a.analyze_sql
python -m src.part_a.validate_data
```

## 9.6 Run Part B

Run the modules in this order from the repository root:

```bash
python -m src.part_b.rfm_analysis
python -m src.part_b.customer_segmentation
python -m src.part_b.export_customer_segments
python -m src.part_b.pca_analysis
python -m src.part_b.lapse_prediction
python -m src.part_b.retention_list
```

The lapse-prediction pipeline generates `outputs/retention_list.csv`. The final `retention_list` command validates and summarizes that saved deliverable; it does not retrain the model or create duplicate retention records.

## 9.7 Run Part C

```bash
python -m src.part_c.ticket_preprocessing
python -m src.part_c.pii_redaction
python -m src.part_c.ticket_classification
python -m src.part_c.retry_validation
python -m src.part_c.token_budget
python -m src.part_c.segment_routing
```

## 9.8 Run Part D

```bash
python -m src.part_d.document_chunking
python -m src.part_d.retrieval
python -m src.part_d.rag_evaluation
python -m src.part_d.refund_tools
python -m src.part_d.react_agent
python -m src.part_d.checkpoint_resume
python -m src.part_d.agent_comparison
```

**Important:** Run commands from the repository root. Some commands create or append output files, including the simulated refund ledger. Re-running them may change or duplicate demonstration records. Use the generated outputs and checkpoints appropriate to each run.

---

# 10. Limitations and Future Improvements

### Current Limitations

1. The project uses synthetic customer, product, order and support-ticket datasets.
2. The language model is simulated using the supplied `mock_llm.py`; the measured results do not represent a live production LLM.
3. Policy retrieval uses TF-IDF and cosine similarity rather than a semantic embedding model.
4. The refund ledger records simulated actions and does not connect to a real payment gateway.
5. The Premium-segment retention list is illustrative because its whole-window segment label includes information from September.
6. Few-shot classification achieved higher accuracy on the supplied synthetic benchmark but consumed more tokens.
7. No Plan-and-Execute benchmark was implemented, so comparative performance is not measured.
8. The project has not been validated against a production Instamart system or real customer support data.

### Future Improvements

1. Integrate a production-grade language model with suitable privacy and security controls.
2. Evaluate semantic embeddings and hybrid retrieval methods.
3. Connect to an authorized order-management system.
4. Integrate a secure payment gateway with proper transaction controls.
5. Improve agent monitoring, human review and audit trails.
6. Evaluate performance using representative real-world data with appropriate authorization.
7. Implement and benchmark a Plan-and-Execute agent for comparison.
8. Add a user interface for customer support representatives.

---

# 11. Security, Privacy and Responsible Use

- PII redaction is applied before ticket text is sent to the simulated model.
- Empty and excessively long tickets are excluded from model processing.
- Model outputs are validated before being accepted.
- Token budgets and iteration limits restrict agent execution.
- Unsupported policy questions receive a fallback response.
- Refund amounts above the automatic approval threshold are escalated.
- Refund actions are simulated and recorded locally.
- No real customer payments or external transactions are performed.

The project is intended for educational demonstration and evaluation, not direct production deployment.

---

# 12. Final Project Status

The four project parts have been implemented. The Part D functional checks recorded during the current verification include the refund-tool checks, all seven scenario outcomes, the three safe-stopping fallbacks, checkpoint recovery, and the saved RAG evaluation summary.

| Project Part | Main Deliverables | Current documentation status |
|---|---|---|
| Part A | Data cleaning, SQLite, SQL analysis and EDA | Documented |
| Part B | RFM, customer segmentation and lapse prediction | Documented |
| Part C | Ticket classification, privacy guardrails and routing | Documented |
| Part D | Policy RAG, refund tools and ReAct agent | Core functional checks passed; final repository audit pending |

A clean-clone reproduction, complete dependency check, final output review and GitHub synchronization should be performed before submission. A passing module-level test does not by itself establish that the entire repository has been reproduced from a clean environment.

---

# 13. Author and Repository

**Project:** Instamart Customer Intelligence & Support Copilot

**GitHub Repository:**  
https://github.com/SANJAIKARTHIKEYAN/Instamart-Customer-Copilot

**Developer:** Sanjai Karthikeyan

---

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end approach to customer analytics and support automation.

By combining data engineering, SQL, machine learning, privacy-aware ticket classification, policy-based retrieval and tool-calling agents, the project illustrates how different technologies can work together to support customer intelligence and controlled customer-service workflows.

The project also emphasizes validation, explainability, privacy, token management, safe stopping and human escalation as important parts of building reliable AI-assisted systems.

## Documentation Guide

Use this root README for the end-to-end project overview, setup and cross-part workflow. For implementation details and part-specific outputs, use the README in the corresponding `src/part_*` directory.
=======
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and results

**Customers**

1. Strip city whitespace and apply title case; this normalizes 24 raw spellings to six cities.
2. Map all gender spellings case-insensitively to `male` or `female`; blank and unrecognized values become `unknown`.
3. Treat ages outside 10–100 as missing (6 invalid ages), then fill missing ages (61 originally blank plus 6 invalid) with the median age for that city using `groupby(...).transform('median')`.
4. Parse `signup_date` as a datetime.

| Cleaned city | Customers | Cleaned gender | Customers |
|---|---:|---|---:|
| Pune | 227 | female | 610 |
| Delhi | 210 | male | 560 |
| Bengaluru | 197 | unknown | 30 |
| Chennai | 191 | | |
| Hyderabad | 191 | | |
| Mumbai | 184 | | |

All 1,200 ages are populated and fall between 18 and 60. City counts and gender counts match the reference checkpoints.

**Orders**

1. Parse `order_ts` as a datetime.
2. Remove 119 exact duplicate rows.
3. Sort by `order_id` and timestamp, then retain the latest row for each order ID. This resolves the earlier pending records; final data has one row per order ID and no pending statuses.
4. Convert `order_amount` to numeric and fill all 120 unparseable values using `round(price_inr × quantity × (1 − discount_pct / 100), 2)` from the product catalogue.
5. Replace negative delivery durations with missing values. For completed orders only, impute missing delivery duration with the median for the product category among completed orders. Leave cancelled-order delivery durations missing because no delivery occurred.

The cleaning sequence removes 775 rows in total (119 exact duplicates and 656 pending records after exact-deduplication), leaving 12,309 orders: 11,110 completed and 1,199 cancelled. There are no missing order amounts, no missing delivery durations among completed orders, and 1,199 missing delivery durations among cancelled orders. Completed-order revenue is **INR 2,998,668.80**.

### Store and validate

Clean tables are written to `data/clean/customers.csv`, `data/clean/products.csv`, and `data/clean/orders.csv`; the SQLite database `data/instamart.db` contains tables `customers`, `products`, and `orders`. The validation script checks more than eight reference checkpoints and prints `ALL CHECKPOINTS PASSED` only when the assertions pass.

### SQL analysis

All seven queries are stored in `sql/queries.sql` and executed with `pandas.read_sql_query` against the SQLite database.

| Query | Result |
|---|---|
| Q1 — `WHERE` and `HAVING` | 30 customers |
| Q2 — three-table join | 30 city/category rows; total revenue INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 — scalar subquery | 11 products priced above catalogue average |
| Q4 — `LEFT JOIN` | 25 customers with no completed order |
| Q5 — CTEs | 550 customers above the mean completed spend; average customer spend INR 2,552.06 |
| Q6 — ranking windows | Top customers: Bengaluru C0082 (INR 12,856.20); Chennai C0930 (INR 10,151.40); Delhi C0291 (INR 13,742.60); Hyderabad C0743 (INR 12,730.05); Mumbai C0693 (INR 14,766.55); Pune C0426 (INR 10,377.10). Eight rows have `RANK()` different from `ROW_NUMBER()`. |
| Q7 — running total | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

`RANK()` leaves gaps after tied values, while `ROW_NUMBER()` assigns a unique sequential number to every row, so ties can make the two rankings differ.

### Pandas cross-check and exploratory analysis

The `pivot_table(..., margins=True)` grand total is INR 2,998,668.80, matching SQL Q2. City aggregates for completed orders:

| City | Sum (INR) | Mean (INR) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order amount has mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR fences are −INR 283.50 and INR 645.30; there are zero values below the lower fence and 897 above the upper fence.

The data describes synthetic completed purchases across six cities and five product categories. Order amounts are strongly right-skewed: the mean is much higher than the median, and 897 orders exceed the upper outlier fence, plausibly reflecting larger quantities or expensive electronics. The largest absolute pairwise correlation is between order amount and discount percentage (about −0.095); order amount and quantity is about +0.087. Both relationships are weak, and correlation of this size does not establish causation.

### Part A verification status

Verified in the current runtime: raw audit counts match the brief; customer city/gender distributions match; cleaned order counts and completed revenue match; the SQLite tables exist; all seven SQL queries return the reference counts/totals; and pandas cross-checks match. `outputs/raw_audit.json` and `outputs/part_a_analysis.json` preserve the measured results.

## Part B

Not implemented yet. This section will be completed after Part A is frozen and verified.

## Part C

Not implemented yet. This section will be completed after Part B is frozen and verified.

## Part D

Not implemented yet. This section will be completed after Part C is frozen and verified.

## Limitations and Next Steps

The data and policy documents are synthetic and fictional. No real customer records, live payment systems, hosted LLMs, or API keys are used. Next, implement Parts B–D in dependency order, record actual metrics and outputs, then rerun the complete reproduction sequence from a clean clone. The Premium-segment retention list in Part B is a post-hoc whole-window label and must be described as illustrative, not a leakage-free evaluation.
>>>>>>> origin/main
