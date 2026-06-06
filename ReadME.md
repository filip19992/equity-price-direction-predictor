# Equity Price Direction Prediction

This repository contains the code, datasets and experiment notebooks used for a master's thesis on predicting short-term stock price direction with market data and alternative data sources.

The current research setup is a three-class classification problem. The model predicts whether the next trading-session return is negative, neutral or positive. The active dataset is the rich-price, session-aligned, full-history panel:

```text
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_raw.csv
```

The active configuration is stored in:

```text
notebook_utils/experiment_config.py
```

## Current Experiment Variant

The current experiment uses:

- a nine-ticker panel dataset with `AAPL`, `AMD`, `AMZN`, `GOOGL`, `META`, `MSFT`, `NFLX`, `NVDA` and `TSLA`,
- rich OHLCV-style price data, not only adjusted close prices,
- Google Trends, GDELT news data and Reddit sentiment/coverage variables,
- a three-class target based on the next-session return,
- chronological train, validation and test splits,
- multiple feature-set variants built centrally in `notebook_utils/feature_set_grid_builder.py`,
- several model families evaluated through notebooks in `notebooks/`.

In the current modeling configuration, `NFLX` is excluded from model training and evaluation. The raw dataset still contains `NFLX`, but the experiment configuration removes it before fitting models.

## Repository Structure

```text
data/                         Raw and processed datasets
  datasets/                   Final panel datasets and dataset audits
  equity_data/                Imported equity and alternative data files
  gpw_data/                   Additional GPW-related data

datamerger/                   Scripts and notebooks for merging source data into panel datasets
equity_data_importers/        Importers for stock prices, Google Trends, GDELT and Reddit data
gpw_data_visualisation/       Additional visualization utilities
notebook_utils/               Shared experiment configuration, feature generation, splits and metrics
notebooks/                    EDA and model-training notebooks
environment.yml               Conda environment definition
ReadME.md                     Project overview
```

## Data Sources

The project combines market data and alternative data. The main source groups are:

| Source group | Main purpose | Import location |
| --- | --- | --- |
| Yahoo Finance | Historical stock prices and trading volume | `equity_data_importers/stock_price_importer.py`, `equity_data_importers/stock_price_rich_importer.py` |
| Yahoo Finance benchmark data | Market context, mainly QQQ benchmark variables | `equity_data_importers/market_benchmark_importer.py` |
| Google Trends | Search interest for company- or ticker-related queries | `equity_data_importers/google_trends_importer.py` |
| GDELT | News/event coverage variables | `equity_data_importers/gdelt_importer.py` |
| Reddit dumps | Reddit post and comment activity related to selected tickers | `equity_data_importers/reddit_importer.py`, `equity_data_importers/reddit_comments_importer.py` |
| Sentiment tools | Sentiment scores calculated during data import | VADER and optional FinBERT utilities in `equity_data_importers/` |

The current modeling feature grid uses VADER-based Reddit comment sentiment features. FinBERT sentiment columns may exist in prepared source files, but they are not part of the currently active model feature grid.

## Dataset Construction

Final modeling datasets are built in `datamerger/`. The current dataset variant is produced by the rich-price session-aligned builder:

```text
datamerger/rich_price_session_aligned_full_history_builder.py
```

This builder combines stock prices and alternative data into one panel. The output is organized by ticker and trading session date. Calendar-based alternative data is assigned to trading sessions so it can be used consistently with market data.

The main current dataset file is:

```text
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_raw.csv
```

Associated metadata and audit files describe the generated dataset:

```text
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_metadata.json
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_summary.csv
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_source_audit.csv
data/datasets/stock_panel_nine_tickers_session_aligned_full_history_rich_price_adjusted_google_score_reddit_coverage_audit.csv
```

The current rich-price raw panel contains:

- 9 tickers,
- 1,255 trading sessions per ticker,
- 11,295 rows in total,
- dates from `2021-01-04` to `2025-12-31`,
- 48 columns before model-specific feature engineering.

## Target Variable

The target is created during feature-frame construction, not stored directly as a ready-made class column in the raw CSV.

The target is based on the next-session return:

```text
future_return_1d = next_close / current_close - 1
```

The current three-class setup uses a neutral band of `0.005`:

| Class | Meaning | Rule |
| --- | --- | --- |
| `0` | down | `future_return_1d < -0.005` |
| `1` | neutral | `-0.005 <= future_return_1d <= 0.005` |
| `2` | up | `future_return_1d > 0.005` |

The class labels are defined in `notebook_utils/metrics.py`.

## Feature Engineering

Feature construction is centralized in:

```text
notebook_utils/modeling.py
notebook_utils/feature_set_grid_builder.py
```

Every feature set includes the core price-derived features:

```text
return_1d
return_5d
return_20d
rolling_volatility_20d
intraday_return
overnight_gap_return
daily_high_low_range
close_position_in_daily_range
```

Additional feature-set variants add selected groups of variables:

- volume features,
- Google Trends features,
- GDELT coverage/news features,
- Reddit coverage and sentiment features,
- attention-type features,
- lagged alternative-data features,
- combined feature groups.

The feature grid is built once and reused by the model notebooks. This keeps model comparisons consistent across algorithms.

## Train, Validation and Test Split

The project uses chronological splitting. Rows are not shuffled. This is important because the data is time series-like and later observations must not leak into earlier training periods.

The current setup uses:

- training period: early dataset history,
- validation period: middle chronological block,
- test period: final chronological block,
- one-session gaps between split boundaries,
- walk-forward validation inside the training period for feature/model selection.

In the current configuration after excluding `NFLX`, the split is approximately:

| Split | Date range | Modeled rows |
| --- | --- | --- |
| Train | `2021-01-04` to `2023-10-19` | 5,632 |
| Validation | `2023-10-23` to `2024-09-27` | 1,880 |
| Test | `2024-10-01` to `2025-12-31` | 2,504 |

Split logic is implemented in:

```text
notebook_utils/split_utils.py
```

## Notebooks

The main notebooks are located in `notebooks/`.

| Notebook | Purpose |
| --- | --- |
| `feature_exploratory_analysis.ipynb` | Preliminary feature analysis: missing values, feature families, correlations, autocorrelation, target relationships and descriptive plots |
| `logistic_regression.ipynb` | Logistic regression experiments |
| `random_forest.ipynb` | Random forest experiments |
| `xgboost.ipynb` | XGBoost experiments |
| `lightgbm.ipynb` | LightGBM experiments |
| `neutral_net.ipynb` | Feed-forward neural network experiments |
| `svm.ipynb` | Support vector machine experiments |
| `rnn.ipynb` | Recurrent neural network experiments |
| `ticker_attention_diagnostics.ipynb` | Diagnostics for ticker-level attention/alternative-data effects |

The notebooks are designed to use the shared configuration and utility modules instead of duplicating split, feature and metric logic.

## Modeling Workflow

The general modeling workflow is:

1. Load the active experiment configuration from `notebook_utils/experiment_config.py`.
2. Load the active raw panel dataset from `data/datasets/`.
3. Build the feature frame and target variable.
4. Apply chronological train, validation and test splits.
5. Generate feature-set variants with `FeatureSetGridBuilder`.
6. Train candidate models on training data.
7. Select model and feature variants using validation or walk-forward validation results.
8. Refit the selected setup on train plus validation data.
9. Evaluate the final selected model on the held-out test set.
10. Save model reports and plots to notebook output folders.

The shared reporting and metrics code is located in:

```text
notebook_utils/metrics.py
notebook_utils/model_report_builder.py
```

## Environment Setup

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate equity-price-direction-predictor
```

If the environment already exists and dependencies changed, update it with:

```bash
conda env update -f environment.yml --prune
```

The environment uses Python 3.11 and includes the main modeling and analysis libraries:

- pandas and numpy,
- scikit-learn,
- XGBoost and LightGBM,
- TensorFlow and PyTorch,
- matplotlib and seaborn,
- yfinance, pytrends and vaderSentiment,
- transformers for optional FinBERT-related processing.

## Running Data Importers

The importer package can be run from the repository root. Example commands:

```bash
python -m equity_data_importers.run_all
python -m equity_data_importers.run_all reddit_comments
python -m equity_data_importers.run_all --ticker AAPL --company-name Apple --gdelt-query "(Apple OR AAPL)" --trends-query "Apple" --output-tag aapl
```

For normal model experiments, the already prepared dataset in `data/datasets/` is used. Re-running importers is only needed when rebuilding source data.

## Suggested Work Order

For the current thesis experiment, the practical order is:

1. Review or rebuild source imports if needed.
2. Build or verify the rich-price session-aligned dataset in `datamerger/`.
3. Run `notebooks/feature_exploratory_analysis.ipynb` to document preliminary feature behavior.
4. Run model notebooks from `notebooks/` using the same active configuration.
5. Compare saved validation and test reports across models.
6. Use the final held-out test results only for final model assessment.

## Reproducibility Notes

- Run notebooks from the repository root or make sure the root directory is available on the Python path.
- The active experiment is controlled by `notebook_utils/experiment_config.py`.
- Feature-set definitions are controlled by `notebook_utils/feature_set_grid_builder.py`.
- The notebooks should not define separate dataset paths or split rules unless a deliberate new experiment variant is being created.
- Large generated outputs should remain in notebook output folders and do not need to be committed unless they are required for reporting.
- The final test split should be treated as held-out data and should not be used during model or feature selection.
