# Build conda env:
1. conda env create -f environment.yml (if restared required conda env update -f environment.yml --prune)
2. conda activate equity-price-direction-predictor
3. For FinBERT on Windows, keep `pytorch` and `transformers` conda-managed. Avoid mixing pip-installed `torch` into this env.

# Import data
1. python -m equity_data_importers.run_all
2. To run only reddit comments importer: `python -m equity_data_importers.run_all reddit_comments`
3. To run for another stock (example AAPL):
   `python -m equity_data_importers.run_all --ticker AAPL --company-name Apple --gdelt-query "(Apple OR AAPL)" --trends-query "Apple" --output-tag aapl`
4. To run only selected importers for another stock:
   `python -m equity_data_importers.run_all google_trends gdelt stock_price --ticker MSFT --company-name Microsoft --output-tag msft`

# Notes
1. The reddit importer now compares VADER and FinBERT sentiment. The first FinBERT run may download the model weights into the local Hugging Face cache.
2. For non-default profiles, output files are automatically suffixed with `_<output_tag>` (defaults to ticker), so runs for different stocks do not overwrite each other.
3. If Google Trends fails with `Retry.__init__() got an unexpected keyword argument 'method_whitelist'`, fix dependencies in the active env with:
   `pip install "urllib3<2"`
