# Build conda env:
1. conda env create -f environment.yml (if restared required conda env update -f environment.yml --prune)
2. conda activate equity-price-direction-predictor
3. For FinBERT on Windows, keep `pytorch` and `transformers` conda-managed. Avoid mixing pip-installed `torch` into this env.

# Import data
1. python -m equity_data_importers.run_all
2. To run only reddit comments importer: `python -m equity_data_importers.run_all reddit_comments`

# Notes
1. The reddit importer now compares VADER and FinBERT sentiment. The first FinBERT run may download the model weights into the local Hugging Face cache.
