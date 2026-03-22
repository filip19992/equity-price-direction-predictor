# Build conda env:
1. conda env create -f environment.yml (if restared required conda env update -f environment.yml --prune)
2. conda activate equity-price-direction-predictor

# Import data
1. python -m equity_data_importers.run_all
