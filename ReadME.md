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


# Models training notes

## Summary of Model Selection, Feature Engineering, and Experimental Refinements

### Initial Single-Ticker Phase

The initial phase of the study was conducted on Tesla-only data and focused on repeated experimentation with multiple neural and tree-based model variants. During this stage, strong in-sample performance was frequently observed, while out-of-sample performance remained weak. This pattern indicated that the main limitation was not the absence of a more advanced architecture, but rather a combination of weak predictive signal, limited sample size, and unstable generalization.

The early experiments showed that:
- neural networks often achieved very high training scores but poor test results,
- tree-based models also exhibited substantial overfitting,
- repeated model changes did not materially improve out-of-sample performance.

As a result, the focus was shifted away from architecture search and toward data representation, evaluation design, and feature quality.

### Transition to a Two-Ticker Panel

The analysis was then extended to a joint `TSLA + AAPL` dataset. A shared panel dataset was constructed while maintaining compatibility with the original Tesla file structure. This enabled pooled and ticker-specific experiments to be performed under a common framework.

The move to two tickers allowed:
- comparison of pooled vs separate-per-ticker training,
- comparison of `price_only`, `alt_only`, and `price_plus_alt` feature sets,
- use of common train/test splits and directly comparable evaluation outputs.

Although the two-ticker setting produced numerically better results than the earliest Tesla-only experiments, the improvement was attributed mainly to:
- a simpler binary prediction target,
- a larger pooled sample,
- clearer baselines,
- and a more structured evaluation framework.

No strong evidence was found that the addition of a second ticker alone solved the predictive problem.

### Baseline Model Comparisons

Several baseline comparisons were introduced in order to determine whether alternative data added predictive value over price-derived variables.

Three main feature configurations were tested:
- `price_only`,
- `alt_only`,
- `price_plus_alt`.

Across these baselines, the following pattern emerged:
- `alt_only` was generally weak,
- `price_only` was consistently strong and stable,
- `price_plus_alt` occasionally improved performance, but not uniformly across tickers or models.

This suggested that alternative data, if useful, provided only incremental rather than dominant signal.

### Horizon Comparison

A comparison between `1D` and `5D` prediction horizons was then conducted. It had been hypothesized that alternative data might become more useful over a slightly longer horizon. This expectation was not supported.

The `5D` experiments generally produced:
- lower balanced accuracy,
- weaker discrimination,
- and results closer to random classification.

Consequently, the main focus remained on `1D` directional prediction.

### Introduction of Lagged and Temporal Features

The next stage involved explicit lag construction and rolling transformations. Instead of relying only on current-day alternative data, lagged and rolling variants were introduced.

These included:
- lagged features,
- rolling means,
- rolling gaps,
- surprise-style features relative to recent history.

This change improved results more than introducing more complex neural architectures. It also suggested that alternative data were more likely to matter as delayed or contextual signals than as raw same-session levels. However, broad lagged feature spaces also increased overfitting when the number of derived variables became too large.

### Model Family Comparisons

A wider comparison across model classes was then performed. The following model families were tested:
- logistic regression,
- SVM,
- random forest,
- histogram gradient boosting,
- small MLP.

No single model class consistently dominated across all settings. More complex models did not systematically outperform simpler ones. In particular:
- MLP did not provide a breakthrough,
- tree-based models frequently overfit,
- simpler regularized models often remained competitive.

This shifted the emphasis further toward feature representation and experimental design.

### Strengthening of the Evaluation Framework

To improve methodological rigor, the evaluation framework was extended with:
- source-specific ablation studies,
- threshold tuning,
- bootstrap confidence intervals,
- DeLong AUC comparisons,
- walk-forward fold comparisons.

These additions made the conclusions more defensible and also showed that several initially promising effects were either unstable or not statistically convincing.

### Session Alignment Correction

One of the most important methodological corrections concerned time alignment. In the earlier pipeline, alternative data from weekends and non-trading days were not being aligned optimally with trading sessions. A new `session-aligned` dataset was then constructed so that alternative information was attached to the same or next relevant trading session.

This correction changed the results materially:
- Apple began to show more plausible positive effects from alternative data,
- Tesla remained difficult and often continued to favor `price_only`,
- the importance of temporal alignment became clear.

This was a substantive methodological improvement rather than a cosmetic change.

### Source-by-Source Ablation

Instead of combining all alternative sources into a single large block, each source was then tested separately:
- Google Trends,
- GDELT,
- Reddit sentiment,
- Reddit attention.

This revealed that the sources behaved very differently:
- `GDELT` was usually the weakest source,
- `Google Trends` performed best in small linear setups,
- `Reddit` was the most promising source overall,
- and all effects were strongly ticker-specific.

This stage showed that broad, undifferentiated alt-data combinations were not justified.

### Investigation of Missing Values and Reddit Aggregation

A separate line of investigation examined whether null or zero sentiment values from Reddit were being interpreted incorrectly. Several comparisons were run using:
- missing-aware encodings,
- presence flags,
- controlled missingness variants.

It was found that pure missing-value handling was not the main issue. The more important problem was the premature aggregation of Reddit submissions and comments into a single sentiment signal.

When Reddit submissions and comments were kept separate:
- Apple improved clearly,
- Tesla still did not show consistent benefit.

Additional tests with prior-filled combined signals showed that some improvement over naive aggregation could be achieved, but performance remained worse than when source separation was preserved.

### Alternative Sentiment Representations

Several alternative representations of Reddit sentiment were then examined:
- numeric sentiment features,
- categorical sentiment (`positive`, `neutral`, `negative`),
- more stationary transformations,
- rolling-relative and surprise-based features.

These changes did not produce a universal improvement. Categorical sentiment helped only locally, mainly for Apple in one nonlinear setup. In most cases, richer sentiment representations increased feature count and made overfitting more likely unless strong feature reduction was applied.

### Identification of Feature Count as a Central Issue

At this stage, the main limitation became clearer: feature sets had become too wide relative to the available sample size.

A recurring pattern was observed:
- training performance increased sharply as more alternative features were added,
- test performance improved little or deteriorated,
- train/test gaps became very large.

This indicated that the strongest source of instability was not necessarily model choice, but excessive feature dimensionality.

### Small Selected Feature Sets

A new series of experiments was therefore designed around compact, manually selected feature sets. Instead of large collections of transformations, small sets were used for:
- Google,
- Reddit,
- and Google+Reddit combinations.

This produced one of the strongest improvements in the project. Small hand-picked feature sets performed substantially better than broad alt-data blocks. In particular:
- `ALL + random_forest + small Reddit` outperformed `price_only`,
- `ALL + logreg + small Google+Reddit` also outperformed `price_only`,
- Apple benefited most clearly,
- Tesla improved in some settings but remained mixed.

This strongly suggested that earlier weak results were partly caused by excessive feature count.

### Feature Count Sweeps by Source

A structured sweep was then performed in which the number of features per source was explicitly varied. The best-performing ranges were found to be small:
- approximately `top 4` Reddit features,
- approximately `top 2` Google features.

The main findings were:
- Reddit was the strongest source overall,
- Google was the second strongest,
- GDELT remained the weakest,
- increasing the number of features beyond a small set usually did not help and often harmed generalization.

This stage was crucial in identifying a practical feature-count sweet spot.

### Final Focused Benchmark

A final narrow benchmark was then constructed using only the strongest variants identified in the sweep:
- `logreg: price_top14 vs price14_plus_google_top2`
- `random_forest: price_top14 vs price14_plus_reddit_top4`

This final benchmark showed consistent positive point-estimate improvements for:
- `AAPL`,
- `TSLA`,
- and the pooled `ALL` case.

The strongest final variants were:
- `ALL + logreg + Google top2`,
- `ALL + random_forest + Reddit top4`.

However, two important limitations remained:
- bootstrap confidence intervals still overlapped,
- DeLong p-values did not cross the conventional `0.05` significance threshold.

Therefore, the final evidence was interpreted as promising but not fully conclusive.

## Overall Conclusions

Several broad conclusions emerged from the full experimental process.

First, the largest improvements were not produced by changing model architecture. The most important gains came from:
- correcting session alignment,
- separating alternative data sources,
- avoiding premature aggregation,
- and reducing feature count.

Second, `price_only` remained a strong and often winning benchmark throughout the study. Alternative data did not provide a universal advantage.

Third, the most credible incremental gains came from:
- small `Google Trends` feature sets in linear models,
- small `Reddit` feature sets in tree-based models.

Fourth, the effect of alternative data was clearly ticker-specific:
- Apple showed more consistent benefit from alternative data,
- Tesla remained harder and often continued to favor `price_only`.

Fifth, `GDELT` was the least useful source within the tested setup.

Finally, the main unresolved limitation remained sample size. With only two tickers, the most defensible interpretation is that a weak but plausible incremental signal from carefully selected alternative data was observed, but that no robust and universally generalizable advantage was established.

## Quantitative Addendum for the `0.5%` Benchmark Series

The later benchmark directories make the same story clearer in metric form: the biggest gains came from problem framing, evaluation fixes, and careful feature design, not from simply adding model complexity.

### Best point estimates by stage

| Stage | Best model | Test balanced accuracy | Test accuracy | Main note |
| --- | --- | ---: | ---: | --- |
| `base_model_benchmark_0p5pct` | `logreg` | `0.3723` | `0.3934` | weak 3-class baseline |
| `base_model_benchmark_0p5pct_improved` | `random_forest_ranked` | `0.4618` | `0.4868` | large gain from richer feature engineering |
| `base_model_benchmark_0p5pct_v2` | `extra_trees_reg_leaf5_mf035` | `0.4961` | `0.5060` | better regularization, still 3-class |
| `base_model_benchmark_binary_0p5pct` | `random_forest_enriched` | `0.5485` | `0.5503` | biggest jump came from the binary target |
| `base_model_benchmark_binary_0p5pct_v4` | `random_forest_enriched_short_backward_selected` | `0.5593` | `0.5594` | best result in the core `v2-v10` single benchmark line |
| `base_model_benchmark_binary_0p5pct_v10` | `random_forest_requested_features_tuned` | `0.5515` | `0.5585` | near-peak result with only `13` numeric features |
| `base_model_benchmark_binary_0p5pct_nine_vs_no_nflx_vader_nullaware` | `extra_trees_requested_features_tuned` | `0.5564` | `0.5577` | best of the `nine_vs_no_nflx*` comparisons |
| `base_model_benchmark_binary_0p5pct_eight_tickers_vader_nullaware_methodology_fixed` | `extra_trees_requested_features_tuned` | `0.5634` | `0.5624` | methodology correction added another step up |
| `base_model_benchmark_binary_0p5pct_eight_tickers_vader_nullaware_ensemble_routing` | `ensemble_et_rf_equal` | `0.5675` | `0.5647` | simple ET/RF ensemble beat either base learner |
| `base_model_benchmark_binary_0p5pct_eight_tickers_vader_nullaware_external_lag_sweep` | `gdelt_lag_5__google_lag_5` | `0.5844` | `0.5774` | best overall point estimate in these directories |

### What helped

- Better feature engineering in the original `0.5%` task helped a lot: best test balanced accuracy improved from `0.3723` to `0.4618`, then to `0.4961`.
- Switching from the 3-class setup to the binary `0.5%` target was the single biggest step change: `0.4961 -> 0.5485`.
- Moderate feature pruning helped in the main binary line: `v2` reached `0.5548`, while backward selection in `v4` improved that to `0.5593`.
- The `eight_tickers_no_nflx + vader + nullaware` setup was clearly stronger than the earlier scenario variants: best test balanced accuracy reached `0.5564`, versus `0.5436` in the plain `nine_tickers_all` setup.
- The methodology fix mattered: `0.5564 -> 0.5634`.
- A simple equal-weight ET/RF ensemble helped a bit more: `0.5634 -> 0.5675` (`+0.0041` balanced accuracy over the methodology-fixed ET baseline).
- External lag features helped the most in the late-stage eight-ticker setup. In the lag sweep, `gdelt_lag_5 + google_lag_5` reached `0.5844` balanced accuracy and improved by `+0.0169` over the no-lag external baseline used inside that sweep.
- `v10` is a useful practical compromise: it did not beat `v4`, but it recovered `0.5515` balanced accuracy with only `13` numeric features instead of `126-127`.

### What did not help

- Per-ticker training in `base_model_benchmark_binary_0p5pct_v7` was a clear regression: best test balanced accuracy fell to `0.5027`, almost back to dummy level.
- The very compact `12`-feature designs in `v8` and `v9` underfit. Their best test balanced accuracy values were only `0.5167` and `0.5255`.
- Adding FinBERT on top of the `vader_nullaware` scenario did not help. Best test balanced accuracy dropped from `0.5564` to `0.5474`.
- The extra Reddit VADER lag sweep did not beat the simpler external lag sweep. Its best result was `0.5780`, below the `0.5844` from the plain external lag sweep.
- Stacking did not improve on the simple ET/RF ensemble. The best rows in `stacking`, `stacking_oof_train`, and `ensemble_adjustments` all stayed at `0.5675` or matched the same ensemble point estimate rather than surpassing it.
- The feature sweet-spot rerun did not beat the current full null-aware setup. The reduced variants topped out at `0.5441` balanced accuracy (`price_plus_reddit_activity4`), while `current_nullaware_full` remained better at `0.5564`.
- The `5D` horizon still looked weaker than the best `1D` runs. In `base_model_benchmark_binary_5d_band_sweep_v6`, the best non-dummy point estimate was `0.5480` balanced accuracy at the `2.0%` neutral band, which remained below the best `1D` binary benchmark (`0.5593` in `v4`).

### Validation-first follow-up around the frozen external baseline

After freezing the leakage-safe base model selected by validation balanced accuracy, a set of targeted follow-up checks was run around:

- `requested_momentum_volume_sentiment__gdelt_lag_5__google_lag_1`
- validation balanced accuracy: `0.5154`
- test balanced accuracy: `0.5633`

The main follow-up findings were:

- Adding external summary or delta features did not beat the frozen baseline on validation.
  - `+ gdelt_mean_lag_1_3` reached `0.5120` validation balanced accuracy.
  - `+ gdelt_delta_1` reached `0.5118`.
  - `+ google_delta_1` reached `0.5081`.
  - `+ gdelt_mean_lag_1_5` reached only `0.5058` on validation, even though its test balanced accuracy rose to `0.5785`.
  - `+ google_mean_lag_1_2` was weakest at `0.4992`.
- Adding short Reddit post-count lags on top of the baseline also did not help on validation.
  - `subm_reddit_posts_lag_1..2` reached `0.5124`.
  - `comm_reddit_posts_lag_1..2` reached `0.5060`.
  - adding both submission and comment post-count lags reached `0.5044`.
  - These variants often looked better on test than on validation, so they were not selected.
- Replacing the current raw Reddit post-count features worked better than simply adding lagged counts.
  - Replacing `subm_reddit_posts` and `comm_reddit_posts` with `subm_reddit_has_posts_lag_1..2` and `comm_reddit_has_posts_lag_1..2` improved validation balanced accuracy to `0.5191` (`+0.0037` vs the frozen baseline).
  - The same candidate had test balanced accuracy `0.5607`, slightly below the baseline `0.5633`.
  - Replacing the current raw post counts with lagged raw post counts did not win on validation: the best such replacement reached `0.5115`.

So the follow-up around the frozen baseline suggests:

- raw external summary/delta additions are not justified by validation,
- simple Reddit lag additions are not justified by validation,
- but replacing current Reddit post-count features with lagged binary `has_posts` indicators for both submissions and comments is the strongest validation-selected refinement tested so far.

### Best models to carry forward

If the goal is the strongest raw test point estimate from these runs, the best candidate is:

- `base_model_benchmark_binary_0p5pct_eight_tickers_vader_nullaware_external_lag_sweep`
  - model: `gdelt_lag_5__google_lag_5`
  - test balanced accuracy: `0.5844`
  - test accuracy: `0.5774`
  - test macro F1: `0.5769`

If the goal is a simpler and more reproducible benchmark line, the strongest choices are:

- `base_model_benchmark_binary_0p5pct_v4` for the best result in the main binary progression (`0.5593` balanced accuracy),
- `base_model_benchmark_binary_0p5pct_v10` for a much smaller feature set with only a small loss (`0.5515` balanced accuracy),
- `base_model_benchmark_binary_0p5pct_eight_tickers_vader_nullaware_ensemble_routing` for the best non-lagged eight-ticker null-aware ensemble (`0.5675` balanced accuracy),
- the frozen validation-selected external baseline `gdelt_lag_5 + google_lag_1` as the clean reproducible starting point for local refinement,
- and the follow-up replacement candidate that swaps current `subm_reddit_posts` and `comm_reddit_posts` for lagged `has_posts` features `1..2`, because it is the strongest validation-selected refinement tested around that frozen baseline (`0.5191` validation balanced accuracy vs `0.5154` for the original frozen base).
