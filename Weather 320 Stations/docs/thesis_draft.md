# Correlation-Adaptive Weather Graph Modeling for Complex Terrain

## Abstract

This draft consolidates the thesis materials that are available in the repository into a single evidence-based manuscript. The central problem is local weather forecasting in complex terrain, where stations that are geographically close may be separated by strong topographic barriers and therefore may not share similar weather behavior. The statement of problem is grounded in the recent Nepal-focused study by Bhandari, Pandeya, Jha, and Jha (2024), which explores graph neural networks for weather prediction in data-scarce environments. The implemented pipeline addresses this problem by constructing a weather graph from absolute Pearson correlation between station-level temperature series rather than from physical proximity alone. In the 320-station code path, graph construction is performed from station `T2M` histories, the graph is sparsified by retaining the top 33% of pairwise correlations, and the resulting graph is used by a spatio-temporal graph convolutional network (STGCN) to forecast `T2M_MIN`, `RH2M`, and `PRECTOTCORR` over a seven-day horizon from 43 input lags.

The available non-PDF materials support the forecasting pipeline, the graph-construction logic, the training configuration, logged training behavior, and an example inference output. They do not support a completed 320-station baseline comparison, a validated crop-recommendation module, or field-level agronomic evaluation. Accordingly, this draft presents the implemented forecasting system as the verified contribution, treats crop recommendation as downstream motivation rather than completed experimental evidence, and records the main reproducibility and validity constraints that remain visible in the repository.

## Introduction

Accurate local weather estimation is a persistent problem in mountainous regions. In plains or gently varying terrain, spatial smoothness is a reasonable first approximation: nearby locations often share similar temperature, humidity, and rainfall behavior. In steep terrain, this assumption weakens. Valleys, ridges, slope orientation, and localized circulation can produce neighboring sites with markedly different weather conditions. When forecasting models rely too heavily on geographic closeness, they risk connecting stations that are near in map distance but dissimilar in actual meteorological behavior.

Bhandari et al. (2024) provide the recent published foundation for this problem in the Nepal context. Their work uses graph-based learning for weather prediction and imputation in data-scarce environments, showing that graph neural networks are a relevant modeling family for Nepal's diverse geography and incomplete station records. The present thesis uses that paper as a recent reference and literature benchmark, then narrows the research question to the graph-construction choice within a 320-station STGCN forecasting pipeline.

The thesis materials in this repository frame that issue as the core motivation for a correlation-adaptive weather graph. The intended argument is that station connectivity should be inferred from shared signal behavior rather than only from geodesic distance. The implementation supports that argument in one specific form: a graph is built from the absolute Pearson correlation of station `T2M` time series, then used inside an STGCN to forecast three weather variables that are directly relevant to downstream agricultural decision support: minimum temperature, relative humidity, and corrected precipitation.

The broader project motivation is crop recommendation in Nepal and similarly rugged regions. That motivation remains important, but the available repository evidence shows that the implemented and logged contribution is the weather-forecasting backbone rather than an end-to-end crop advisory system. For that reason, this draft is intentionally narrower than some of the support notes. It documents the forecasting methodology that is visible in code and data artifacts, records the available training and inference evidence, and distinguishes verified implementation details from narrative ambitions that are not yet substantiated by the non-PDF materials.

## Problem Statement and Scope

The problem addressed here is station-level weather forecasting over complex terrain using a graph representation that better reflects functional similarity between stations. The implemented pipeline assumes that historical temperature co-variation can serve as a proxy for broader weather similarity and therefore for graph connectivity. In formal terms, the goal is to replace a strictly proximity-driven neighborhood definition with a graph in which strong edges connect stations whose historical `T2M` series behave similarly over time.

The scope of the implemented work is narrower than a full agricultural decision system. The repository supports the following claims:

- a 320-station weather graph can be constructed from correlation over station `T2M` histories;
- a three-block STGCN can be trained over graph-structured temporal weather inputs;
- the active 320-station model is configured to predict `T2M_MIN`, `RH2M`, and `PRECTOTCORR`;
- logged training traces and an example inference output are available in the repository.

The repository does not support several stronger claims that appear in the support notes. There is no documented 320-station baseline experiment with comparable logged metrics, no implemented crop-recommendation engine, no yield-validation workflow, and no editable original thesis manuscript that would allow direct reconciliation against prior prose chapter by chapter. This draft therefore treats completed forecasting implementation as the primary contribution and frames crop recommendation as a plausible downstream application rather than a finished experimental outcome.

For baseline terminology, this thesis separates two ideas. The **direct experimental baseline** is a distance-based STGCN trained and evaluated under the same 320-station protocol as the proposed correlation-based STGCN. The **literature benchmark** is Bhandari et al. (2024), whose GraphSAGE/GNN-based weather imputation study is recent and Nepal-specific but not directly equivalent unless reimplemented on the same data, targets, and metrics.

## Dataset and Study Inputs

The repository contains several weather-related data files with distinct roles, and the thesis needs to distinguish them carefully.

The largest archive is `Weather_320_Stations_DB.csv`. Its visible file boundaries show records beginning on `1981-01-04` and extending at least to `2023-11-24`. The file uses the columns `Date`, `Location`, `Latitude`, `Longitude`, `Altitude`, `T2M`, `T2MWET`, `TS`, `T2M_RANGE`, `T2M_MAX`, `T2M_MIN`, `QV2M`, `RH2M`, `PRECTOTCORR`, `PS`, `WS10M`, `WS10M_MAX`, `WS10M_MIN`, `WS10M_RANGE`, `WS50M`, `WS50M_MAX`, `WS50M_MIN`, and `WS50M_RANGE`. This is the only repository file that is visibly consistent with the correlation graph logic, because that logic requires multi-date station histories rather than a single observation per station.

The file `Stations_320_Final.csv` is different. In the repository snapshot used for this revision, it contains 320 rows, one row per station, all for a single date. That makes it useful as a station-aligned snapshot, but not sufficient by itself for time-series correlation analysis. This distinction matters because the active graph-construction script points to `Stations_320_Final.csv`, while its internal logic assumes a multi-date table that can be pivoted into a `Date x Station` matrix. The method itself is coherent, but the default file path and the visible contents of that file do not fully align. This is therefore a reproducibility issue, not merely a documentation detail.

The repository also includes an explicit test bundle under `GNN TRAIN TEST/TEST`. That bundle contains:

- `test_data.csv` with 14,400 rows corresponding to 320 stations over 45 dates from `2023-12-12` through `2024-1-9`;
- `locations.csv` with station coordinates and altitude;
- `stations.txt` with the station ordering;
- `mean_values.csv` and `std_values.csv` for feature normalization;
- `edge_index.pt` and `edge_weights.pt` for the graph structure;
- model checkpoints and logged outputs in `NW_WT`.

The active 320-station preprocessing and inference code uses a 14-feature input vector:

| Category | Variables |
| --- | --- |
| Thermodynamic inputs | `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN` |
| Moisture and precipitation inputs | `RH2M`, `PRECTOTCORR` |
| Pressure input | `PS` |
| Wind inputs | `WS10M`, `WS10M_MAX`, `WS10M_MIN`, `WS50M`, `WS50M_MAX`, `WS50M_MIN` |

The implemented forecast targets are a three-variable subset of that feature set: `T2M_MIN`, `RH2M`, and `PRECTOTCORR`. The target choice is explicit in the inference script and is more reliable than the broader variable descriptions found in the support notes, which are inconsistent across files.

### Table 1. Dataset, feature, and graph settings visible in the repository

| Item | Repository evidence | Evidence-based interpretation |
| --- | --- | --- |
| Station set | `Stations_320_Final.csv`, `processed_data/stations.txt`, `TEST/stations.txt` | The implemented 320-station pipeline is organized around 320 named locations. |
| Historical archive | `Weather_320_Stations_DB.csv` | Multi-date weather archive likely intended for temporal graph construction. |
| Snapshot file | `Stations_320_Final.csv` | One row per station for a single visible date; not sufficient alone for correlation-over-time computation. |
| Test bundle | `GNN TRAIN TEST/TEST/test_data.csv` | 45 days of inputs for 320 stations, used for inference preparation. |
| Input features | `Final_Preprocess_Global.py`, `Main_Test.py` | The active 320-station path uses 14 weather variables. |
| Forecast targets | `Main_Test.py` | The model predicts `T2M_MIN`, `RH2M`, and `PRECTOTCORR`. |
| Graph signal used for edge weights | `Final_Preprocess_Global.py` | Graph weights are computed from the absolute Pearson correlation of station `T2M` series. |
| Sparsification rule | `Final_Preprocess_Global.py` | The top 33% of pairwise correlations are retained. |
| Node metadata | `Final_Preprocess_Global.py`, `locations.csv` | Latitude, longitude, and altitude are stored, but the active 320-station graph weights are not directly defined by them. |

## Correlation-Based Graph Construction

The implemented graph-construction logic appears in `GNN TRAIN TEST/Final_Preprocess_Global.py`. The method constructs an undirected station graph from temperature co-variation rather than spatial distance. Let the station set be \(V = \{v_1, v_2, \ldots, v_n\}\), with \(n = 320\), and let \(x_i\) denote the historical `T2M` series for station \(v_i\). The script pivots the source table into a `Date x Station` matrix and computes the absolute Pearson correlation between station columns:

\[
w_{ij} = \left| \rho(x_i, x_j) \right|,
\]

where

\[
\rho(x_i, x_j) =
\frac{\sum_t (x_{i,t} - \bar{x}_i)(x_{j,t} - \bar{x}_j)}
{\sqrt{\sum_t (x_{i,t} - \bar{x}_i)^2}\sqrt{\sum_t (x_{j,t} - \bar{x}_j)^2}}.
\]

Using the absolute value means that strong anti-correlation is treated as a strong relationship magnitude, not as the absence of a relationship. After computing all pairwise weights, the script applies percentile-based sparsification and retains only the top 33% of weights. The graph is then symmetrized by adding reversed edges, and self-loops of weight 1 are inserted for the active nodes.

This design encodes a clear methodological position: station connectivity is defined by historical signal similarity. In the active 320-station code path, altitude, latitude, and longitude are preserved as node metadata but do not directly determine the graph weights. That is an important difference from the older location-driven preprocessing script under `Weather_Forecast750_Station_HarishSir/preprocessing.py`, where weights are built from altitude similarity and short-range geodesic rules. The older script is useful as historical context for the project's conceptual evolution, but it does not constitute a documented 320-station baseline experiment in the present repository state.

From a thesis-writing perspective, the main scientific contribution supported by the code is therefore the shift from a location-centered graph heuristic to a time-series correlation graph. The repository does not currently support stronger claims, such as quantified superiority over a matched baseline, but it does support a precise formal description of the implemented graph definition.

## STGCN Architecture

The active 320-station training script is `GNN TRAIN TEST/Train_Global.py`. Several model classes appear in that file, but the training block instantiates `STGCN_Best_BRC`, so that is the architecture that should be treated as operative in this manuscript.

The model receives graph-structured temporal inputs with 320 nodes and 14 features per node. Its architecture is composed of three ST-Conv blocks followed by a linear output head:

1. `STConv(320, 14, 64, 128, 9, 4)`
2. `STConv(320, 128, 256, 64, 7, 4)`
3. `STConv(320, 64, 32, 16, 5, 3)`
4. `Linear(16, 3)`

The temporal kernels decrease from 9 to 7 to 5 across the network, while the graph convolution orders are 4, 4, and 3 respectively. The final linear layer maps the learned hidden representation to three outputs corresponding to `T2M_MIN`, `RH2M`, and `PRECTOTCORR`.

The support notes in the repository describe the model more loosely, sometimes as a two-block architecture and sometimes with different output variables. Those descriptions are less reliable than the training script itself. In an evidence-first thesis, the model section should therefore follow the instantiated code rather than the earlier summary prose.

The data loader used during training builds temporal samples from 43-lag windows and a seven-step forecast horizon. This is also the configuration reflected in the inference script. Within the active 320-station code path, normalization is performed using global mean and standard deviation values over the 14 selected input features, as written by the preprocessing utilities and re-used during test-time normalization.

## Training and Inference Procedure

The repository shows a partially serialized training workflow. `Train_Global.py` expects precomputed graph tensors and temporal snapshots loaded from absolute Windows paths under `D:/ICIMOD/YASH_PROCESS/`. Although the raw `snapshots.npy` file is not included in the visible repository tree, the script still documents the intended training procedure.

The implemented configuration is as follows:

- temporal context window: 43 lags;
- forecast horizon: 7 output steps;
- optimizer: Adam;
- initial learning rate: 0.05;
- batch size: 32;
- maximum epochs: 300;
- train-validation split: 85% / 15% using `random_split`;
- early stopping patience: 15 epochs;
- learning-rate reduction: factor of 10 when the no-improvement counter reaches half the patience threshold.

The training code uses `random_split` on serialized temporal graph windows rather than an explicit chronological holdout. That choice is visible in the script and should be documented because it affects how validation results are interpreted. For time-series forecasting, randomly splitting overlapping temporal windows can yield a more optimistic validation estimate than a strictly chronological evaluation would.

The inference workflow in `GNN TRAIN TEST/Main_Test.py` is also well defined. It reads `test_data.csv`, normalizes the 14 features using the saved mean and standard deviation files, constructs the last 43-lag input tensor, loads graph tensors from the test bundle, and runs the model in evaluation mode. The script then de-normalizes the three predicted targets and selects the station nearest to a user-supplied coordinate. In the visible example, the coordinate is `[28.6616, 80.6392]`. This design implies a pragmatic application layer: the model forecasts at station nodes, and the external query location is mapped to the nearest available station rather than handled by a separate spatial interpolation step.

One further detail deserves explicit mention. The current inference script points to `Model_43Lags_STConv_Last.pth`, while the accompanying `Test Results.txt` reports outputs for both a best-weight and a last-weight checkpoint. The repository therefore contains usable inference evidence, but the active script and the logged comparison are not fully harmonized.

### Table 2. Training, inference, and logged outputs supported by the repository

| Item | Repository evidence | Evidence-based reading |
| --- | --- | --- |
| Active training model | `Train_Global.py` | `STGCN_Best_BRC` is the instantiated model for the 320-station run. |
| Input window | `Train_Global.py`, `Main_Test.py` | 43 lags are used as model input. |
| Forecast horizon | `Train_Global.py`, `Main_Test.py` | The model is configured for 7 output steps. |
| Training optimizer | `Train_Global.py` | Adam with initial learning rate `0.05`. |
| Train-validation split | `Train_Global.py` | 85/15 split via `random_split`. |
| Batch size | `Train_Global.py` | 32. |
| Early stopping | `Train_Global.py`, `TEST/NW_WT/History.txt` | Patience of 15 epochs; training stops when no improvement persists. |
| Logged validation behavior | `TEST/NW_WT/History.txt` | Validation loss falls from `0.6252009707528192` at epoch 1 to a saved-best checkpoint at epoch index 170, corresponding to printed epoch 171 with validation loss `0.024574211521728617`. |
| Logged stopping point | `TEST/NW_WT/History.txt` | Training halts after printed epoch 185 because of early stopping. |
| Logged runtime | `TEST/NW_WT/History.txt` | Total training time is reported as `469494.870601` seconds. |
| Example inference bundle | `TEST/Test Results.txt`, `Main_Test.py` | The repository contains a 7-step sample forecast for a query coordinate, but no ground-truth comparison for those predictions. |

## Available Experimental Evidence

The strongest experimental evidence available in the non-PDF repository is the training history under `GNN TRAIN TEST/TEST/NW_WT/History.txt`. That log shows consistent reduction in both training and validation loss over the course of training. The first printed epoch reports a training loss of `0.6457320012205796` and a validation loss of `0.6252009707528192`. Over time, the logged validation loss decreases substantially, with repeated checkpoint saves indicating successive improvements. The final saved-best message occurs at epoch index 170, which corresponds to the printed epoch 171 and a validation loss of `0.024574211521728617`.

This trajectory indicates that the implemented model can fit the training setup provided by the repository and that the optimization process converges to a substantially lower validation loss than at initialization. The log also shows several learning-rate reductions and eventual early stopping after printed epoch 185. In that limited sense, the repository supports a convergence claim for the configured training procedure.

The second available evidence source is `GNN TRAIN TEST/TEST/Test Results.txt`, which records example seven-step predictions for both a best-weight and a last-weight checkpoint. For the best-weight output, the predicted `T2M_MIN` values range from approximately `9.4428` to `11.9839`, `RH2M` remains near `83.7` to `84.2`, and `PRECTOTCORR` is zero for most forecast steps with one non-zero value of `1.4388`. The last-weight checkpoint produces a similar pattern, though not identical values. These outputs show that the inference pipeline is operational and that the repository preserves concrete forecast outputs from trained weights.

What the non-PDF materials do not provide is equally important. There is no logged 320-station baseline experiment against which the correlation graph can be quantitatively compared. There is no test-set accuracy table with MSE, MAE, or RMSE by target variable. There is no district-level evaluation table, no ablation study for sparsity thresholds, and no external validation of the sample forecasts. As a result, the available evidence supports statements about implementation, training convergence, and example inference behavior, but not strong claims about comparative forecasting superiority.

The repository also includes figure assets that can support a limited manuscript presentation. Under `LOCATION PLOT`, there are multiple station-map images and location files. Under `GNN TRAIN TEST/TEST/NW_WT`, there are loss-curve and learning-curve images. These can be used as descriptive figures for station coverage and training behavior, provided that captions remain literal and do not overstate what the figures prove.

## Limitations and Threats to Validity

Several limitations are visible directly in the repository and should be stated openly in the thesis.

First, there is no editable original thesis source, no chapter-level LaTeX, and no actionable written reviewer feedback in the non-PDF materials. This limits direct reconciliation against prior manuscript text and makes the present draft necessarily reconstructive.

Second, the baseline comparison is incomplete. The repository contains an older location-driven graph-construction path for a 750-station setting, but it does not contain a clearly logged, directly comparable 320-station baseline run using the same data preparation and evaluation protocol. Any claim that the correlation graph outperforms a distance-based graph for the 320-station experiment would therefore exceed the currently available evidence.

Third, the crop-recommendation narrative is not implemented end to end in the visible code. The forecasting outputs are agriculturally relevant, but there is no repository evidence of a crop-suitability engine, rule base, yield benchmark, or field validation. Crop recommendation should therefore be presented as downstream motivation and future integration, not as a completed evaluated subsystem.

Fourth, reproducibility is reduced by environment coupling. The training and inference scripts use hard-coded absolute Windows paths such as `D:/ICIMOD/YASH_PROCESS/...` and `E:/WEATHER_FORECAST/...`, and some required intermediate artifacts, especially serialized training snapshots, are not included in the visible tree. The code documents the intended workflow, but not all assets needed to rerun it are present in a portable form.

Fifth, the graph-construction script and the visible repository files are not perfectly aligned. The correlation method requires multi-date station histories, yet the script's default input path references `Stations_320_Final.csv`, which appears in this repository as a single-date snapshot. The presence of `Weather_320_Stations_DB.csv` suggests the intended historical source, but the path mismatch remains a material reproducibility concern.

Sixth, the validation protocol in the visible training script uses `random_split` over temporal graph windows. Because neighboring windows may share overlapping temporal context, this choice may weaken the independence of the validation estimate relative to a chronological split.

Seventh, the test bundle itself shows evidence of repeated station-day profiles. During this repository audit, the dates `2023-12-12` and `2024-1-9` each yielded only 20 unique tuples across 320 stations when comparing `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`, and `PS`. This pattern does not invalidate the forecasting code, but it does limit how strongly one can interpret fine-grained microclimate variation from the released test subset alone.

Finally, the example inference outputs are forecasts without an accompanying ground-truth comparison in the same file. They demonstrate operational inference, not verified predictive accuracy on a held-out future horizon.

## Agricultural Relevance and Future Integration

Even within a strict evidence-first reading, the agricultural motivation of the project remains credible. Minimum temperature, relative humidity, and precipitation are central variables in crop planning, crop stress, and field-level advisory systems. A weather graph that captures functional similarity between stations could therefore serve as a stronger upstream input layer than a graph that assumes geographic closeness is always informative.

What the repository currently supports is the forecasting backbone for that idea. The inference design also hints at a plausible deployment path: a user-supplied coordinate is linked to the nearest station in the modeled network, and the model returns a seven-step weather forecast for the corresponding node. A downstream crop-recommendation system could, in principle, consume those forecasts together with crop thresholds, soil constraints, and seasonal agronomic rules.

That said, such a downstream stage remains future work in the current repository state. A complete next step would require at least three additions: a transparent crop-suitability model, a documented decision rule linking forecast variables to crop advice, and an evaluation framework that compares advisory outcomes against agronomic references or observed outcomes. Until those components are implemented and logged, the thesis should describe the contribution as enabling infrastructure for crop recommendation rather than as a completed recommendation engine.

## Conclusion

The available non-PDF materials support a clear and technically meaningful thesis contribution: a 320-station weather-forecasting pipeline for complex terrain in which station connectivity is defined by absolute Pearson correlation of historical temperature signals and forecasting is performed by a graph-based spatio-temporal neural network. The implemented method is specific, formalizable, and grounded in the visible code. The repository also preserves concrete evidence of training convergence and sample inference behavior.

At the same time, an evidence-first reading requires a narrower and more careful conclusion than the support notes sometimes suggest. The current materials do not substantiate a completed baseline comparison, they do not demonstrate an implemented crop-recommendation module, and they do not yet provide the kind of evaluation tables that would justify broad performance claims. The most defensible thesis position is therefore that the repository documents a promising and partially validated forecasting framework whose core methodological idea is visible and coherent, while its comparative evaluation and agricultural integration remain incomplete.

## References

1. Bhandari, H. C., Pandeya, Y. R., Jha, K., & Jha, S. (2024). *Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments.* Environmental Research Communications, 6, 105010. https://doi.org/10.1088/2515-7620/ad8108
2. Rasp, S., Hoyer, S., Merose, A., Langmore, I., Battaglia, P., Russell, T., et al. (2024). *WeatherBench 2: A benchmark for the next generation of data-driven global weather models.* Journal of Advances in Modeling Earth Systems, 16, e2023MS004019. https://doi.org/10.1029/2023MS004019
3. Kochkov, D., Yuval, J., Langmore, I., Norgaard, P., Smith, J., Mooers, G., et al. (2024). *Neural general circulation models for weather and climate.* Nature, 632, 1060-1066. https://doi.org/10.1038/s41586-024-07744-y
4. Pathak, J., Cohen, Y., Garg, P., Harrington, P., Brenowitz, N., Durran, D., et al. (2024). *Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling.* arXiv:2408.10958.
5. Kazadi, A., Doss-Gollin, J., Sebastian, A., and Silva, A. (2024). *FloodGNN-GRU: a spatio-temporal graph neural network for flood prediction.* Environmental Data Science, 3, e21. https://doi.org/10.1017/eds.2024.19
6. Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T. R., El-Kadi, A., Masters, D., et al. (2025). *Probabilistic weather forecasting with machine learning.* Nature, 637, 84-90. https://doi.org/10.1038/s41586-024-08252-9
7. Sun, X., Li, J., Zhao, Z., Jing, G., Chen, B., Hu, J., Wang, F., and Zhang, Y. (2025). *Utility of Graph Neural Networks in Short-to Medium-Range Weather Forecasting.* Computers, Materials and Continua, 84(2), 2121-2149. https://doi.org/10.32604/cmc.2025.063373
