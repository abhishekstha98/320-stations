# Ablation Study Plan

## Purpose

This document proposes three ablation studies for the thesis project **Correlation-Adaptive Weather Graph Modeling for Complex Terrain**. The goal is to make the thesis stronger by showing which parts of the forecasting pipeline actually matter most to performance.

The plan is aligned with the implemented repository setup:

- 320 weather stations
- STGCN model family
- 43 input lags
- 7-step forecast horizon
- 14 input weather features
- forecast targets: `T2M_MIN`, `RH2M`, and `PRECTOTCORR`

## Common Experimental Protocol

To keep the ablations fair, the following should remain fixed unless the ablation explicitly changes it:

- same station set: 320 stations
- same train, validation, and test split
- same preprocessing and normalization procedure
- same forecast horizon: 7 steps
- same lag window: 43
- same model backbone: `STGCN_Best_BRC` or one clearly declared STGCN configuration
- same optimizer, learning rate policy, batch size, and stopping criteria
- same evaluation metrics for every run

Recommended metrics:

- MAE
- RMSE
- MSE

Recommended reporting views:

- overall average across all 7 forecast steps
- per-target results for `T2M_MIN`, `RH2M`, and `PRECTOTCORR`
- optional per-horizon-step error from step 1 to step 7

## Ablation Study 1: Graph Construction Strategy

### Question

Does the performance gain come from the **correlation-based graph definition** itself, or would a simpler spatial graph perform similarly?

### What changes

Run the same STGCN with different graph definitions:

1. distance-based graph
2. correlation-based graph using absolute Pearson correlation on `T2M`
3. hybrid graph combining distance and correlation

### What stays fixed

- same STGCN architecture
- same features
- same lags and horizon
- same data split

### Why this ablation matters

This is the most important ablation because it directly tests the main thesis claim: that **functional similarity is more useful than physical proximity in complex terrain**.

### Expected interpretation

- If correlation-based performs best, it supports the core thesis argument.
- If hybrid performs best, you can argue that physical and functional relations are complementary.
- If distance-based performs similarly, your conclusion should become more cautious and focus on interpretability or terrain relevance rather than large accuracy gains.

### Suggested thesis sentence

"This ablation isolates the effect of graph topology and tests whether the proposed correlation-adaptive connectivity contributes measurable forecasting benefit beyond conventional spatial proximity."

## Ablation Study 2: Edge Sparsity Ratio

### Question

Is the chosen **top-33% edge retention** actually a good design choice, or is performance sensitive to graph density?

### What changes

Keep the same correlation-based graph logic, but vary the sparsification threshold:

1. top 20% strongest edges
2. top 33% strongest edges
3. top 50% strongest edges
4. optional fully connected weighted graph if computationally feasible

### What stays fixed

- graph signal remains `T2M`
- model architecture unchanged
- feature set unchanged
- same split and training setup

### Why this ablation matters

Your current method uses `desired_ratio = 0.33`. This is a reasonable heuristic, but without an ablation it looks arbitrary. This study gives a justification for that design choice.

### Expected interpretation

- If 33% performs best, it validates the current implementation choice.
- If a sparser graph performs best, it suggests weak correlations add noise.
- If a denser graph performs best, it suggests the model benefits from broader inter-station context.

### Suggested thesis sentence

"This ablation evaluates whether the proposed graph should emphasize only the strongest station relationships or preserve a denser set of interactions."

## Ablation Study 3: Input Feature Contribution

### Question

How much of the forecasting performance comes from the **multivariate feature design**, and which input groups are most useful?

### What changes

Use the same graph and same STGCN, but vary the input feature groups:

1. full 14-feature input
2. thermal-only features
   `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN`
3. thermal + moisture + precipitation
   `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`
4. full input without wind variables

### What stays fixed

- same correlation-based graph
- same sparsity ratio
- same model architecture
- same temporal setup and data split

### Why this ablation matters

This study helps answer whether the model really needs the full meteorological input set, or whether most of the useful information already comes from temperature and moisture signals.

It also strengthens the agricultural relevance of the thesis, because `T2M_MIN`, `RH2M`, and `PRECTOTCORR` are already the output targets and are directly related to crop conditions.

### Expected interpretation

- If full 14-feature input performs best, it supports the richer multivariate design.
- If thermal + moisture performs almost as well, you can argue for a lighter model with lower complexity.
- If removing wind causes little change, wind may be less important for this forecasting task.

### Suggested thesis sentence

"This ablation examines whether the proposed model benefits from broad multivariate weather context or whether a smaller, agriculturally focused feature subset is sufficient."

## Recommended Priority Order

If time is limited, run the ablations in this order:

1. graph construction strategy
2. edge sparsity ratio
3. input feature contribution

This order is recommended because the first ablation tests the main thesis claim most directly.

## Suggested Result Tables

### Table A. Graph construction ablation

| Graph variant | MAE | RMSE | MSE | Best target | Worst target | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Distance-based | - | - | - | - | - | baseline |
| Correlation-based | - | - | - | - | - | proposed |
| Hybrid | - | - | - | - | - | optional |

### Table B. Sparsity ablation

| Edge retention ratio | MAE | RMSE | MSE | Observation |
| --- | --- | --- | --- | --- |
| 20% | - | - | - | - |
| 33% | - | - | - | current design |
| 50% | - | - | - | - |
| 100% | - | - | - | optional |

### Table C. Feature ablation

| Input setting | Number of features | MAE | RMSE | MSE | Observation |
| --- | --- | --- | --- | --- | --- |
| Full input | 14 | - | - | - | - |
| Thermal only | 5 | - | - | - | - |
| Thermal + moisture + precipitation | 7 | - | - | - | - |
| No wind | 10 | - | - | - | - |

## How to Write the Ablation Section in the Thesis

You can organize the section in this order:

1. state that ablation studies were performed to identify the contribution of graph design, graph density, and input variables
2. describe the common controlled setup
3. present Ablation 1 results and interpretation
4. present Ablation 2 results and interpretation
5. present Ablation 3 results and interpretation
6. end with one synthesis paragraph about what components matter most

## Final Thesis Position

If these three ablations are completed, your thesis becomes much stronger because you will be able to show:

- whether correlation-based connectivity is truly useful
- whether the chosen sparsity threshold is justified
- whether the multivariate input design is necessary

That combination gives a cleaner experimental story than only reporting one final model result.
