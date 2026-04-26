# Proposal Defense Questions and Answers

## How to Use This File

This file is tailored to the updated proposal defense presentation in `Proposal Defense.pptx`.

Each question includes:

- **Short answer**: a quick response for viva or slide-by-slide defense
- **Expanded answer**: a slightly fuller version for follow-up questions

The answers are written to stay consistent with the revised proposal and the current evidence-based thesis framing:

- the statement of problem is grounded in the recent Nepal-focused paper by **Bhandari et al. (2024)**
- the core contribution is the **correlation-adaptive weather forecasting backbone**
- crop recommendation is treated as **future integration / downstream application**
- comparative claims are framed carefully unless fully supported by evidence

---

## 1. Opening and Overview

### Q1. What is your proposed research about?
**Short answer:**  
My proposed research is about improving weather forecasting in complex mountainous terrain by connecting weather stations using historical signal similarity instead of only geographic distance.

**Expanded answer:**  
The research proposes a correlation-adaptive weather graph for 320 stations and uses that graph inside a Spatio-Temporal Graph Convolutional Network, or STGCN. The goal is to produce more meaningful local forecasts in terrain where nearby stations may behave very differently because of topography.

### Q2. What is the title of your proposal?
**Short answer:**  
The title is **Correlation-Adaptive Weather Graph Modeling for Precision Crop Recommendation in Complex Mountain Terrain**.

**Expanded answer:**  
The title reflects two layers of the work. The immediate technical contribution is correlation-adaptive weather graph modeling, and the broader motivation is to support precision agriculture and future crop advisory systems in mountainous regions.

### Q3. What is the main idea of your proposal in one sentence?
**Short answer:**  
The main idea is that weather stations in mountains should be connected by meteorological similarity, not by map distance alone.

**Expanded answer:**  
Traditional distance-based graphs assume nearby stations are similar, but this is often false in complex terrain. My proposal replaces that assumption with a graph built from historical correlation between station time series.

### Q4. Why is this topic important?
**Short answer:**  
It is important because local weather strongly affects agriculture, planning, and risk management, especially in complex terrain where microclimates are common.

**Expanded answer:**  
In Nepal and similar regions, temperature, humidity, and rainfall can change sharply over short distances. If local weather is poorly estimated, then downstream decisions such as planting, irrigation, and hazard preparedness also become less reliable.

### Q4a. Which recent paper is used as the foundation for your problem statement?
**Short answer:**  
I use Bhandari et al. (2024), published in *Environmental Research Communications*, as the recent Nepal-specific foundation.

**Expanded answer:**  
The paper is titled **"Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments."** It is suitable because it is recent, Nepal-focused, and directly discusses graph neural networks for weather prediction and missing atmospheric variables in data-scarce environments. My work builds on that direction but extends the framing to a 320-station STGCN forecasting setup.

---

## 2. Background and Motivation

### Q5. Why is weather forecasting difficult in mountainous terrain?
**Short answer:**  
Because mountains create microclimates through elevation differences, ridges, valleys, slope aspect, and rain-shadow effects.

**Expanded answer:**  
Two stations that are close on a map may still experience very different temperatures, humidity, and rainfall because of terrain barriers and local circulation. That makes simple spatial smoothness assumptions unreliable.

### Q6. What do you mean by the “distance trap”?
**Short answer:**  
The distance trap is the problem of assuming that geographically close stations are always meteorologically similar.

**Expanded answer:**  
In flat regions that assumption may work reasonably well, but in mountains it can be misleading. A ridge can separate two nearby stations into very different weather regimes, while two distant valleys may still behave similarly.

### Q7. Why is geographic distance alone not enough in your study?
**Short answer:**  
Because geographic closeness does not always reflect atmospheric similarity in complex terrain.

**Expanded answer:**  
A distance-based graph can force the model to aggregate information from physically close but behaviorally dissimilar stations. That can introduce noise into forecasting rather than useful spatial context.

### Q8. Why is Nepal or a similar mountainous setting a good study context?
**Short answer:**  
It is a strong study context because terrain variation is high and the weakness of distance-based assumptions becomes very visible.

**Expanded answer:**  
Nepal contains large elevation changes over short horizontal distances, so it is an ideal environment for testing whether functional connectivity is more meaningful than physical proximity. If the method works in such terrain, it has relevance for other mountainous regions too.

---

## 3. Research Gap and Objectives

### Q9. What research gap are you addressing?
**Short answer:**  
The gap is the lack of a transparent, correlation-driven graph construction strategy for weather forecasting in complex terrain.

**Expanded answer:**  
Many spatio-temporal graph forecasting methods still use fixed distance-based adjacency. My proposal addresses the need for a graph that reflects actual historical station behavior rather than only physical closeness.

### Q10. What is your general objective?
**Short answer:**  
My general objective is to design and evaluate a weather forecasting framework for complex terrain using correlation-defined station connectivity.

**Expanded answer:**  
The proposal aims to improve how spatial relationships are represented inside a forecasting model. Instead of starting from geographic distance, I define connectivity from historical signal similarity and use that structure within an STGCN.

### Q11. What are your specific objectives?
**Short answer:**  
My specific objectives are to build the correlation-adaptive graph, prepare the forecasting data, implement the STGCN, compare it with a distance-based graph, and position the forecasts for future crop advisory use.

**Expanded answer:**  
More specifically, I will construct a graph for 320 stations using absolute Pearson correlation, create 43-lag inputs with a 7-step horizon, predict `T2M_MIN`, `RH2M`, and `PRECTOTCORR`, compare correlation and distance graph settings, and discuss how the outputs can support downstream agricultural systems.

### Q12. What is your main hypothesis?
**Short answer:**  
My hypothesis is that a correlation-based station graph better represents complex terrain relationships than a purely distance-based graph.

**Expanded answer:**  
If the graph structure is closer to real meteorological affinity, then the model should learn more meaningful spatial patterns. That should improve the quality and interpretability of forecasting in mountainous terrain.

---

## 4. Dataset and Inputs

### Q13. What data are you using?
**Short answer:**  
I am using a 320-station weather dataset with multivariate daily weather records and supporting processed graph artifacts.

**Expanded answer:**  
The proposal is centered on a 320-station workflow with historical weather variables, graph tensors, normalization statistics, and inference/test artifacts. The broader repository also includes the long historical archive used to motivate graph construction and forecasting.

### Q14. Why did you choose 320 stations?
**Short answer:**  
Because 320 stations provide wide spatial coverage while still being computationally manageable for graph-based modeling.

**Expanded answer:**  
This station count gives a substantial node network for learning inter-station relationships. At the same time, it remains practical for preprocessing, graph construction, training, and evaluation within the current project scope.

### Q15. What are the input features to the model?
**Short answer:**  
The active pipeline uses 14 weather features including temperature, humidity, precipitation, pressure, and wind variables.

**Expanded answer:**  
These include variables such as `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`, `PS`, and wind measurements at 10 m and 50 m. This gives the model broader atmospheric context than relying on a single feature.

### Q16. What are your forecast target variables?
**Short answer:**  
The target variables are `T2M_MIN`, `RH2M`, and `PRECTOTCORR`.

**Expanded answer:**  
These correspond to minimum temperature, relative humidity, and corrected precipitation. They were selected because they are meteorologically important and also highly relevant for future agricultural interpretation.

### Q17. Why are these three variables important?
**Short answer:**  
Because they are directly relevant to moisture conditions, and rainfall availability.

**Expanded answer:**  
Minimum temperature relates to cold stress and frost risk, humidity relates to moisture conditions and disease environment, and precipitation affects irrigation and water availability. Together they provide a practical weather profile for downstream decision support.

### Q18. What is the temporal forecasting setup?
**Short answer:**  
The model uses 43 past time steps as input and predicts 7 future steps.

**Expanded answer:**  
This means the forecasting problem is multi-step rather than single-step. The 43-lag input provides enough temporal context, while the 7-step horizon gives a short-term outlook that can be operationally useful.

---

## 5. Graph Construction

### Q19. What is a graph in your study?
**Short answer:**  
A graph is a network where weather stations are nodes and the relationships between stations are edges.

**Expanded answer:**  
The graph defines how information flows between stations during the spatial part of the model. Instead of hard-coding those relationships by distance alone, I estimate them from historical station similarity.

### Q20. How do you construct the correlation-adaptive graph?
**Short answer:**  
I compute pairwise absolute Pearson correlations between station `T2M` time series and use those values as edge weights.

**Expanded answer:**  
First, the station histories are arranged into a time-by-station matrix. Then I compute the Pearson correlation for each station pair, take the absolute value, and keep the strongest connections after sparsification.

### Q21. Why did you use Pearson correlation?
**Short answer:**  
Because it is a simple and interpretable way to measure how strongly two station signals co-vary over time.

**Expanded answer:**  
Pearson correlation is a practical first measure of functional similarity. It gives a transparent statistical basis for graph construction and is easy to explain, reproduce, and analyze.

### Q22. Why did you use the absolute value of correlation?
**Short answer:**  
Because strong negative correlation can still indicate a meaningful meteorological relationship.

**Expanded answer:**  
In mountains, some stations may be strongly anti-correlated because of inversions, slope differences, or local circulation effects. Taking the absolute value preserves the strength of relationship rather than treating anti-correlation as no connection.

### Q23. Why did you use `T2M` for graph construction?
**Short answer:**  
Because `T2M` is a strong, widely available signal and serves as a practical proxy for station similarity.

**Expanded answer:**  
Temperature at 2 meters is informative, stable, and directly used in the implemented graph-construction logic. It provides a simple and evidence-based basis for defining connectivity before expanding to more complex multi-variable graph definitions in future work.

### Q24. What does top-33% sparsification mean?
**Short answer:**  
It means I keep only the strongest one-third of pairwise correlation values as graph edges.

**Expanded answer:**  
Without sparsification, the graph becomes too dense and may contain many weak or noisy connections. Keeping the top 33% helps preserve the strongest functional relationships while reducing noise and computational burden.

### Q25. Why is sparsification necessary?
**Short answer:**  
Because a dense graph can introduce noise and make message passing less meaningful.

**Expanded answer:**  
If every station is connected to too many weak neighbors, the graph loses selectivity. Sparsification makes the graph more informative by emphasizing the strongest and most relevant relationships.

---

## 6. Model and Methodology

### Q26. Why did you choose STGCN?
**Short answer:**  
Because STGCN can model both temporal dependence and spatial dependence in graph-structured time series.

**Expanded answer:**  
Weather forecasting across stations has both time dynamics and inter-station relationships. STGCN is appropriate because it combines temporal convolution with graph convolution in a unified framework.

### Q27. What is the architecture of your proposed model?
**Short answer:**  
It is a three-block STGCN followed by a linear output head.

**Expanded answer:**  
The proposal uses three ST-Conv blocks with temporal kernel sizes 9, 7, and 5, along with graph convolution orders 4, 4, and 3. The final linear layer maps the learned representation to the three forecast targets.

### Q28. Why did you choose three ST-Conv blocks?
**Short answer:**  
Because they allow the model to progressively learn spatio-temporal patterns at multiple levels.

**Expanded answer:**  
Using multiple blocks helps the model capture both short and somewhat broader temporal dependencies while repeatedly passing information through the graph. It creates a deeper representation than a single graph-temporal block.

### Q29. Why do the temporal kernels decrease from 9 to 7 to 5?
**Short answer:**  
Because the model first captures broader temporal structure and then refines it at smaller scales.

**Expanded answer:**  
The larger initial kernel helps the network detect broader weather evolution patterns, while smaller later kernels operate on increasingly compressed representations. This staged design is physically and computationally reasonable for the forecasting task.

### Q30. What does Chebyshev order 4, 4, and 3 mean?
**Short answer:**  
It determines how many graph hops the convolution can aggregate over in each ST-Conv block.

**Expanded answer:**  
Higher Chebyshev order allows each graph convolution to capture information from a wider neighborhood. In my model, the first two blocks use order 4 and the third uses order 3, giving strong spatial aggregation while keeping the model tractable.

### Q31. How does the graph help the model?
**Short answer:**  
The graph tells the model which stations should exchange information during spatial learning.

**Expanded answer:**  
Instead of treating all stations as independent or using only nearest geography, the graph guides message passing according to learned similarity structure. This helps the model exploit meaningful cross-station dependencies.

---

## 7. Splitting, Training, and Evaluation

### Q32. Why did you use chronological splitting?
**Short answer:**  
Because forecasting is inherently time-ordered, so training on the past and testing on the future is the most realistic setup.

**Expanded answer:**  
Chronological splitting avoids temporal leakage and better reflects operational forecasting conditions. It is therefore the primary evaluation protocol in the proposal.

### Q33. Why did you also mention random split?
**Short answer:**  
As a secondary reference to show how performance can look artificially better when temporal leakage is not controlled.

**Expanded answer:**  
Random split is common in deep learning experiments, but in time-series forecasting it can mix similar overlapping windows between train and validation data. Comparing it with chronological split helps demonstrate why the stricter protocol is more trustworthy.

### Q34. What optimizer and main training settings are proposed?
**Short answer:**  
The proposal uses Adam, batch size 32, learning rate 0.05, maximum 300 epochs, and early stopping.

**Expanded answer:**  
These settings are consistent with the active implementation path. Early stopping and learning rate reduction are included to improve convergence stability and reduce overfitting risk.

### Q35. What evaluation metrics will you use?
**Short answer:**  
I will use MSE, MAE, and RMSE, reported per target variable and per forecast horizon step.

**Expanded answer:**  
These are standard regression metrics for multi-step forecasting. Reporting them separately by variable and horizon gives a clearer picture of where the model performs well and where performance degrades with lead time.

### Q36. Why report metrics per horizon step?
**Short answer:**  
Because forecast quality usually changes as lead time increases.

**Expanded answer:**  
A 1-step forecast is usually easier than a 7-step forecast, so averaging everything into one score can hide important behavior. Per-horizon reporting makes the evaluation more informative and transparent.

### Q37. Why include physical plausibility checks?
**Short answer:**  
Because a forecast can have acceptable numerical error while still being meteorologically unrealistic.

**Expanded answer:**  
In practical weather forecasting, plausibility matters in addition to error values. For example, a forecast should remain consistent with seasonal and regional expectations, not just minimize loss mathematically.

---

## 8. Baseline and Comparison

### Q38. What is your baseline comparison?
**Short answer:**  
The main baseline is a distance-based STGCN using the same overall forecasting setup.

**Expanded answer:**  
The purpose of the baseline is to isolate the effect of graph construction. If the architecture, data, and targets remain matched, then differences in performance can be more reasonably attributed to the graph definition itself.

### Q38a. Is the Bhandari et al. (2024) paper your baseline model?
**Short answer:**  
It is my literature benchmark and problem-reference paper, not the direct experimental baseline.

**Expanded answer:**  
Bhandari et al. (2024) is very useful because it is recent, Nepal-specific, and graph-weather focused. However, their setup uses a different station count and focuses on graph-based imputation/classification with GraphSAGE/GNN variants. For a fair experiment in my project, the direct baseline should be a distance-based STGCN using the same 320-station data, targets, split, and metrics as my proposed correlation-based STGCN.

### Q39. Why is this comparison important?
**Short answer:**  
Because it tests whether the proposed correlation graph adds value beyond a standard distance-based graph.

**Expanded answer:**  
Without a matched baseline, it is difficult to argue that the graph design itself matters. This comparison directly addresses the central research question of the proposal.

### Q40. Can you already claim that your method is definitively better?
**Short answer:**  
Not yet as a final claim in the proposal stage.

**Expanded answer:**  
At the proposal stage, I present it as a motivated and testable hypothesis rather than a completed proof. The aim of the research is to implement, evaluate, and verify whether the correlation graph performs better under matched conditions.

---

## 9. Application Relevance

### Q41. How does this relate to crop recommendation?
**Short answer:**  
It provides the weather forecasting backbone that a crop recommendation system can use later.

**Expanded answer:**  
The proposal does not claim a fully implemented end-to-end crop recommendation engine at this stage. Instead, it focuses on improving the upstream weather inputs that such a system would depend on.

### Q42. Why did you not position crop recommendation as the completed core contribution?
**Short answer:**  
Because the strongest current contribution is the forecasting framework, and that is the part directly supported by the implementation path.

**Expanded answer:**  
It is more defensible to present crop recommendation as future integration rather than overstate it as a validated finished module. This keeps the proposal scientifically honest and focused.

### Q43. What kinds of agricultural decisions could eventually use these forecasts?
**Short answer:**  
They could support irrigation planning, crop suitability screening, disease-risk awareness, and short-term advisory services.

**Expanded answer:**  
Forecasted minimum temperature, humidity, and precipitation can be mapped into practical categories such as frost risk, humidity regime, and rainfall condition. Those outputs can later feed crop rules, agronomic thresholds, or advisory models.

---

## 10. Limitations and Risks

### Q44. What are the main limitations of your proposed work?
**Short answer:**  
The main limitations are reproducibility challenges, sensitivity to graph design choices, and the difficulty of forecasting precipitation in complex terrain.

**Expanded answer:**  
Graph construction depends on the chosen signal and sparsification threshold, some data paths may require cleanup for reproducibility, and precipitation is inherently harder to predict than smoother variables. These are all important to acknowledge during defense.

### Q45. Is correlation enough to represent all station relationships?
**Short answer:**  
No, it is a practical and transparent starting point, but not the only possible relationship measure.

**Expanded answer:**  
Correlation captures co-variation well, but it does not capture every nonlinear or regime-dependent interaction. Future work could explore learned adjacency, multi-variable graphs, or attention-based extensions.

### Q46. What if the panel asks why not use a learned adaptive adjacency matrix directly?
**Short answer:**  
A learned adjacency is powerful, but my proposal prioritizes interpretability and a physically grounded prior.

**Expanded answer:**  
Fully learned graphs can be flexible, but they are also harder to interpret and may require more data to train reliably. My approach sits in the middle: it is data-driven, but still transparent and easy to explain scientifically.

### Q47. What if they ask whether anti-correlation should really be treated as similarity?
**Short answer:**  
I would say it is treated as relationship strength rather than same-direction similarity.

**Expanded answer:**  
Using the absolute value means the graph captures stations that are strongly coupled, whether positively or negatively. I would also acknowledge that this is a modeling choice and that preserving sign information could be explored in future work.

---

## 11. Expected Outcomes and Contribution

### Q48. What are your expected methodological outcomes?
**Short answer:**  
I expect to produce a clear and reproducible correlation-based graph construction strategy for complex terrain forecasting.

**Expanded answer:**  
The proposal aims to show that graph connectivity can be defined transparently from historical station behavior, not only from physical distance. That is the main methodological contribution.

### Q49. What are your expected technical outcomes?
**Short answer:**  
I expect to produce a convergent 320-station STGCN forecasting pipeline for the selected target variables.

**Expanded answer:**  
This includes data preparation, graph construction, training, evaluation, and inference using the correlation-adaptive graph. The technical outcome is the full forecasting backbone rather than only a conceptual proposal.

### Q50. What is the practical contribution of your work?
**Short answer:**  
The practical contribution is a stronger weather forecasting foundation for future local advisory systems in mountainous terrain.

**Expanded answer:**  
Even before full crop recommendation is added, a better local forecast is already valuable. It can support future agricultural, planning, and risk-related applications that need more realistic station-level weather estimates.

### Q51. What is your most defensible contribution statement?
**Short answer:**  
My most defensible contribution is a correlation-adaptive graph-based forecasting framework for complex terrain, with crop recommendation framed as future integration.

**Expanded answer:**  
This statement is strong because it matches the technical focus, the revised proposal, and the evidence path of the project. It avoids overclaiming while still clearly stating the novelty and importance of the work.

---

## 12. Timeline and Closing

### Q52. What is your proposed timeline?
**Short answer:**  
The timeline covers data preprocessing and graph construction, baseline implementation, model training and tuning, comparative evaluation, and final thesis writing.

**Expanded answer:**  
In sequence, the work begins with preprocessing and graph creation, then baseline setup, then model training and analysis, followed by evaluation and thesis writing. This structure supports steady progress from data preparation to final defense.

### Q53. Why is your work feasible within the proposed timeline?
**Short answer:**  
Because the methodology is clearly staged and the core modeling path is already well defined.

**Expanded answer:**  
The project is not trying to solve every downstream application at once. By focusing first on the forecasting backbone, the work remains technically ambitious but still manageable within a thesis schedule.

### Q54. If the panel asks for your final takeaway, what will you say?
**Short answer:**  
My final takeaway is that complex terrain needs functionally defined station relationships, and correlation-adaptive graph modeling is a promising way to build them.

**Expanded answer:**  
The proposal argues that distance alone is too weak a basis for weather graphs in mountains. By building connectivity from historical meteorological similarity and using that inside an STGCN, the research aims to produce a more meaningful forecasting backbone for complex terrain and future agricultural decision support.

---

## Very Short Viva Lines

Use these when you need a fast answer in one breath:

- `Why this topic?` Because mountain weather changes sharply over short distances, so distance-based forecasting is often misleading.
- `Main novelty?` I replace geographic proximity with correlation-based station connectivity.
- `Main model?` A three-block STGCN over a correlation-adaptive weather graph.
- `Targets?` `T2M_MIN`, `RH2M`, and `PRECTOTCORR`.
- `Why chronological split?` Because forecasting must train on the past and test on the future.
- `Why random split too?` As a comparison to show possible optimistic bias from temporal leakage.
- `Main contribution?` A forecasting backbone for complex terrain, with crop recommendation as future integration.
