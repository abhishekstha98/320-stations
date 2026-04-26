# Thesis Proposal: Correlation-Adaptive Weather Graph Modeling for Crop Recommendation in Complex Mountain Terrain

**Author**: [Your Name]  
**Supervisor**: [Supervisor Name]  
**Department**: [Department Name]  
**Date**: [Date]

---

## **Abstract**

Agriculture in Nepal, and similar mountainous regions, faces a critical challenge: the lack of reliable, hyper-local weather data necessary for precise crop recommendation. This problem is grounded in recent Nepal-focused graph-weather research by Bhandari, Pandeya, Jha, and Jha (2024), who show that data-scarce and geographically complex regions require models that can represent non-Euclidean relationships among weather stations. Traditional systems rely on the "nearest neighbor" assumption, using data from the geographically closest weather station to inform agricultural decisions. In complex topography, however, physical proximity often fails to represent climatological similarity due to drastic variations in altitude and microclimates.

This proposal builds on that recent graph-based direction and introduces a **Correlation-Adaptive Spatio-Temporal Graph Neural Network**. By replacing standard distance-based graphs with functional correlation graphs derived from historical time-series data, we aim to uncover hidden "teleconnections" - distant stations that share weather patterns. This approach is expected to improve the accuracy of weather inputs for crop recommendation systems, thereby supporting agricultural resilience and productivity in data-scarce, rugged terrain.

---

## **1. Introduction**

### **1.1 Background**

Agriculture is the backbone of Nepal's economy, employing a large share of the population. The success of this sector is heavily dependent on the accurate timing of planting, irrigation, and harvesting - decisions that are fundamentally driven by weather conditions. Modern **Crop Recommendation Systems** (CRS) utilize machine learning to suggest optimal crops based on soil and weather parameters. However, the efficacy of these systems is bottlenecked by the quality of the meteorological input.

Recent work by Bhandari et al. (2024), published in *Environmental Research Communications*, provides the immediate research foundation for this proposal. Their study, "Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments," uses Nepal weather-station data and demonstrates the relevance of graph neural networks for missing atmospheric variable prediction in complex terrain. This proposal extends that foundation from graph-based imputation/classification toward a larger 320-station spatio-temporal forecasting setup.

### **1.2 The Problem: The Distance Trap**

In flat terrain, the weather at a farm is often similar to the weather at a nearby station, so nearest-neighbor assignment and simple spatial interpolation can be reasonable first approximations. In the Himalayas, this assumption can break down. A warm river-valley farm may be only a few kilometers from a high-altitude ridge station, but the two locations can have very different weather behavior because of altitude, slope, valley circulation, and rain-shadow effects.

This creates the "Distance Trap": geographic closeness does not always imply meteorological similarity. If crop recommendation systems consume weather values from the geographically nearest station, they may produce unsuitable agricultural advice when the nearest station belongs to a different microclimate.

### **1.3 Research Gap**

Existing graph-based weather studies, including Bhandari et al. (2024), confirm that Nepal's terrain and data scarcity make graph neural networks a relevant modeling direction. However, there remains a specific gap for this thesis: many spatio-temporal weather models still define station neighborhoods using **Geodesic Distance**, **Inverse Distance Weighting (IDW)**, or other location-centered assumptions. These assumptions can fail to capture complex atmospheric dynamics that connect distant but behaviorally similar locations while separating nearby but climatically different ones.

---

## **2. Proposed Solution**

We propose a shift from **Physical Proximity** to **Functional Connectivity**.

Instead of asking "Which station is closest?", we ask "Which station behaves most similarly?"

### **2.1 Core Hypothesis**

We hypothesize that a graph neural network constructed using **Pearson correlation coefficients** of historical weather signals will outperform a comparable network based on geodesic distance in complex terrain. This functional graph will identify statistically meaningful links between distant stations, effectively creating a virtual sensor network for locations without reliable local data.

### **2.2 Application to Agriculture**

By improving the accuracy of the underlying weather forecast, especially temperature, humidity, and precipitation, the proposed model can improve the quality of weather inputs used by downstream crop recommendation systems. In this proposal, crop recommendation is treated as the application motivation, while the primary research contribution is the weather-forecasting backbone.

---

## **3. Methodology**

### **3.1 Dataset**

*   **Source**: Department of Hydrology and Meteorology (DHM), ICIMOD, and/or station-aligned meteorological records available in the project dataset.
*   **Coverage**: 320 weather stations across Nepal.
*   **Temporal Resolution**: Historical daily station-level records.
*   **Features**:
    *   Target variables: temperature/minimum temperature, relative humidity, and precipitation.
    *   Auxiliary variables: surface pressure, wind variables, humidity variables, and station metadata such as latitude, longitude, and altitude where available.

### **3.2 Graph Construction**

We define the graph \(G = (V, E, W)\), where \(V\) is the set of 320 weather stations.

*   **Direct Baseline: Distance-Based Graph**

    \[
    W_{ij} = \exp\left(-\frac{dist(i,j)^2}{\sigma^2}\right)
    \]

    This baseline represents the conventional assumption that geographically closer stations should have stronger influence.

*   **Proposed Graph: Correlation-Based Graph**

    \[
    W_{ij} = |\rho(X_i, X_j)|
    \]

    Here, \(\rho\) is the Pearson correlation coefficient between the historical time series \(X_i\) and \(X_j\) of stations \(i\) and \(j\). In the current implementation, the graph retains the top 33% of pairwise correlation weights to remove weak and noisy edges.

### **3.3 Baseline and Benchmark Strategy**

The project uses two comparison levels:

1.  **Direct Experimental Baseline**: A distance-based STGCN using the same 320-station data, target variables, train/test split, and evaluation metrics as the proposed correlation-based STGCN. This is the primary baseline because it allows a fair model-to-model comparison.
2.  **Literature Benchmark**: The 2024 Bhandari et al. GraphSAGE/GNN-based weather imputation study is used as a recent Nepal-specific benchmark and research foundation. Because their study uses a different station count, task formulation, and classification/imputation setup, it should be cited as a benchmark reference rather than treated as a one-to-one experimental baseline unless reimplemented on the same 320-station protocol.

### **3.4 Model Architecture: Spatio-Temporal Graph Convolutional Network**

The model processes graph-structured time-series data using STGCN blocks:

1.  **Spatial Block**: Aggregates information from graph neighbors using graph convolution. In the proposed model, neighbors are selected by historical signal similarity rather than only by physical proximity.
2.  **Temporal Block**: Captures short-term weather patterns and multi-day dependencies from station time-series windows.
3.  **Output Layer**: Projects the learned representation into a seven-day forecast horizon for the key agricultural weather variables.

### **3.5 Environment and Tools**

*   **Programming Language**: Python
*   **Deep Learning Framework**: PyTorch
*   **Graph Operations**: PyTorch Geometric and torch_geometric_temporal
*   **Data Processing**: Pandas and NumPy

---

## **4. Objectives**

1.  **To Ground the Problem in Recent Literature**: Use Bhandari et al. (2024) as the recent Nepal-specific graph-weather reference for data-scarce and complex-terrain weather prediction.
2.  **To Construct a Functional Weather Graph**: Generate a correlation-based adjacency matrix for 320 stations in Nepal, mapping station relationships based on data similarity rather than geography alone.
3.  **To Develop the STGCN Model**: Implement and train the neural network to forecast crop-relevant weather variables such as temperature, humidity, and precipitation.
4.  **To Validate Against a Baseline**: Compare the proposed correlation-based STGCN against a distance-based STGCN using forecasting metrics such as MSE, MAE, and/or RMSE.

---

## **5. Expected Outcomes**

*   **Scientific Contribution**: Evidence that data-driven topology can be a stronger representation than distance-only topology for weather modeling in mountainous regions.
*   **Technical Output**: A 320-station correlation graph and STGCN forecasting pipeline.
*   **Benchmarking Output**: A clear comparison against a distance-based STGCN baseline, with Bhandari et al. (2024) used as the recent literature benchmark.
*   **Agricultural Relevance**: A forecasting backend that can later support crop recommendation by providing more reliable weather variables for downstream advisory models.

---

## **6. Work Plan & Timeline**

| Phase | Activity | Status |
| :--- | :--- | :--- |
| **Phase 1** | Data cleaning and preprocessing | Completed |
| **Phase 2** | Correlation graph construction | Completed |
| **Phase 3** | STGCN model implementation | In Progress |
| **Phase 4** | Distance-based baseline setup | Pending |
| **Phase 5** | Model training and hyperparameter tuning | Pending |
| **Phase 6** | Comparative analysis: correlation graph vs distance graph | Pending |
| **Phase 7** | Literature benchmarking with Bhandari et al. (2024) | Pending |
| **Phase 8** | Thesis writing and defense preparation | Pending |

---

## **7. References**

1.  Bhandari, H. C., Pandeya, Y. R., Jha, K., & Jha, S. (2024). *Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments.* Environmental Research Communications, 6, 105010. https://doi.org/10.1088/2515-7620/ad8108
2.  Yu, B., Yin, H., & Zhu, Z. (2018). *Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting.* IJCAI.
3.  Wu, Z., et al. (2020). *Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks.* KDD.
4.  Guo, S., et al. (2019). *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.* AAAI.
5.  Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
