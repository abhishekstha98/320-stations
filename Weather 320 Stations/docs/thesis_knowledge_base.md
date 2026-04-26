# Thesis Master Knowledge Base
**Project**: Precision Crop Recommendation in Complex Terrain via Correlation-Adaptive Weather Graphs  
**Role**: Context Document for LLM Generation

---

## **1. Core Concept and Pitch**

*   **Recent Foundation**: The statement of problem should cite Bhandari, Pandeya, Jha, and Jha (2024), a recent Nepal-focused paper on graph neural networks for weather prediction in data-scarce environments.
*   **The Problem**: Weather forecasting in mountainous regions such as Nepal is difficult because spatial closeness does not always imply weather similarity. A valley station and a ridge station may be close in map distance but very different in temperature, humidity, and rainfall behavior.
*   **The Pivot**: Move from **geodesic distance graphs** to **Pearson correlation graphs**.
*   **The Goal**: Improve the weather-forecasting backbone used by downstream crop recommendation systems.

---

## **2. Reference and Benchmark Position**

### **2.1 Paper to Use as Recent Problem Reference**

Bhandari, H. C., Pandeya, Y. R., Jha, K., & Jha, S. (2024). *Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments.* Environmental Research Communications, 6, 105010. https://doi.org/10.1088/2515-7620/ad8108

Use this paper because it is recent, Nepal-specific, weather-focused, and graph-neural-network based.

### **2.2 How to Describe Its Role**

Use Bhandari et al. (2024) as:

*   the recent foundation for the statement of problem;
*   the Nepal-specific literature benchmark;
*   evidence that GNNs are suitable for weather prediction/imputation in data-scarce complex terrain.

Do not present it as the direct experimental baseline unless its model is reimplemented under the same 320-station dataset, target variables, and evaluation protocol.

### **2.3 Direct Baseline for This Project**

The primary direct baseline should be:

*   **Distance-based STGCN**: same architecture, same 320-station data, same targets, same split, same metrics, but with graph edges defined by geodesic distance.

The proposed model should be:

*   **Correlation-based STGCN**: same setup, but graph edges defined by Pearson correlation over historical weather signals.

---

## **3. Technical Specifications**

### **3.1 Dataset**

*   **Station Set**: 320 meteorological stations across Nepal.
*   **Historical Archive**: `Weather_320_Stations_DB.csv` appears to be the multi-date station archive suitable for time-series graph construction.
*   **Variables**:
    *   `T2M`: temperature at 2 meters, used in the active graph-construction logic.
    *   `T2M_MIN`: minimum temperature target in the active inference/training path.
    *   `RH2M`: relative humidity target.
    *   `PRECTOTCORR`: precipitation target.
    *   Other inputs include pressure, humidity, wet-bulb temperature, wind, and related meteorological variables.
*   **Input Shape**: The active 320-station model uses 320 nodes and 14 features per node over temporal windows.

### **3.2 Methodology: Graph Construction**

*   **Baseline Graph**

    \[
    A_{ij} = \exp\left(-\frac{dist(v_i, v_j)^2}{\sigma^2}\right)
    \]

    This tests the conventional physical-proximity assumption.

*   **Proposed Graph**

    \[
    A_{ij} = |\text{Pearson}(T2M_i, T2M_j)|
    \]

    \[
    \text{Pearson}(x, y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}
    \]

    The current implementation retains the top 33% of pairwise correlation weights to remove weak edges.

### **3.3 Model**

*   **Framework**: Spatio-Temporal Graph Convolutional Network.
*   **Library**: `torch_geometric_temporal`.
*   **Operational Model in Code**: `STGCN_Best_BRC` in `GNN TRAIN TEST/Train_Global.py`.
*   **Forecast Targets in Active Path**: `T2M_MIN`, `RH2M`, and `PRECTOTCORR`.
*   **Forecast Horizon**: seven steps.

---

## **4. Evidence Boundaries**

Use careful language:

*   Say the project **implements** a correlation-based weather graph and STGCN forecasting pipeline.
*   Say the project **uses** Bhandari et al. (2024) as the recent literature foundation and benchmark.
*   Say the distance-based STGCN is the **direct baseline to compare against**.
*   Say crop recommendation is the **downstream application motivation** unless a full crop-recommendation engine and validation are added.

Avoid unsupported claims:

*   Do not claim 30-35% improvement unless a completed baseline table exists.
*   Do not claim rice/yield validation unless there is actual implemented evidence and data.
*   Do not claim the Bhandari et al. model is directly comparable unless re-run under this project setup.

---

## **5. Suggested Thesis Flow**

1.  **Introduction**: Start from Bhandari et al. (2024) and Nepal's data-scarce, complex-terrain weather problem.
2.  **Problem Statement**: Explain why nearest-station and distance-only assumptions can fail in mountains.
3.  **Literature Review**: Review GNN weather work, including Bhandari et al. as the recent Nepal-specific benchmark, then review STGCN foundations.
4.  **Methodology**: Define distance-based baseline graph and Pearson-correlation proposed graph.
5.  **Experiments**: Compare correlation-based STGCN against distance-based STGCN using the same 320-station protocol.
6.  **Discussion**: Explain whether functional connectivity better captures microclimate similarity.
7.  **Application**: Present crop recommendation as a downstream use case enabled by better weather forecasts.
