# Mid-Term Progress Report
**Topic**: Precision Crop Recommendation in Complex Terrain via Correlation-Adaptive Weather Graphs

**Date**: [Current Date]  
**Student Name**: [Your Name]

---

## **1. Executive Summary**

This report outlines progress toward the thesis titled **Precision Crop Recommendation in Complex Terrain via Correlation-Adaptive Weather Graphs**. Following recent Nepal-focused work by Bhandari et al. (2024), which demonstrates the suitability of graph neural networks for weather prediction in data-scarce environments, this project investigates whether station connectivity based on historical weather similarity can improve a spatio-temporal weather forecasting model.

The core methodological work completed so far is the conversion of the weather station network from a **geodesic distance-based** topology to a **functional correlation-based** topology. The Spatio-Temporal Graph Convolutional Network (STGCN) model has been adapted to ingest this graph structure. The main remaining experimental task is to train and evaluate the proposed model against a directly comparable distance-based STGCN baseline.

---

## **2. Work Completed**

### **2.1 Literature Foundation and Benchmark Selection**

The thesis problem statement has been aligned with the recent paper:

Bhandari, H. C., Pandeya, Y. R., Jha, K., & Jha, S. (2024). *Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments.* Environmental Research Communications, 6, 105010. https://doi.org/10.1088/2515-7620/ad8108

This paper is used as the recent Nepal-specific reference and literature benchmark. Because its task setup differs from this project, it is not treated as the direct experimental baseline unless reimplemented under the same 320-station protocol.

### **2.2 Data Preprocessing and Graph Construction**

The most significant technical milestone is the re-engineering of the graph adjacency matrix.

*   **Baseline to be compared**: Graph edges defined by physical distance.
*   **Completed proposed graph**:
    *   Implemented a Pearson correlation kernel in `Final_Preprocess_Global.py`.
    *   Processed historical temperature (`T2M`) signals for the 320-station setup.
    *   Generated a global correlation matrix and retained the strongest 33% of pairwise correlation weights.
    *   Produced graph artifacts such as `edge_index.pt` and `edge_weights.pt` for the functional weather network.

### **2.3 Model Implementation**

The deep learning pipeline has been prepared around the STGCN model.

*   **Codebase**: The `Weather 320 Stations` repository is configured to use graph artifacts in the 320-station workflow.
*   **Model Type**: Spatio-Temporal Graph Convolutional Network.
*   **Forecast Targets**: The active path forecasts weather variables such as minimum temperature, relative humidity, and corrected precipitation.
*   **Application Link**: These weather outputs are relevant to crop recommendation, but crop recommendation remains a downstream application layer rather than the primary completed experiment.

---

## **3. Preliminary Findings**

Initial inspection of the generated correlation graph supports the research motivation:

1.  **Non-local connectivity**: Some physically distant stations can be strongly connected when their historical weather behavior is similar.
2.  **Topographical separation**: Nearby stations separated by altitude or terrain barriers can receive lower graph weights when their weather histories differ.

These observations suggest that the functional graph is capturing information that a distance-only graph may miss. Final conclusions require the pending baseline experiment.

---

## **4. Pending Tasks & Timeline**

| Task | Description | Deadline |
| :--- | :--- | :--- |
| **Distance Baseline** | Train the distance-based STGCN using the same data and evaluation protocol. | [Date + 1 Week] |
| **Proposed Model Training** | Train the correlation-based STGCN to convergence. | [Date + 2 Weeks] |
| **Evaluation** | Compute MSE, MAE, and/or RMSE for both models on the same test set. | [Date + 3 Weeks] |
| **Benchmark Discussion** | Compare findings with the Bhandari et al. (2024) literature benchmark. | [Date + 4 Weeks] |
| **Application Framing** | Explain how the forecasts can support future crop recommendation. | [Date + 5 Weeks] |
| **Final Report** | Write the final thesis document. | [Date + 6 Weeks] |

---

## **5. Conclusion**

The project is aligned with recent Nepal-specific graph-weather research and has completed the core graph-construction step for the proposed method. The next critical milestone is the fair baseline comparison: distance-based STGCN versus correlation-based STGCN on the same 320-station forecasting task. This comparison will determine how strongly the thesis can claim that functional connectivity improves weather forecasting in complex terrain.
