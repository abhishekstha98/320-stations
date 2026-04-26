# Final Thesis Summary Report
**Title**: Correlation-Adaptive Weather Graph Modeling for Crop Recommendation in Complex Mountain Terrain

**Candidate**: [Your Name]  
**Date**: [Date]

---

## **1. Introduction and Objectives**

This thesis addresses localized weather forecasting in complex topographical regions, where standard spatial interpolation and nearest-station assumptions may fail to capture microclimates. The statement of problem is grounded in the recent Nepal-focused study by Bhandari, Pandeya, Jha, and Jha (2024), which demonstrates the relevance of graph neural networks for weather prediction in data-scarce environments.

Building on that foundation, this project investigates a **Correlation-Adaptive STGCN** that defines station connectivity using historical weather-signal similarity rather than physical proximity alone. The broader application motivation is precision crop recommendation, because crop advisory systems depend heavily on reliable weather inputs.

## **2. Methodology**

The project is framed around a fair baseline comparison:

1.  **Literature Benchmark**: Bhandari et al. (2024), a recent Nepal-specific GNN weather study, is used as the foundation for the problem statement and as a literature benchmark.
2.  **Direct Baseline Model**: A distance-based STGCN using the same 320-station data, target variables, and evaluation protocol.
3.  **Proposed Model**: A correlation-based STGCN using Pearson correlation over historical weather signals to define graph edge weights.
4.  **Data**: 320 station-aligned weather records with crop-relevant variables such as temperature, humidity, and precipitation.
5.  **Environment**: PyTorch, PyTorch Geometric, and torch_geometric_temporal for graph and spatio-temporal model operations.

## **3. Evaluation Plan**

The direct comparison should report forecasting metrics for both the baseline and proposed models, such as:

*   Mean Squared Error (MSE)
*   Mean Absolute Error (MAE)
*   Root Mean Squared Error (RMSE), if required

The key research question is whether a correlation-based graph improves forecasts compared with a distance-based graph under the same STGCN architecture and dataset.

## **4. Expected Interpretation**

If the proposed model improves on the direct baseline, the result can support the argument that functional connectivity is more suitable than physical proximity for weather modeling in Nepal's complex terrain. If the improvement is mixed, the thesis can still contribute a documented comparison of graph-construction strategies for a 320-station Nepal weather forecasting task.

## **5. Agricultural Application**

The improved weather forecasts are positioned as inputs for downstream crop recommendation systems. In the current evidence framing, crop recommendation should be described as an application pathway rather than as a fully validated end-to-end system unless additional crop-suitability logic and field/yield validation are added.

## **6. Conclusion**

The thesis is now aligned with the supervisor feedback: it starts from a recent published paper, uses that paper as a reference and literature benchmark, and defines a clear direct baseline for comparison. The defensible baseline is a distance-based STGCN evaluated under the same 320-station setup, while Bhandari et al. (2024) provides the recent Nepal-specific foundation for the research problem.

## **7. Reference**

Bhandari, H. C., Pandeya, Y. R., Jha, K., & Jha, S. (2024). *Recent advances in electrical engineering: exploring graph neural networks for weather prediction in data-scarce environments.* Environmental Research Communications, 6, 105010. https://doi.org/10.1088/2515-7620/ad8108
