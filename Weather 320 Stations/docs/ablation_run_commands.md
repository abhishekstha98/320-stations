# Ablation Run Commands

This repository now includes a reproducible experiment runner:

- script: `GNN TRAIN TEST/ablation_runner.py`
- default historical dataset: `Weather_320_Stations_DB.csv`
- default station order: `GNN TRAIN TEST/TEST/stations.txt`
- default output root: `GNN TRAIN TEST/ablation_runs`

If the bundled `stations.txt` has naming mismatches against the historical archive, the runner automatically falls back to the archive's native station order and records that order in each experiment folder.

## Recommended Working Directory

Change into:

```powershell
Set-Location "c:\Users\lula\source\Weather 320 Stations\Weather 320 Stations\GNN TRAIN TEST"
```

If `python` is not on your PATH, use:

```powershell
& "C:\Users\lula\miniconda3\python.exe" .\ablation_runner.py --help
```

Otherwise you can use:

```powershell
python .\ablation_runner.py --help
```

## 1. Run One Experiment

Correlation graph, 33% sparsity, full 14-feature input:

```powershell
python .\ablation_runner.py run --name correlation_r33_full14 --graph-type correlation --edge-ratio 0.33 --feature-set full14
```

Distance graph baseline, matched setup:

```powershell
python .\ablation_runner.py run --name distance_r33_full14 --graph-type distance --edge-ratio 0.33 --feature-set full14
```

Hybrid graph, matched setup:

```powershell
python .\ablation_runner.py run --name hybrid_r33_full14 --graph-type hybrid --edge-ratio 0.33 --feature-set full14
```

## 2. Run the Full Graph Construction Ablation

This runs:

- distance, 33%, full14
- correlation, 33%, full14
- hybrid, 33%, full14

```powershell
python .\ablation_runner.py study --study graph
```

## 3. Run the Full Edge Sparsity Ablation

This runs:

- correlation, 20%, full14
- correlation, 33%, full14
- correlation, 50%, full14
- correlation, 100%, full14

```powershell
python .\ablation_runner.py study --study sparsity
```

## 4. Run the Full Feature Ablation

This runs:

- `full14`
- `thermal5`
- `thermo_hydro7`
- `no_wind8`

```powershell
python .\ablation_runner.py study --study features
```

## 5. Prepare Artifacts Without Training

Useful when you want to verify graph generation and split setup first:

```powershell
python .\ablation_runner.py run --name correlation_r33_full14_prep --graph-type correlation --edge-ratio 0.33 --feature-set full14 --prepare-only
```

## 6. Common Training Overrides

Example with custom epochs, batch size, and seed:

```powershell
python .\ablation_runner.py run --name correlation_r33_full14_e150 --graph-type correlation --edge-ratio 0.33 --feature-set full14 --epochs 150 --batch-size 8 --seed 42
```

Example forcing CPU:

```powershell
python .\ablation_runner.py run --name correlation_cpu --graph-type correlation --edge-ratio 0.33 --feature-set full14 --device cpu
```

## 7. Summarize Finished Results

Graph study:

```powershell
python .\ablation_runner.py summarize --study-dir .\ablation_runs\graph
```

Sparsity study:

```powershell
python .\ablation_runner.py summarize --study-dir .\ablation_runs\sparsity
```

Feature study:

```powershell
python .\ablation_runner.py summarize --study-dir .\ablation_runs\features
```

## 8. Output Files

Each experiment directory contains:

- `config.json`
- `graph_summary.json`
- `edge_index.pt`
- `edge_weight.pt`
- `stations.txt`
- `station_metadata.csv`
- `target_stats.csv`
- `history.csv`
- `best_model.pt`
- `last_model.pt`
- `metrics.json`

Each study directory also gets:

- `summary.csv`

## 9. Feature-Set Meanings

- `full14`: all 14 active input variables used by the current 320-station pipeline
- `thermal5`: `T2M`, `T2MWET`, `TS`, `T2M_MAX`, `T2M_MIN`
- `thermo_hydro7`: thermal features plus `RH2M` and `PRECTOTCORR`
- `no_wind8`: all active non-wind variables in the current code path

## 10. Important Note

Training still requires these Python packages in your environment:

- `torch`
- `torch_geometric`
- `torch_geometric_temporal`

The new runner removes the hard-coded `D:/...` path dependency and rebuilds splits and graphs directly from the historical CSV in this repository.
