from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric_temporal.nn.attention.stgcn import STConv
except ImportError as exc:  # pragma: no cover - dependency is optional at authoring time
    STConv = None
    STCONV_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only in a configured training environment
    STCONV_IMPORT_ERROR = None


ALL_FEATURES = [
    "T2M",
    "T2MWET",
    "TS",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "PS",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "WS50M",
    "WS50M_MAX",
    "WS50M_MIN",
]

TARGET_FEATURES = ["T2M_MIN", "RH2M", "PRECTOTCORR"]

FEATURE_SET_DEFS = {
    "full14": ALL_FEATURES,
    "thermal5": ["T2M", "T2MWET", "TS", "T2M_MAX", "T2M_MIN"],
    "thermo_hydro7": ["T2M", "T2MWET", "TS", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"],
    "no_wind8": ["T2M", "T2MWET", "TS", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "PS"],
}

GRAPH_TYPES = ("correlation", "distance", "hybrid")
STUDY_TYPES = ("graph", "sparsity", "features")


@dataclass
class LoadedSourceData:
    frame: pd.DataFrame
    timestamps: list[str]
    stations: list[str]
    metadata: pd.DataFrame
    feature_tensor: np.ndarray


@dataclass
class SplitIndices:
    train_starts: np.ndarray
    val_starts: np.ndarray
    test_starts: np.ndarray
    train_time_end: int
    val_time_end: int


@dataclass
class PreparedExperiment:
    source: LoadedSourceData
    split: SplitIndices
    input_features: list[str]
    target_features: list[str]
    normalized_tensor: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    graph_summary: dict[str, Any]


class TemporalWindowDataset(Dataset):
    def __init__(
        self,
        normalized_tensor: np.ndarray,
        sample_starts: np.ndarray,
        input_indices: list[int],
        target_indices: list[int],
        lags: int,
        horizon: int,
    ) -> None:
        self.normalized_tensor = normalized_tensor
        self.sample_starts = sample_starts.astype(np.int64)
        self.input_indices = input_indices
        self.target_indices = target_indices
        self.lags = lags
        self.horizon = horizon

    def __len__(self) -> int:
        return int(self.sample_starts.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.sample_starts[index])
        input_end = start + self.lags
        target_end = input_end + self.horizon

        x = self.normalized_tensor[start:input_end, :, self.input_indices]
        y = self.normalized_tensor[input_end:target_end, :, self.target_indices]

        return (
            torch.from_numpy(x).unsqueeze(0).to(torch.float32),
            torch.from_numpy(y).unsqueeze(0).to(torch.float32),
        )


class STGCNForecastModel(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int) -> None:
        super().__init__()

        if STConv is None:
            raise ImportError(
                "torch_geometric_temporal is required to train this model. "
                "Install torch_geometric and torch_geometric_temporal first."
            ) from STCONV_IMPORT_ERROR

        self.stconv_block1 = STConv(num_nodes, in_channels, 64, 128, 9, 4)
        self.stconv_block2 = STConv(num_nodes, 128, 256, 64, 7, 4)
        self.stconv_block3 = STConv(num_nodes, 64, 32, 16, 5, 3)
        self.fc = nn.Linear(16, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        temp = self.stconv_block1(x, edge_index, edge_weight)
        temp = self.stconv_block2(temp, edge_index, edge_weight)
        temp = self.stconv_block3(temp, edge_index, edge_weight)
        return self.fc(temp)


def default_paths() -> dict[str, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return {
        "script_dir": script_dir,
        "repo_root": repo_root,
        "data_csv": repo_root / "Weather_320_Stations_DB.csv",
        "stations_file": script_dir / "TEST" / "stations.txt",
        "output_root": script_dir / "ablation_runs",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip()]
    cleaned = "".join(keep).strip("_")
    return cleaned or "experiment"


def load_station_order(stations_file: Path | None, frame: pd.DataFrame) -> list[str]:
    observed = frame["Location"].drop_duplicates().tolist()
    if stations_file is None or not stations_file.exists():
        return observed

    stations = [line.strip() for line in stations_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_set = set(observed)
    missing = [station for station in stations if station not in observed_set]
    extras = [station for station in observed if station not in set(stations)]

    if missing:
        print(
            f"Station order fallback: {stations_file} contains names not found in the historical archive "
            f"(examples: {missing[:5]}). Using the archive order instead."
        )
        return observed
    if extras:
        print(
            f"Station order fallback: the historical archive contains stations not present in {stations_file} "
            f"(examples: {extras[:5]}). Using the archive order instead."
        )
        return observed
    return stations


def load_source_data(data_csv: Path, stations_file: Path | None) -> LoadedSourceData:
    required_columns = ["Date", "Location", "Latitude", "Longitude", "Altitude", *ALL_FEATURES]
    frame = pd.read_csv(data_csv, usecols=required_columns)
    frame["Date"] = pd.to_datetime(frame["Date"])

    stations = load_station_order(stations_file, frame)
    frame["Location"] = pd.Categorical(frame["Location"], categories=stations, ordered=True)
    frame = frame.sort_values(["Date", "Location"], kind="mergesort").reset_index(drop=True)

    counts_per_day = frame.groupby("Date", sort=True)["Location"].count()
    station_count = len(stations)
    if not bool((counts_per_day == station_count).all()):
        raise ValueError("The historical archive is not station-complete for every date.")

    timestamps = frame["Date"].drop_duplicates().dt.strftime("%Y-%m-%d").tolist()
    metadata = (
        frame.drop_duplicates("Location")
        .set_index("Location")
        .loc[stations, ["Latitude", "Longitude", "Altitude"]]
        .copy()
    )

    feature_tensor = frame[ALL_FEATURES].to_numpy(dtype=np.float32).reshape(
        len(timestamps),
        station_count,
        len(ALL_FEATURES),
    )

    return LoadedSourceData(
        frame=frame,
        timestamps=timestamps,
        stations=stations,
        metadata=metadata,
        feature_tensor=feature_tensor,
    )


def build_strict_block_splits(
    num_timesteps: int,
    lags: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> SplitIndices:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    required_span = lags + horizon
    if num_timesteps < required_span * 3:
        raise ValueError(
            "Not enough timesteps for strict train/validation/test blocks with the chosen lags and horizon."
        )

    train_time_end = int(num_timesteps * train_ratio)
    val_time_end = int(num_timesteps * (train_ratio + val_ratio))

    if train_time_end < required_span:
        raise ValueError("Training block is too small for the chosen lags and horizon.")
    if (val_time_end - train_time_end) < required_span:
        raise ValueError("Validation block is too small for the chosen lags and horizon.")
    if (num_timesteps - val_time_end) < required_span:
        raise ValueError("Test block is too small for the chosen lags and horizon.")

    all_sample_starts = np.arange(0, num_timesteps - required_span + 1, dtype=np.int64)

    train_mask = (all_sample_starts + required_span) <= train_time_end
    val_mask = ((all_sample_starts >= train_time_end) & ((all_sample_starts + required_span) <= val_time_end))
    test_mask = all_sample_starts >= val_time_end

    train_starts = all_sample_starts[train_mask]
    val_starts = all_sample_starts[val_mask]
    test_starts = all_sample_starts[test_mask]

    if len(train_starts) == 0 or len(val_starts) == 0 or len(test_starts) == 0:
        raise ValueError("One of the strict temporal splits produced zero samples.")

    return SplitIndices(
        train_starts=train_starts,
        val_starts=val_starts,
        test_starts=test_starts,
        train_time_end=train_time_end,
        val_time_end=val_time_end,
    )


def compute_normalization_stats(feature_tensor: np.ndarray, train_time_end: int) -> tuple[np.ndarray, np.ndarray]:
    train_slice = feature_tensor[:train_time_end]
    flattened = train_slice.reshape(-1, train_slice.shape[-1])
    mean = flattened.mean(axis=0).astype(np.float32)
    std = flattened.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize_feature_tensor(feature_tensor: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((feature_tensor - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def absolute_correlation_matrix(signal_matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(signal_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32)


def haversine_distance_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    lat_rad = np.radians(latitudes.astype(np.float64))
    lon_rad = np.radians(longitudes.astype(np.float64))

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * sin_dlon**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))

    distance = radius_km * c
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32)


def distance_similarity_matrix(metadata: pd.DataFrame) -> np.ndarray:
    distance = haversine_distance_matrix(
        metadata["Latitude"].to_numpy(),
        metadata["Longitude"].to_numpy(),
    )
    upper = distance[np.triu_indices(distance.shape[0], k=1)]
    scale = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    similarity = np.exp(-(distance / max(scale, 1e-6)))
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32)


def build_graph_matrix(
    graph_type: str,
    feature_tensor: np.ndarray,
    metadata: pd.DataFrame,
    graph_signal: str,
) -> np.ndarray:
    if graph_signal not in ALL_FEATURES:
        raise ValueError(f"graph_signal must be one of: {', '.join(ALL_FEATURES)}")

    signal_index = ALL_FEATURES.index(graph_signal)
    train_signal = feature_tensor[:, :, signal_index]
    corr_matrix = absolute_correlation_matrix(train_signal)

    if graph_type == "correlation":
        return corr_matrix

    dist_matrix = distance_similarity_matrix(metadata)
    if graph_type == "distance":
        return dist_matrix
    if graph_type == "hybrid":
        return ((corr_matrix + dist_matrix) / 2.0).astype(np.float32)

    raise ValueError(f"Unsupported graph_type: {graph_type}")


def sparsify_graph(matrix: np.ndarray, edge_ratio: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not 0 < edge_ratio <= 1.0:
        raise ValueError("edge_ratio must be in the range (0, 1].")

    node_count = matrix.shape[0]
    upper_rows, upper_cols = np.triu_indices(node_count, k=1)
    pair_weights = matrix[upper_rows, upper_cols]
    pair_count = pair_weights.shape[0]
    keep_count = pair_count if math.isclose(edge_ratio, 1.0) else max(1, int(round(pair_count * edge_ratio)))

    if keep_count >= pair_count:
        keep_indices = np.arange(pair_count)
    else:
        keep_indices = np.argpartition(pair_weights, -keep_count)[-keep_count:]
        keep_indices = keep_indices[np.argsort(pair_weights[keep_indices])[::-1]]

    kept_rows = upper_rows[keep_indices]
    kept_cols = upper_cols[keep_indices]
    kept_weights = pair_weights[keep_indices].astype(np.float32)

    source_nodes = np.concatenate([kept_rows, kept_cols, np.arange(node_count, dtype=np.int64)])
    target_nodes = np.concatenate([kept_cols, kept_rows, np.arange(node_count, dtype=np.int64)])
    edge_values = np.concatenate([kept_weights, kept_weights, np.ones(node_count, dtype=np.float32)])

    edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)
    edge_weight = torch.tensor(edge_values, dtype=torch.float32)

    summary = {
        "num_nodes": node_count,
        "undirected_candidate_edges": int(pair_count),
        "undirected_edges_kept": int(keep_count),
        "directed_edges_kept": int(keep_count * 2),
        "self_loops_added": int(node_count),
        "edge_ratio_requested": float(edge_ratio),
        "edge_ratio_realized": float(keep_count / pair_count),
        "weight_min": float(kept_weights.min()) if kept_weights.size else 0.0,
        "weight_mean": float(kept_weights.mean()) if kept_weights.size else 0.0,
        "weight_max": float(kept_weights.max()) if kept_weights.size else 0.0,
    }
    return edge_index, edge_weight, summary


def prepare_experiment(args: argparse.Namespace, source: LoadedSourceData) -> PreparedExperiment:
    split = build_strict_block_splits(
        num_timesteps=source.feature_tensor.shape[0],
        lags=args.lags,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    input_features = FEATURE_SET_DEFS[args.feature_set]
    input_indices = [ALL_FEATURES.index(feature) for feature in input_features]
    target_indices = [ALL_FEATURES.index(feature) for feature in TARGET_FEATURES]

    feature_mean, feature_std = compute_normalization_stats(source.feature_tensor, split.train_time_end)
    normalized_tensor = normalize_feature_tensor(source.feature_tensor, feature_mean, feature_std)

    graph_train_tensor = source.feature_tensor[:split.train_time_end]
    graph_matrix = build_graph_matrix(args.graph_type, graph_train_tensor, source.metadata, args.graph_signal)
    edge_index, edge_weight, graph_summary = sparsify_graph(graph_matrix, args.edge_ratio)

    return PreparedExperiment(
        source=source,
        split=split,
        input_features=input_features,
        target_features=TARGET_FEATURES.copy(),
        normalized_tensor=normalized_tensor,
        target_mean=feature_mean[target_indices],
        target_std=feature_std[target_indices],
        edge_index=edge_index,
        edge_weight=edge_weight,
        graph_summary=graph_summary,
    )


def build_dataloaders(
    prepared: PreparedExperiment,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    input_indices = [ALL_FEATURES.index(feature) for feature in prepared.input_features]
    target_indices = [ALL_FEATURES.index(feature) for feature in prepared.target_features]

    datasets = [
        TemporalWindowDataset(
            normalized_tensor=prepared.normalized_tensor,
            sample_starts=prepared.split.train_starts,
            input_indices=input_indices,
            target_indices=target_indices,
            lags=args.lags,
            horizon=args.horizon,
        ),
        TemporalWindowDataset(
            normalized_tensor=prepared.normalized_tensor,
            sample_starts=prepared.split.val_starts,
            input_indices=input_indices,
            target_indices=target_indices,
            lags=args.lags,
            horizon=args.horizon,
        ),
        TemporalWindowDataset(
            normalized_tensor=prepared.normalized_tensor,
            sample_starts=prepared.split.test_starts,
            input_indices=input_indices,
            target_indices=target_indices,
            lags=args.lags,
            horizon=args.horizon,
        ),
    ]

    loaders = []
    for index, dataset in enumerate(datasets):
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=index == 0,
                num_workers=0,
                collate_fn=lambda batch: batch,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def run_loader_loss(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    losses: list[float] = []
    training = optimizer is not None
    criterion = nn.MSELoss()

    if training:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch_losses = []
            for x, y in batch:
                x = x.to(device)
                y = y.to(device)
                pred = model(x, edge_index, edge_weight)
                batch_losses.append(criterion(pred, y))

            batch_loss = torch.stack(batch_losses).mean()
            losses.append(float(batch_loss.detach().cpu()))

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()

    return float(np.mean(losses)) if losses else float("nan")


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            for x, y in batch:
                pred = model(x.to(device), edge_index, edge_weight).detach().cpu().numpy()
                preds.append(pred.squeeze(0))
                targets.append(y.numpy().squeeze(0))

    return np.stack(preds, axis=0), np.stack(targets, axis=0)


def denormalize_targets(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std.reshape(1, 1, 1, -1)) + mean.reshape(1, 1, 1, -1)


def clip_precipitation(values: np.ndarray, target_features: list[str]) -> np.ndarray:
    clipped = values.copy()
    if "PRECTOTCORR" in target_features:
        precip_index = target_features.index("PRECTOTCORR")
        clipped[..., precip_index] = np.maximum(clipped[..., precip_index], 0.0)
    return clipped


def metric_triplet(diff: np.ndarray) -> dict[str, float]:
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def compute_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    prepared: PreparedExperiment,
) -> dict[str, Any]:
    pred = denormalize_targets(pred_norm, prepared.target_mean, prepared.target_std)
    target = denormalize_targets(target_norm, prepared.target_mean, prepared.target_std)
    pred = clip_precipitation(pred, prepared.target_features)

    diff = pred - target
    per_target = {}
    for feature_index, feature_name in enumerate(prepared.target_features):
        per_target[feature_name] = metric_triplet(diff[..., feature_index])

    per_horizon = {}
    for horizon_step in range(diff.shape[1]):
        per_horizon[str(horizon_step + 1)] = metric_triplet(diff[:, horizon_step, :, :])

    return {
        "overall": metric_triplet(diff),
        "per_target": per_target,
        "per_horizon": per_horizon,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def experiment_output_dir(args: argparse.Namespace) -> Path:
    study_root = Path(args.output_root).resolve()
    if getattr(args, "study", None):
        study_root = study_root / args.study
    return study_root / sanitize_name(args.name)


def save_preparation_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    prepared: PreparedExperiment,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prepared.edge_index, output_dir / "edge_index.pt")
    torch.save(prepared.edge_weight, output_dir / "edge_weight.pt")

    config_payload = {
        "name": args.name,
        "graph_type": args.graph_type,
        "edge_ratio": args.edge_ratio,
        "feature_set": args.feature_set,
        "graph_signal": args.graph_signal,
        "lags": args.lags,
        "horizon": args.horizon,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "data_csv": str(Path(args.data_csv).resolve()),
        "stations_file": str(Path(args.stations_file).resolve()) if args.stations_file else None,
        "input_features": prepared.input_features,
        "target_features": prepared.target_features,
        "train_samples": int(len(prepared.split.train_starts)),
        "val_samples": int(len(prepared.split.val_starts)),
        "test_samples": int(len(prepared.split.test_starts)),
        "train_time_end": int(prepared.split.train_time_end),
        "val_time_end": int(prepared.split.val_time_end),
        "station_count": int(len(prepared.source.stations)),
        "time_steps": int(len(prepared.source.timestamps)),
    }
    graph_payload = {
        **prepared.graph_summary,
        "stations_file_used": str(output_dir / "stations.txt"),
    }

    write_json(output_dir / "config.json", config_payload)
    write_json(output_dir / "graph_summary.json", graph_payload)
    pd.DataFrame({"station": prepared.source.stations}).to_csv(output_dir / "stations.txt", index=False, header=False)
    pd.DataFrame(prepared.source.metadata).to_csv(output_dir / "station_metadata.csv")
    pd.DataFrame(
        [
            {
                "feature": feature,
                "mean": float(prepared.target_mean[index]) if feature in TARGET_FEATURES else None,
                "std": float(prepared.target_std[index]) if feature in TARGET_FEATURES else None,
            }
            for index, feature in enumerate(TARGET_FEATURES)
        ]
    ).to_csv(output_dir / "target_stats.csv", index=False)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def train_experiment(
    args: argparse.Namespace,
    prepared: PreparedExperiment,
    output_dir: Path,
) -> dict[str, Any]:
    if STConv is None:
        raise ImportError(
            "Cannot train because torch_geometric_temporal is not installed in this environment."
        ) from STCONV_IMPORT_ERROR

    set_seed(args.seed)
    train_loader, val_loader, test_loader = build_dataloaders(prepared, args)

    device = resolve_device(args.device)
    edge_index = prepared.edge_index.to(device)
    edge_weight = prepared.edge_weight.to(device)

    model = STGCNForecastModel(
        num_nodes=len(prepared.source.stations),
        in_channels=len(prepared.input_features),
        out_channels=len(prepared.target_features),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    lr_reduced = False
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_loader_loss(model, train_loader, edge_index, edge_weight, device, optimizer=optimizer)
        val_loss = run_loader_loss(model, val_loader, edge_index, edge_weight, device, optimizer=None)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if (not lr_reduced) and epochs_without_improvement >= max(1, args.patience // 2):
            optimizer.param_groups[0]["lr"] = current_lr * args.lr_drop_factor
            lr_reduced = True

        if epochs_without_improvement >= args.patience:
            break

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    write_history(output_dir / "history.csv", history_rows)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    val_pred, val_target = collect_predictions(model, val_loader, edge_index, edge_weight, device)
    test_pred, test_target = collect_predictions(model, test_loader, edge_index, edge_weight, device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "completed_epochs": len(history_rows),
        "validation": compute_metrics(val_pred, val_target, prepared),
        "test": compute_metrics(test_pred, test_target, prepared),
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def run_single_experiment(
    args: argparse.Namespace,
    source: LoadedSourceData | None = None,
) -> dict[str, Any]:
    source = source or load_source_data(Path(args.data_csv), Path(args.stations_file) if args.stations_file else None)
    prepared = prepare_experiment(args, source)
    output_dir = experiment_output_dir(args)
    save_preparation_artifacts(output_dir, args, prepared)

    if args.prepare_only:
        return {
            "name": args.name,
            "prepared_only": True,
            "output_dir": str(output_dir),
            "graph_type": args.graph_type,
            "edge_ratio": args.edge_ratio,
            "feature_set": args.feature_set,
        }

    metrics = train_experiment(args, prepared, output_dir)
    metrics["name"] = args.name
    metrics["output_dir"] = str(output_dir)
    return metrics


def make_study_variants(args: argparse.Namespace) -> list[argparse.Namespace]:
    base_payload = vars(args)
    variants: list[dict[str, Any]]

    if args.study == "graph":
        variants = [
            {"name": "distance_r33_full14", "graph_type": "distance", "edge_ratio": 0.33, "feature_set": "full14"},
            {"name": "correlation_r33_full14", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "full14"},
            {"name": "hybrid_r33_full14", "graph_type": "hybrid", "edge_ratio": 0.33, "feature_set": "full14"},
        ]
    elif args.study == "sparsity":
        variants = [
            {"name": "correlation_r20_full14", "graph_type": "correlation", "edge_ratio": 0.20, "feature_set": "full14"},
            {"name": "correlation_r33_full14", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "full14"},
            {"name": "correlation_r50_full14", "graph_type": "correlation", "edge_ratio": 0.50, "feature_set": "full14"},
            {"name": "correlation_r100_full14", "graph_type": "correlation", "edge_ratio": 1.00, "feature_set": "full14"},
        ]
    elif args.study == "features":
        variants = [
            {"name": "correlation_r33_full14", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "full14"},
            {"name": "correlation_r33_thermal5", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "thermal5"},
            {"name": "correlation_r33_thermo_hydro7", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "thermo_hydro7"},
            {"name": "correlation_r33_no_wind8", "graph_type": "correlation", "edge_ratio": 0.33, "feature_set": "no_wind8"},
        ]
    else:
        raise ValueError(f"Unsupported study type: {args.study}")

    namespaced_variants: list[argparse.Namespace] = []
    for variant in variants:
        payload = copy.deepcopy(base_payload)
        payload.update(variant)
        namespaced_variants.append(argparse.Namespace(**payload))
    return namespaced_variants


def summarize_results(study_dir: Path) -> Path:
    metric_files = sorted(study_dir.glob("*/metrics.json"))
    rows: list[dict[str, Any]] = []

    for metric_file in metric_files:
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        config_path = metric_file.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        rows.append(
            {
                "experiment": metric_file.parent.name,
                "graph_type": config.get("graph_type"),
                "edge_ratio": config.get("edge_ratio"),
                "feature_set": config.get("feature_set"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_loss": metrics.get("best_val_loss"),
                "test_mae": metrics.get("test", {}).get("overall", {}).get("mae"),
                "test_rmse": metrics.get("test", {}).get("overall", {}).get("rmse"),
                "test_mse": metrics.get("test", {}).get("overall", {}).get("mse"),
            }
        )

    summary_path = study_dir / "summary.csv"
    if rows:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return summary_path


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    source = load_source_data(Path(args.data_csv), Path(args.stations_file) if args.stations_file else None)
    results = []
    for variant_args in make_study_variants(args):
        results.append(run_single_experiment(variant_args, source=source))

    study_dir = Path(args.output_root).resolve() / args.study
    summary_path = summarize_results(study_dir)
    return {
        "study": args.study,
        "experiment_count": len(results),
        "summary_csv": str(summary_path),
        "results": results,
    }


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    defaults = default_paths()
    parser.add_argument("--data-csv", default=str(defaults["data_csv"]))
    parser.add_argument("--stations-file", default=str(defaults["stations_file"]))
    parser.add_argument("--output-root", default=str(defaults["output_root"]))
    parser.add_argument("--lags", type=int, default=43)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--graph-type", choices=GRAPH_TYPES, default="correlation")
    parser.add_argument("--edge-ratio", type=float, default=0.33)
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SET_DEFS.keys()), default="full14")
    parser.add_argument("--graph-signal", default="T2M")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible 320-station ablation runner for graph construction, "
            "edge sparsity, and feature-set experiments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one configured experiment.")
    run_parser.add_argument("--name", required=True)
    add_common_run_args(run_parser)

    study_parser = subparsers.add_parser("study", help="Run a predefined ablation family.")
    study_parser.add_argument("--study", choices=STUDY_TYPES, required=True)
    add_common_run_args(study_parser)

    summary_parser = subparsers.add_parser("summarize", help="Build a summary CSV from finished study outputs.")
    summary_parser.add_argument("--study-dir", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = run_single_experiment(args)
    elif args.command == "study":
        result = run_study(args)
    elif args.command == "summarize":
        summary_path = summarize_results(Path(args.study_dir).resolve())
        result = {"summary_csv": str(summary_path)}
    else:  # pragma: no cover - argparse guards this path
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
