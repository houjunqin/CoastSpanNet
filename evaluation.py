"""Evaluation and prediction export for CoastSpanNet.

The default evaluation produces both direct and autoregressive-rollout outputs.
For every held-out hurricane and forecast mode it writes:

1. one compressed ``.npz`` archive per subgraph; and
2. disk-backed ``.npy`` arrays reconstructed on the complete graph using the
   saved global ``node_id`` values.

The whole-graph writer uses a running mean for nodes shared by overlapping
subgraphs.  This avoids allocating another complete prediction tensor in RAM.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist


def _numeric_arrays(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    mape_threshold: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-batch/per-horizon metric sums on valid wet targets."""
    pred = pred.detach().float()
    target = target.detach().float()
    wet = wet.detach().float().expand_as(pred)

    valid = (wet > 0) & torch.isfinite(target) & torch.isfinite(pred)
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = safe_pred - safe_target

    reduce_dims = (2, 3)
    abs_sum = (error.abs() * valid).sum(dim=reduce_dims)
    sq_sum = (error.square() * valid).sum(dim=reduce_dims)
    valid_count = valid.sum(dim=reduce_dims)

    percentage_valid = valid & (safe_target.abs() >= mape_threshold)
    ape = error.abs() / safe_target.abs().clamp_min(mape_threshold)
    ape_sum = (ape * percentage_valid).sum(dim=reduce_dims)
    percentage_count = percentage_valid.sum(dim=reduce_dims)

    return tuple(
        value.cpu().numpy().astype(np.float64, copy=False)
        for value in (abs_sum, ape_sum, sq_sum, valid_count, percentage_count)
    )


def sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    mape_threshold: float = 1.0e-3,
) -> np.ndarray:
    """Return [5,B,H] accumulators used by training and evaluation.

    Rows are absolute-error sum, absolute-percentage-error sum, squared-error
    sum, valid count, and percentage-valid count.
    """
    return np.stack(_numeric_arrays(pred, target, wet, mape_threshold), axis=0)


def metrics(total: np.ndarray) -> dict[str, float | int]:
    """Convert the five pooled accumulators into reporting metrics."""
    value = np.asarray(total, dtype=np.float64).reshape(5, -1).sum(axis=1)
    abs_sum, ape_sum, sq_sum, valid_count, percentage_count = value
    if valid_count <= 0:
        return {
            "MAE": float("nan"),
            "MAPE": float("nan"),
            "RMSE": float("nan"),
            "Valid_Count": 0,
            "MAPE_Valid_Count": int(percentage_count),
        }
    return {
        "MAE": float(abs_sum / valid_count),
        "MAPE": float(100.0 * ape_sum / percentage_count)
        if percentage_count > 0 else float("nan"),
        "RMSE": float(np.sqrt(sq_sum / valid_count)),
        "Valid_Count": int(valid_count),
        "MAPE_Valid_Count": int(percentage_count),
    }


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unnamed"


def _ordered_subgraphs(package: dict[str, Any]) -> list[tuple[Any, dict[str, Any]]]:
    return [(key, package["subgraphs"][key]) for key in sorted(
        package["subgraphs"], key=lambda item: str(item)
    )]


def _build_node_layout(
    package: dict[str, Any],
) -> tuple[np.ndarray, dict[Any, np.ndarray], bool]:
    """Return complete node IDs and each subgraph's complete-graph positions."""
    ordered = _ordered_subgraphs(package)
    all_have_ids = all("node_id" in subgraph for _, subgraph in ordered)
    local_ids: dict[Any, np.ndarray] = {}

    if all_have_ids:
        for key, subgraph in ordered:
            n = int(np.shape(subgraph["data"])[1])
            ids = np.asarray(subgraph["node_id"], dtype=np.int64).reshape(-1)
            if ids.size != n:
                raise ValueError(
                    f"Subgraph {key!r}: node_id length {ids.size} does not match N={n}"
                )
            if np.unique(ids).size != n:
                raise ValueError(f"Subgraph {key!r}: node_id values must be unique")
            local_ids[key] = ids
        complete_ids = np.unique(np.concatenate(list(local_ids.values())))
        positions = {
            key: np.searchsorted(complete_ids, ids).astype(np.int64, copy=False)
            for key, ids in local_ids.items()
        }
        return complete_ids, positions, False

    # Backward-compatible fallback for packages without global IDs.  The saved
    # graph is a deterministic concatenation, not the original mesh numbering.
    positions: dict[Any, np.ndarray] = {}
    offset = 0
    for key, subgraph in ordered:
        n = int(np.shape(subgraph["data"])[1])
        positions[key] = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
    return np.arange(offset, dtype=np.int64), positions, True


def _subgraph_node_id(
    subgraph: dict[str, Any], complete_ids: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    if "node_id" in subgraph:
        return np.asarray(subgraph["node_id"], dtype=np.int64).reshape(-1)
    return complete_ids[positions]


def _fill_memmap(array: np.memmap, value: float, node_chunk: int = 1024) -> None:
    """Initialize a large [S,H,N,C] memmap without a same-size RAM allocation."""
    nodes = array.shape[2]
    for start in range(0, nodes, node_chunk):
        array[:, :, start:start + node_chunk, :] = value
    array.flush()


def _create_whole_graph_arrays(
    directory: Path,
    sample_count: int,
    steps: int,
    node_count: int,
    output_dim: int,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    shape = (sample_count, steps, node_count, output_dim)
    wet_shape = (sample_count, steps, node_count, 1)
    prediction = np.lib.format.open_memmap(
        directory / "prediction.npy", mode="w+", dtype=np.float32, shape=shape
    )
    target = np.lib.format.open_memmap(
        directory / "target.npy", mode="w+", dtype=np.float32, shape=shape
    )
    wet = np.lib.format.open_memmap(
        directory / "wet.npy", mode="w+", dtype=np.float32, shape=wet_shape
    )
    _fill_memmap(prediction, np.nan)
    _fill_memmap(target, np.nan)
    _fill_memmap(wet, 0.0)
    return {
        "prediction": prediction,
        "target": target,
        "wet": wet,
        "overlap_count": np.zeros(node_count, dtype=np.uint32),
    }


def _running_mean_insert(
    destination: np.memmap,
    source: np.ndarray,
    global_position: np.ndarray,
    old_count: np.ndarray,
    node_chunk: int = 512,
) -> None:
    """Insert one subgraph into a whole-graph memmap, averaging overlaps."""
    for start in range(0, len(global_position), node_chunk):
        stop = min(start + node_chunk, len(global_position))
        index = global_position[start:stop]
        count = old_count[start:stop].astype(np.float32)
        incoming = source[:, :, start:stop, :]
        existing = np.asarray(destination[:, :, index, :])
        fresh = count == 0
        if np.all(fresh):
            combined = incoming
        else:
            weight = count.reshape(1, 1, -1, 1)
            safe_existing = np.where(np.isfinite(existing), existing, 0.0)
            combined = (safe_existing * weight + incoming) / (weight + 1.0)
        destination[:, :, index, :] = combined


def _write_subgraph_archive(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    wet: np.ndarray,
    window_index: np.ndarray,
    node_id: np.ndarray,
    position: np.ndarray,
    bathymetry: np.ndarray | None,
    manning: np.ndarray | None,
) -> None:
    payload: dict[str, np.ndarray] = {
        "prediction": prediction.astype(np.float32, copy=False),
        "target": target.astype(np.float32, copy=False),
        "wet": wet.astype(np.float32, copy=False),
        "window_index": window_index.astype(np.int64, copy=False),
        "node_id": node_id.astype(np.int64, copy=False),
        "position": position.astype(np.float32, copy=False),
    }
    if bathymetry is not None:
        payload["bathymetry"] = bathymetry.astype(np.float32, copy=False)
    if manning is not None:
        payload["manning"] = manning.astype(np.float32, copy=False)
    np.savez_compressed(path, **payload)


def _whole_graph_metric_sums(
    prediction: np.memmap,
    target: np.memmap,
    wet: np.memmap,
    node_chunk: int = 256,
    mape_threshold: float = 1.0e-3,
) -> np.ndarray:
    total = np.zeros(5, dtype=np.float64)
    for start in range(0, prediction.shape[2], node_chunk):
        stop = min(start + node_chunk, prediction.shape[2])
        pred = np.asarray(prediction[:, :, start:stop, :], dtype=np.float64)
        truth = np.asarray(target[:, :, start:stop, :], dtype=np.float64)
        mask_value = np.broadcast_to(
            np.asarray(wet[:, :, start:stop, :], dtype=np.float64), pred.shape
        )
        valid = (mask_value > 0) & np.isfinite(pred) & np.isfinite(truth)
        error = np.where(valid, pred - truth, 0.0)
        total[0] += np.abs(error).sum()
        total[2] += np.square(error).sum()
        total[3] += valid.sum()
        pvalid = valid & (np.abs(truth) >= mape_threshold)
        total[1] += np.where(
            pvalid, np.abs(error) / np.maximum(np.abs(truth), mape_threshold), 0.0
        ).sum()
        total[4] += pvalid.sum()
    return total


def _save_metric_table(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    columns = [
        "scenario", "mode", "scope", "subgraph", "MAE", "MAPE", "RMSE",
        "Valid_Count", "MAPE_Valid_Count", "samples", "forecast_steps",
        "nodes", "prediction_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_modes(modes: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = ("direct", "rollout") if modes is None else tuple(modes)
    normalized: list[str] = []
    for mode in selected:
        item = "direct" if mode in {"current", "direct"} else str(mode)
        if item not in {"direct", "rollout"}:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("At least one evaluation mode is required")
    return tuple(normalized)


@torch.no_grad()
def evaluate(
    trainer,
    load_best: bool = True,
    modes: tuple[str, ...] | None = None,
    save_npz: bool = True,
) -> dict[str, Any]:
    """Evaluate held-out hurricanes and optionally save all prediction fields."""
    selected_modes = _normalize_modes(modes)

    # Only rank zero performs complete-domain export.  Other ranks wait so the
    # caller can continue safely after evaluation without file-write collisions.
    if trainer.distributed:
        dist.barrier()
        if not trainer.is_main_process:
            dist.barrier()
            return {}

    model = trainer.unwrap_model()
    if load_best:
        if not os.path.isfile(trainer.model_file):
            raise FileNotFoundError(f"Best model not found: {trainer.model_file}")
        state = torch.load(trainer.model_file, map_location=trainer.device, weights_only=True)
        model.load_state_dict(state)
    model.eval()

    output_root = Path(trainer.cfg_model.get(
        "prediction_output_dir", os.path.join(trainer.log_dir, "predictions")
    ))
    if save_npz:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "output_directory": str(output_root),
        "modes": list(selected_modes),
        "scenarios": {},
    }

    for hurricane in trainer.store.test_hurricanes:
        package = trainer.store.load_test(hurricane)
        scenario = str(package.get("scenario", hurricane))
        scenario_name = _safe_name(scenario)
        complete_ids, graph_positions, synthetic_ids = _build_node_layout(package)
        ordered = _ordered_subgraphs(package)

        sample_counts = {
            len(trainer._make_loader(subgraph, False, 0, position, use_distributed=False).dataset)
            for position, (_, subgraph) in enumerate(ordered)
        }
        if len(sample_counts) != 1:
            raise ValueError(
                f"Scenario {scenario!r} has unequal subgraph window counts: "
                f"{sorted(sample_counts)}. Whole-graph reconstruction requires aligned windows."
            )
        sample_count = sample_counts.pop()
        scenario_result: dict[str, Any] = {
            "global_node_count": int(len(complete_ids)),
            "synthetic_node_ids": bool(synthetic_ids),
            "modes": {},
        }

        for mode in selected_modes:
            mode_dir = output_root / scenario_name / mode
            subgraph_dir = mode_dir / "subgraphs"
            whole_dir = mode_dir / "entire_graph"
            if save_npz:
                subgraph_dir.mkdir(parents=True, exist_ok=True)

            whole = None
            whole_steps = None
            full_position = np.full((len(complete_ids), 2), np.nan, dtype=np.float32)
            position_count = np.zeros(len(complete_ids), dtype=np.uint32)
            window_reference: np.ndarray | None = None
            pooled_total = np.zeros(5, dtype=np.float64)
            subgraph_outputs: list[str] = []

            for position_number, (subgraph_id, subgraph) in enumerate(ordered):
                loader = trainer._make_loader(
                    subgraph, False, 0, position_number, use_distributed=False
                )
                bathymetry, manning, pos, edge_index = trainer._graph_tensors(subgraph)
                predictions: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                wets: list[np.ndarray] = []
                indices: list[np.ndarray] = []
                sub_total = np.zeros(5, dtype=np.float64)

                for x_cpu, y_cpu, index_cpu in loader:
                    x, target, wet, future = trainer._prepare_batch(x_cpu, y_cpu)
                    if mode == "rollout":
                        available = min(
                            int(target.shape[1]),
                            int(future.shape[1]),
                            int(trainer.rollout_steps),
                        )
                        pred = trainer._run_model(
                            x, bathymetry, manning, pos, edge_index,
                            "rollout", future_inputs=future,
                            rollout_steps=available,
                            detach_feedback=True,
                        )
                    else:
                        pred = trainer._run_model(
                            x, bathymetry, manning, pos, edge_index, "direct"
                        )

                    steps = min(int(pred.shape[1]), int(target.shape[1]))
                    pred = pred[:, :steps]
                    target_step = target[:, :steps]
                    wet_step = wet[:, :steps]
                    batch_total = sums(pred, target_step, wet_step).sum(axis=(1, 2))
                    sub_total += batch_total
                    pooled_total += batch_total

                    if save_npz:
                        predictions.append(pred.detach().float().cpu().numpy())
                        targets.append(target_step.detach().float().cpu().numpy())
                        wets.append(wet_step.detach().float().cpu().numpy())
                        indices.append(np.asarray(index_cpu, dtype=np.int64).reshape(-1))

                sub_metric = metrics(sub_total)
                if not save_npz:
                    rows.append({
                        "scenario": scenario, "mode": mode, "scope": "subgraph",
                        "subgraph": str(subgraph_id), **sub_metric,
                        "samples": len(loader.dataset), "forecast_steps": steps,
                        "nodes": int(np.shape(subgraph["data"])[1]),
                        "prediction_path": "",
                    })
                    continue

                pred_np = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
                target_np = np.concatenate(targets, axis=0).astype(np.float32, copy=False)
                wet_np = np.concatenate(wets, axis=0).astype(np.float32, copy=False)
                window_index = np.concatenate(indices).astype(np.int64, copy=False)
                order = np.argsort(window_index, kind="stable")
                pred_np = pred_np[order]
                target_np = target_np[order]
                wet_np = wet_np[order]
                window_index = window_index[order]

                if window_reference is None:
                    window_reference = window_index
                elif not np.array_equal(window_reference, window_index):
                    raise ValueError(
                        f"Scenario {scenario!r}: subgraph {subgraph_id!r} has "
                        "misaligned window indices"
                    )

                if whole is None:
                    whole_steps = int(pred_np.shape[1])
                    whole = _create_whole_graph_arrays(
                        whole_dir, sample_count, whole_steps, len(complete_ids),
                        int(pred_np.shape[-1]),
                    )
                elif int(pred_np.shape[1]) != whole_steps:
                    raise ValueError("Subgraphs produced inconsistent forecast lengths")

                graph_index = graph_positions[subgraph_id]
                old_count = whole["overlap_count"][graph_index].copy()
                _running_mean_insert(
                    whole["prediction"], pred_np, graph_index, old_count
                )
                _running_mean_insert(
                    whole["target"], target_np, graph_index, old_count
                )
                _running_mean_insert(
                    whole["wet"], wet_np, graph_index, old_count
                )
                whole["overlap_count"][graph_index] += 1

                pos_np = pos.detach().float().cpu().numpy()
                for local_start in range(0, len(graph_index), 4096):
                    local_stop = min(local_start + 4096, len(graph_index))
                    idx = graph_index[local_start:local_stop]
                    count = position_count[idx].astype(np.float32)
                    incoming = pos_np[local_start:local_stop]
                    fresh = count == 0
                    if np.all(fresh):
                        full_position[idx] = incoming
                    else:
                        safe = np.where(np.isfinite(full_position[idx]), full_position[idx], 0.0)
                        full_position[idx] = (
                            safe * count[:, None] + incoming
                        ) / (count[:, None] + 1.0)
                    position_count[idx] += 1

                node_id = _subgraph_node_id(subgraph, complete_ids, graph_index)
                archive = subgraph_dir / f"subgraph_{_safe_name(subgraph_id)}.npz"
                bath_np = None if bathymetry is None else bathymetry.detach().cpu().numpy().reshape(-1)
                mann_np = None if manning is None else manning.detach().cpu().numpy().reshape(-1)
                _write_subgraph_archive(
                    archive, pred_np, target_np, wet_np, window_index, node_id,
                    pos_np, bath_np, mann_np,
                )
                subgraph_outputs.append(str(archive))
                rows.append({
                    "scenario": scenario, "mode": mode, "scope": "subgraph",
                    "subgraph": str(subgraph_id), **sub_metric,
                    "samples": int(pred_np.shape[0]),
                    "forecast_steps": int(pred_np.shape[1]),
                    "nodes": int(pred_np.shape[2]),
                    "prediction_path": str(archive),
                })

                del pred_np, target_np, wet_np, predictions, targets, wets

            if save_npz:
                if whole is None or window_reference is None:
                    raise RuntimeError(f"No predictions were produced for {scenario}/{mode}")
                for name in ("prediction", "target", "wet"):
                    whole[name].flush()
                np.save(whole_dir / "node_id.npy", complete_ids)
                np.save(whole_dir / "position.npy", full_position)
                np.save(whole_dir / "window_index.npy", window_reference)
                np.save(whole_dir / "overlap_count.npy", whole["overlap_count"])

                whole_total = _whole_graph_metric_sums(
                    whole["prediction"], whole["target"], whole["wet"]
                )
                whole_metric = metrics(whole_total)
                metadata = {
                    "scenario": scenario,
                    "mode": mode,
                    "prediction_shape": list(whole["prediction"].shape),
                    "target_shape": list(whole["target"].shape),
                    "wet_shape": list(whole["wet"].shape),
                    "node_id_is_synthetic": bool(synthetic_ids),
                    "input_feature_idx": list(trainer.input_feature_idx),
                    "output_feature_idx": list(trainer.output_feature_idx),
                    "input_len": int(trainer.input_len),
                    "configured_horizon": int(trainer.horizon),
                    "effective_forecast_steps": int(whole_steps),
                    "files": {
                        "prediction": "prediction.npy",
                        "target": "target.npy",
                        "wet": "wet.npy",
                        "node_id": "node_id.npy",
                        "position": "position.npy",
                        "window_index": "window_index.npy",
                        "overlap_count": "overlap_count.npy",
                    },
                    "metrics": whole_metric,
                }
                with (whole_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                    json.dump(metadata, handle, indent=2, allow_nan=True)

                rows.append({
                    "scenario": scenario, "mode": mode, "scope": "entire_graph",
                    "subgraph": "", **whole_metric,
                    "samples": sample_count,
                    "forecast_steps": int(whole_steps),
                    "nodes": int(len(complete_ids)),
                    "prediction_path": str(whole_dir / "prediction.npy"),
                })
                scenario_result["modes"][mode] = {
                    "metrics": whole_metric,
                    "subgraph_outputs": subgraph_outputs,
                    "entire_graph_directory": str(whole_dir),
                    "prediction_shape": list(whole["prediction"].shape),
                }
                del whole
            else:
                scenario_result["modes"][mode] = {
                    "metrics": metrics(pooled_total),
                    "subgraph_outputs": [],
                    "entire_graph_directory": None,
                }

        result["scenarios"][scenario] = scenario_result
        trainer._release(package)

    if save_npz:
        _save_metric_table(output_root / "test_metrics.csv", rows)
        with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=True)
        trainer._log(f"Saved prediction outputs: {output_root}")

    if trainer.distributed:
        dist.barrier()
    return result
