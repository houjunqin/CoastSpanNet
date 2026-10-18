"""Training logic for CoastSpanNet using the original hurricane-separated loader.

Expected dataset layout (same as the previous STERCHA trainer):

    dataset_root/training/<hurricane>/train.pt
    dataset_root/training/<hurricane>/validation.pt
    dataset_root/test/<hurricane>/test.pt

Each .pt package contains ``subgraphs``.  ``data`` is required; bathymetry,
Manning roughness, coordinates and graph edges are consumed only when their
corresponding model switches are enabled.  The saved data order remains
``[physical features, time_index, DOW, TOD, wet]``.  DOW and TOD can be added
as raw or cyclic calendar controls without rebuilding existing packages.
"""
from __future__ import annotations

import gc
import json
import random
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.CoastSpanNet.model import CoastSpanNet
from models.CoastSpanNet.evaluation import sums, metrics, evaluate


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 0.0):
        self.patience = int(patience)
        self.delta = float(delta)
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        else:
            self.best_score = score
            self.counter = 0


class TimeSeriesWindowDataset(Dataset):
    """Same lazy window loader as the previous STERCHA trainer.

    Saved subgraph data: [T_total, N, F_total]
    x: [input_len, N, F_total]
    y: [horizon, N, F_total]
    """
    def __init__(self, data: np.ndarray, input_len: int, horizon: int):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"Saved subgraph data must be [T,N,F], found {data.shape}")
        self.data = torch.from_numpy(data)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.sample_count = len(data) - self.input_len - self.horizon + 1
        if self.sample_count <= 0:
            raise ValueError(
                f"Split T={len(data)} is shorter than input_len+horizon="
                f"{self.input_len + self.horizon}"
            )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int):
        middle = index + self.input_len
        end = middle + self.horizon
        return self.data[index:middle], self.data[middle:end], index


class HurricaneDatasetStore:
    """Discover/load one hurricane package at a time, preserving low RAM usage."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.training_root = self.root / "training"
        self.test_root = self.root / "test"
        if not self.training_root.is_dir():
            raise FileNotFoundError(f"Training directory not found: {self.training_root}")
        if not self.test_root.is_dir():
            raise FileNotFoundError(f"Test directory not found: {self.test_root}")

        self.training_hurricanes = sorted(
            d.name for d in self.training_root.iterdir()
            if d.is_dir() and (d / "train.pt").is_file() and (d / "validation.pt").is_file()
        )
        self.test_hurricanes = sorted(
            d.name for d in self.test_root.iterdir()
            if d.is_dir() and (d / "test.pt").is_file()
        )
        if not self.training_hurricanes:
            raise ValueError(f"No complete training hurricanes under {self.training_root}")
        if not self.test_hurricanes:
            raise ValueError(f"No test hurricanes under {self.test_root}")

    @staticmethod
    def _load(path: Path, expected_split: str) -> dict[str, Any]:
        package = torch.load(path, map_location="cpu", weights_only=False)
        if package.get("split") != expected_split:
            raise ValueError(
                f"{path} declares split={package.get('split')!r}, expected {expected_split!r}"
            )
        if not package.get("subgraphs"):
            raise ValueError(f"{path} contains no subgraphs")
        return package

    def load_training(self, hurricane: str, split: str) -> dict[str, Any]:
        if hurricane not in self.training_hurricanes:
            raise KeyError(f"Unknown training hurricane: {hurricane}")
        if split not in {"train", "validation"}:
            raise ValueError("Training split must be 'train' or 'validation'")
        return self._load(self.training_root / hurricane / f"{split}.pt", split)

    def load_test(self, hurricane: str) -> dict[str, Any]:
        if hurricane not in self.test_hurricanes:
            raise KeyError(f"Unknown test hurricane: {hurricane}")
        return self._load(self.test_root / hurricane / "test.pt", "test")


def _numeric_mask(pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    """Return a numeric 0/1 mask with the same dtype/shape as ``pred``.

    ``wet`` is deliberately stored as floating point rather than torch.bool.
    No boolean advanced indexing is used by the losses/metrics.
    """
    wet_value = wet.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    wet_value = torch.nan_to_num(wet_value, nan=0.0).clamp(0.0, 1.0)
    valid = (wet_value > 0) & torch.isfinite(target)
    if ((~torch.isfinite(pred)) & valid).any():
        raise FloatingPointError("Non-finite prediction at a valid target")
    finite_value = torch.isfinite(target).to(pred.dtype)
    return wet_value * finite_value


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    mask = _numeric_mask(pred, target, wet)
    error = torch.nan_to_num((pred - target).abs(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = mask.sum().clamp_min(1.0)
    return (error * mask).sum() / denom


def _masked_rmse(pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    mask = _numeric_mask(pred, target, wet)
    error2 = torch.nan_to_num((pred - target).square(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt((error2 * mask).sum() / denom)


def _masked_mape(
    pred: torch.Tensor, target: torch.Tensor, wet: torch.Tensor, threshold: float = 1e-3
) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    mask = _numeric_mask(pred, target, wet)
    magnitude = (target.abs() >= threshold).to(pred.dtype)
    mask = mask * magnitude
    denom_target = target.abs().clamp_min(threshold)
    percentage = torch.nan_to_num((pred - target).abs() / denom_target, nan=0.0, posinf=0.0, neginf=0.0)
    denom = mask.sum().clamp_min(1.0)
    return 100.0 * (percentage * mask).sum() / denom


def _horizon_weights(horizon: int, alpha: float, device: torch.device) -> torch.Tensor:
    if horizon <= 1 or alpha <= 0:
        return torch.ones(horizon, device=device)
    h = torch.arange(horizon, device=device, dtype=torch.float32)
    w = 1.0 + alpha * h / float(horizon - 1)
    return w / w.mean()


def _masked_weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    wet: torch.Tensor,
    beta: float,
    horizon_alpha: float,
) -> torch.Tensor:
    """Smooth-L1 on wet nodes, with optional larger weights at later horizons."""
    h = min(pred.shape[1], target.shape[1], wet.shape[1])
    pred, target, wet = pred[:, :h], target[:, :h], wet[:, :h]
    pred, target = pred.float(), target.float()
    mask = _numeric_mask(pred, target, wet)
    # Mask BEFORE arithmetic: multiplying a NaN loss by zero is still NaN.
    safe_pred = torch.where(mask > 0, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask > 0, target, torch.zeros_like(target))
    element = F.smooth_l1_loss(safe_pred, safe_target, beta=beta, reduction="none")
    weights = _horizon_weights(h, horizon_alpha, pred.device).view(1, h, 1, 1)
    # Numeric 0/1 wet weights; avoid bool wet tensors and bool advanced indexing.
    mask = _numeric_mask(pred, target, wet).to(element.dtype)
    denom = (mask * weights).sum().clamp_min(1.0)
    return (element * mask * weights).sum() / denom


class Trainer:
    """CoastSpanNet trainer using the original hurricane/subgraph data pipeline."""

    def __init__(self, datasets: str | Path, **kwargs):
        self.store = HurricaneDatasetStore(datasets)
        self.cfg_model = {**kwargs.get("data", {}), **kwargs.get("CoastSpanNet", {})}
        seed = int(self.cfg_model.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if set(self.store.training_hurricanes) & set(self.store.test_hurricanes):
            raise ValueError("Training and held-out hurricane names must be disjoint")
        self.use_wet_mask = bool(self.cfg_model.get("use_wet_mask", True))
        self.wet_threshold = float(self.cfg_model.get("wet_threshold", 0.5))
        self.log_dir = self.cfg_model.get("log", "logs/CoastSpanNet/")
        self.model_file = os.path.join(self.log_dir, "best_model.pth")
        self.last_checkpoint_file = os.path.join(self.log_dir, "last_checkpoint.pth")
        self.resume_training = bool(kwargs.get("resume", self.cfg_model.get("resume", True)))

        self.input_feature_idx = list(self.cfg_model.get("input_feature_idx", [0, 1, 2]))
        self.output_feature_idx = list(self.cfg_model.get("output_feature_idx", [0, 1, 2]))
        self.use_bathymetry = bool(self.cfg_model.get("use_bathymetry", True))
        self.use_manning = bool(self.cfg_model.get("use_manning", True))
        self.use_time_of_day = bool(self.cfg_model.get("use_time_of_day", False))
        self.use_day_of_week = bool(self.cfg_model.get("use_day_of_week", False))
        self.cyclic_calendar = bool(self.cfg_model.get("cyclic_calendar", True))
        self.time_of_day_idx = int(self.cfg_model.get("time_of_day_idx", -2))
        self.day_of_week_idx = int(self.cfg_model.get("day_of_week_idx", -3))
        # The existing span packages store TOD as a fraction of one day and DOW
        # as 0..6.  Override these periods if a dataset uses different units.
        self.time_of_day_period = float(self.cfg_model.get("time_of_day_period", 1.0))
        self.day_of_week_period = float(self.cfg_model.get("day_of_week_period", 7.0))
        if self.cyclic_calendar:
            if self.use_time_of_day and self.time_of_day_period <= 0:
                raise ValueError("time_of_day_period must be positive")
            if self.use_day_of_week and self.day_of_week_period <= 0:
                raise ValueError("day_of_week_period must be positive")
        self.input_len = int(self.cfg_model.get("input_len", 12))
        self.horizon = int(self.cfg_model.get("horizon", 48))
        self.batch_size = int(self.cfg_model.get("batch_size", 1))
        self.num_workers = int(self.cfg_model.get("num_workers", 0))
        self.drop_last = bool(self.cfg_model.get("drop_last", True))

        # Direct is primary. Full rollout is mainly test/diagnostic output.
        self.train_mode = self.cfg_model.get("train_mode", "direct")
        self.val_mode = self.cfg_model.get("val_mode", "direct")
        self.test_mode = self.cfg_model.get("test_mode", "direct")
        self.rollout_steps = min(int(self.cfg_model.get("rollout_steps", self.horizon)), self.horizon)

        # Direct + occasional short rollout training.
        self.smooth_l1_beta = float(self.cfg_model.get("smooth_l1_beta", 0.05))
        self.horizon_weight_alpha = float(self.cfg_model.get("horizon_weight_alpha", 1.0))
        self.short_rollout_weight = float(self.cfg_model.get("short_rollout_weight", 0.10))
        self.short_rollout_steps = int(self.cfg_model.get("short_rollout_steps", 3))
        self.short_rollout_every = int(self.cfg_model.get("short_rollout_every", 4))
        self.short_rollout_start_epoch = int(self.cfg_model.get("short_rollout_start_epoch", 3))
        self.short_rollout_detach_feedback = bool(self.cfg_model.get("short_rollout_detach_feedback", True))
        self.short_rollout_fp32 = bool(self.cfg_model.get("short_rollout_fp32", True))
        self.short_rollout_streaming = bool(self.cfg_model.get("short_rollout_streaming", True))
        self.grad_clip = float(self.cfg_model.get("grad_clip", 1.0))
        self.rollout_batches = 0
        if self.short_rollout_weight > 0 and self.short_rollout_steps < 1:
            raise ValueError("Active short rollout requires positive steps")
        if not self.short_rollout_detach_feedback:
            raise ValueError("This memory-bounded trainer requires detached rollout feedback")
        if self.train_mode not in {"direct", "current"}:
            raise ValueError("Training uses direct loss plus optional short rollout")

        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_distributed()

        # float16 + GradScaler is the conservative CUDA default. BF16 can be selected
        # explicitly, but some CUDA/cuBLAS combinations fail during multi-forward
        # rollout backward with CUBLAS_STATUS_INTERNAL_ERROR.
        amp_dtype_name = str(self.cfg_model.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
        self.use_amp = self.device.type == "cuda" and bool(self.cfg_model.get("use_amp", True))
        self.gradscaler = amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        self.build_model()
        self._validate_saved_schema()

    def _setup_distributed(self) -> None:
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main_process = self.rank == 0

    def _amp_context(self):
        return amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        )

    def cleanup(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _log(self, message: str) -> None:
        if self.is_main_process:
            print(message, flush=True)
            with open(os.path.join(self.log_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def _reduce_sum_count(self, total: float, count: int) -> tuple[float, int]:
        if not self.distributed:
            return float(total), int(count)
        value = torch.tensor([total, count], dtype=torch.float64, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return float(value[0].item()), int(value[1].item())

    def build_model(self) -> None:
        model = CoastSpanNet(self.cfg_model).to(self.device)
        if self.distributed:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg_model.get("learning_rate", 1e-4)),
            weight_decay=float(self.cfg_model.get("weight_decay", 5e-4)),
            betas=tuple(self.cfg_model.get("betas", [0.9, 0.95])),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(self.cfg_model.get("scheduler_factor", 0.5)),
            patience=int(self.cfg_model.get("scheduler_patience", 4)),
            threshold=float(self.cfg_model.get("scheduler_threshold", 1e-4)),
        )

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _validate_saved_schema(self) -> None:
        hurricane = self.store.training_hurricanes[0]
        package = self.store.load_training(hurricane, "train")
        if "input_feature_idx" in package and list(package["input_feature_idx"]) != self.input_feature_idx:
            raise ValueError(
                f"input_feature_idx differs: trainer={self.input_feature_idx}, "
                f"dataset={package['input_feature_idx']}"
            )
        if "output_feature_idx" in package and list(package["output_feature_idx"]) != self.output_feature_idx:
            raise ValueError(
                f"output_feature_idx differs: trainer={self.output_feature_idx}, "
                f"dataset={package['output_feature_idx']}"
            )
        del package

    def _optimizer_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _restore_training_state(self, early_stopping: EarlyStopping):
        if not self.resume_training or not os.path.isfile(self.last_checkpoint_file):
            return 1, [], float("inf")
        checkpoint = torch.load(self.last_checkpoint_file, map_location=self.device, weights_only=False)
        self.unwrap_model().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._optimizer_to_device()
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("gradscaler_state_dict")
        if scaler_state:
            self.gradscaler.load_state_dict(scaler_state)
        early_state = checkpoint.get("early_stopping", {})
        early_stopping.counter = int(early_state.get("counter", 0))
        early_stopping.best_score = early_state.get("best_score")
        early_stopping.early_stop = False
        completed_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        best_val = float(checkpoint.get("best_val", float("inf")))
        self._log(
            f"Resumed from {self.last_checkpoint_file}: completed epoch {completed_epoch}, "
            f"next epoch {completed_epoch + 1}"
        )
        return completed_epoch + 1, history, best_val

    def _save_last_checkpoint(
        self, epoch: int, history: list[dict[str, Any]], best_val: float,
        early_stopping: EarlyStopping,
    ) -> None:
        if not self.is_main_process:
            return
        checkpoint = {
            "format_version": 2,
            "config": self.cfg_model,
            "epoch": int(epoch),
            "model_state_dict": self.unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "gradscaler_state_dict": self.gradscaler.state_dict(),
            "best_val": float(best_val),
            "history": history,
            "early_stopping": {
                "counter": int(early_stopping.counter),
                "best_score": early_stopping.best_score,
            },
            "dataset_root": str(self.store.root),
            "input_feature_idx": self.input_feature_idx,
            "output_feature_idx": self.output_feature_idx,
        }
        temporary = f"{self.last_checkpoint_file}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, self.last_checkpoint_file)

    def _make_loader(
        self, subgraph: dict[str, Any], shuffle: bool, epoch: int, stream_id: int,
        use_distributed: bool = True,
    ) -> DataLoader:
        dataset = TimeSeriesWindowDataset(
            subgraph["data"], input_len=self.input_len, horizon=self.horizon
        )
        sampler = None
        if self.distributed and use_distributed and not shuffle:
            # Unequal evaluation shards are safe with the unwrapped model.
            sampler = list(range(self.rank, len(dataset), self.world_size))
        elif self.distributed and use_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=self.drop_last,
            )
            sampler.set_epoch(epoch * 100_000 + stream_id)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=self.drop_last and shuffle,
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _first_existing(mapping: dict[str, Any], keys: tuple[str, ...]):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _localize_edge_index(edge_value, subgraph: dict[str, Any], n: int) -> np.ndarray:
        """Return a CPU int64 edge_index whose endpoints are local indices [0, N-1].

        Older span packages sometimes store edges using *global* mesh node IDs while
        the graph block indexes tensors of shape [B,N,D] using *local* subgraph indices.
        Passing global IDs directly to CUDA scatter/gather can trigger a device-side
        illegal memory access.  This routine detects that case and maps global IDs to
        local positions using ``subgraph['node_id']``.  Cross-subgraph edges are dropped.
        """
        edge = np.asarray(edge_value)
        if edge.ndim != 2:
            raise ValueError(f"edge_index must be rank 2, got {edge.shape}")
        if edge.shape[0] != 2 and edge.shape[1] == 2:
            edge = edge.T
        if edge.shape[0] != 2:
            raise ValueError(f"edge_index must be [2,E] or [E,2], got {edge.shape}")
        edge = np.asarray(edge, dtype=np.int64)
        if edge.size == 0:
            return edge.reshape(2, 0)

        e_min = int(edge.min())
        e_max = int(edge.max())
        space = subgraph.get("edge_index_space", "auto")
        if space not in {"auto", "local", "global"}:
            raise ValueError("edge_index_space must be auto, local, or global")
        if space != "global" and e_min >= 0 and e_max < n:
            return np.ascontiguousarray(edge)

        if "node_id" not in subgraph:
            raise ValueError(
                f"edge_index contains non-local node IDs: min={e_min}, max={e_max}, N={n}, "
                "but subgraph has no node_id array for global->local conversion."
            )

        node_id = np.asarray(subgraph["node_id"], dtype=np.int64).reshape(-1)
        if node_id.size != n:
            raise ValueError(f"node_id length {node_id.size} does not match N={n}")
        if np.unique(node_id).size != n:
            raise ValueError("node_id must contain unique global IDs within each subgraph")

        order = np.argsort(node_id)
        sorted_ids = node_id[order]
        flat = edge.reshape(-1)
        loc = np.searchsorted(sorted_ids, flat)
        valid = loc < n
        safe_loc = np.minimum(loc, n - 1)
        valid &= sorted_ids[safe_loc] == flat
        valid = valid.reshape(2, -1)
        keep = valid[0] & valid[1]

        dropped = int((~keep).sum())
        if not np.any(keep):
            raise ValueError(
                f"None of the {edge.shape[1]} edges can be mapped into this subgraph. "
                f"edge range=[{e_min},{e_max}], node_id range=[{int(node_id.min())},{int(node_id.max())}]."
            )

        mapped = order[safe_loc].reshape(2, -1)[:, keep]
        if mapped.min() < 0 or mapped.max() >= n:
            raise RuntimeError("Internal global->local edge mapping produced invalid local indices")
        if dropped:
            # Cross-boundary/full-mesh edges are expected when a package was built from
            # global connectivity.  Only edges with both endpoints in this span are kept.
            pass
        return np.ascontiguousarray(mapped, dtype=np.int64)

    def _graph_tensors(self, subgraph: dict[str, Any]):
        """Read enabled static/graph fields and validate indexing before CUDA."""
        data_shape = np.shape(subgraph.get("data"))
        if len(data_shape) != 3:
            raise ValueError(f"subgraph data must be [T,N,F], got {data_shape}")
        n = int(data_shape[1])

        static = None
        if subgraph.get("bathymetry_manning") is not None:
            static = np.asarray(subgraph["bathymetry_manning"], dtype=np.float32)
            if static.ndim != 2 or static.shape != (n, 2):
                raise ValueError(f"bathymetry_manning must be [N,2], got {static.shape}")

        bathymetry = None
        if self.use_bathymetry:
            bath_value = static[:, 0] if static is not None else self._first_existing(
                subgraph, ("bathymetry", "bathymetric_elevation", "elevation")
            )
            if bath_value is None:
                raise ValueError(
                    "use_bathymetry=True but no bathymetry_manning/bathymetry field exists"
                )
            bath_np = np.asarray(bath_value, dtype=np.float32).reshape(-1)
            if bath_np.shape != (n,) or not np.isfinite(bath_np).all():
                raise ValueError(f"bathymetry must be finite [N], got {bath_np.shape}")
            bathymetry = torch.as_tensor(bath_np, dtype=torch.float32, device=self.device)

        manning = None
        if self.use_manning:
            manning_value = static[:, 1] if static is not None else self._first_existing(
                subgraph, ("manning", "mannings_n", "roughness")
            )
            if manning_value is None:
                raise ValueError(
                    "use_manning=True but no bathymetry_manning/manning field exists"
                )
            manning_np = np.asarray(manning_value, dtype=np.float32).reshape(-1)
            if manning_np.shape != (n,) or not np.isfinite(manning_np).all():
                raise ValueError(f"Manning roughness must be finite [N], got {manning_np.shape}")
            manning = torch.as_tensor(manning_np, dtype=torch.float32, device=self.device)

        pos_value = self._first_existing(subgraph, ("pos", "coordinates", "lonlat", "longitude_latitude"))
        if pos_value is None and "longitude" in subgraph and "latitude" in subgraph:
            pos_value = np.column_stack([subgraph["longitude"], subgraph["latitude"]])
        if pos_value is None:
            pos = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        else:
            pos_np = np.asarray(pos_value, dtype=np.float32)
            if pos_np.ndim != 2 or pos_np.shape != (n, 2):
                raise ValueError(f"Expected pos [N,2], got {pos_np.shape} for N={n}")
            if not np.isfinite(pos_np).all():
                raise ValueError("pos contains NaN/Inf")
            pos = torch.as_tensor(pos_np, dtype=torch.float32, device=self.device)

        edge_value = self._first_existing(subgraph, ("edge_index", "fine_edge_index", "edges"))
        edge_index = None
        if edge_value is not None:
            edge_np = self._localize_edge_index(edge_value, subgraph, n)
            # These assertions happen on CPU, before any CUDA gather/scatter kernel.
            if edge_np.size:
                assert int(edge_np.min()) >= 0
                assert int(edge_np.max()) < n
            edge_index = torch.as_tensor(edge_np, dtype=torch.long, device=self.device)

        return bathymetry, manning, pos, edge_index

    def _calendar_channels(
        self, data: torch.Tensor, index: int, period: float, name: str
    ) -> torch.Tensor:
        """Select one calendar scalar and optionally convert it to sin/cos."""
        feature_count = data.shape[-1]
        resolved = index if index >= 0 else feature_count + index
        if resolved < 0 or resolved >= feature_count:
            raise IndexError(
                f"{name} index {index} is invalid for F_total={feature_count}"
            )
        value = data[..., resolved:resolved + 1].to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
        if not self.cyclic_calendar:
            return value
        angle = value * (2.0 * np.pi / period)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    def _append_calendar(self, physical: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        pieces = [physical]
        # Keep a deterministic order shared with CoastSpanNet's input-dimension logic.
        if self.use_day_of_week:
            pieces.append(self._calendar_channels(
                source, self.day_of_week_idx, self.day_of_week_period, "day_of_week"
            ))
        if self.use_time_of_day:
            pieces.append(self._calendar_channels(
                source, self.time_of_day_idx, self.time_of_day_period, "time_of_day"
            ))
        return torch.cat(pieces, dim=-1)

    def _prepare_batch(self, x: torch.Tensor, y: torch.Tensor):
        """Prepare selected physical inputs plus optional DOW/TOD controls.

        The return signature is unchanged for compatibility with ``evaluation``.
        Calendar channels are appended after physical channels; their future values
        are retained during rollout because they are deterministic known inputs.
        """
        physical_x = x[..., self.input_feature_idx].to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        x_model = self._append_calendar(physical_x, x)
        target = y[..., self.output_feature_idx].to(
            self.device, dtype=torch.float32, non_blocking=True
        )

        if self.use_wet_mask:
            wet = y[..., -1:].to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            wet = (wet > self.wet_threshold).to(dtype=torch.float32)
        else:
            wet = torch.ones(
                (*target.shape[:-1], 1),
                device=self.device, dtype=torch.float32
            )

        # Only selected physical/exogenous inputs are retained for rollout.
        future_physical = y[..., self.input_feature_idx].to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        # Future observations are NOT assumed to be operationally available.
        # Only explicitly declared known-future feature IDs may use them.
        allowed = set(self.cfg_model.get("known_future_feature_idx", []))
        future_physical = future_physical.clone()
        for j, feature in enumerate(self.input_feature_idx):
            if feature not in allowed:
                future_physical[..., j] = physical_x[:, -1:, :, j]
        future_inputs = self._append_calendar(future_physical, y)
        if not torch.isfinite(x_model).all():
            raise ValueError("History has NaN/Inf; impute inputs during preprocessing")
        return x_model, target, wet, future_inputs

    def _run_model(
        self, x, bathymetry, manning, pos, edge_index,
        mode: str, future_inputs=None,
        rollout_steps: int | None = None, detach_feedback: bool = False,
    ):
        normalized_mode = "direct" if mode in {"current", "direct"} else mode
        model = self.model if self.model.training else self.unwrap_model()
        if normalized_mode == "direct":
            return model(
                x, bathymetry, manning, pos, edge_index, mode="direct"
            )
        if normalized_mode == "rollout":
            steps = self.rollout_steps if rollout_steps is None else int(rollout_steps)
            return model(
                x, bathymetry, manning, pos, edge_index,
                mode="rollout",
                rollout_steps=steps,
                future_inputs=future_inputs[:, :steps] if future_inputs is not None else None,
                detach_feedback=detach_feedback,
            )
        raise ValueError(f"Unknown mode: {mode}")

    def _backward_short_rollout_streaming(
        self,
        x, target, wet, bathymetry, manning, pos, edge_index,
        future_inputs, k: int,
    ) -> torch.Tensor:
        """Memory-safe short rollout regularizer with optional calendar inputs."""
        cur_x = x
        total_value = 0.0
        base_model = self.unwrap_model()

        for s in range(k):
            rollout_amp_enabled = self.use_amp and not self.short_rollout_fp32
            with amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=rollout_amp_enabled,
            ):
                pred_block = self.model(
                    cur_x, bathymetry, manning, pos, edge_index, mode="direct", forecast_steps=1
                )
                pred_1 = pred_block[:, 0:1]
                step_loss = _masked_weighted_smooth_l1(
                    pred_1, target[:, s:s + 1], wet[:, s:s + 1],
                    beta=self.smooth_l1_beta, horizon_alpha=0.0,
                )

            scaled_step_loss = (self.short_rollout_weight / float(k)) * step_loss
            self.gradscaler.scale(scaled_step_loss).backward()
            total_value += float(step_loss.detach().item())

            feedback = pred_1.detach()
            if future_inputs is not None:
                next_frame = future_inputs[:, s:s + 1].clone()
            else:
                next_frame = cur_x[:, -1:].detach().clone()

            for out_j, local_j in enumerate(base_model.output_local_idx):
                if local_j >= 0:
                    next_frame[..., local_j] = feedback[..., out_j]

            cur_x = torch.cat([cur_x[:, 1:].detach(), next_frame], dim=1)
            del pred_block, pred_1, step_loss, scaled_step_loss

        return x.new_tensor(total_value / float(k))

    def _run_training_package(self, package: dict[str, Any], epoch: int):
        self.model.train()
        totals = np.zeros(6, dtype=np.float64)
        count = 0

        for position, subgraph_id in enumerate(sorted(package["subgraphs"])):
            subgraph = package["subgraphs"][subgraph_id]
            loader = self._make_loader(subgraph, True, epoch, position)
            bathymetry, manning, pos, edge_index = self._graph_tensors(subgraph)
            progress = tqdm(
                loader,
                desc=f"Rank {self.rank} | E{epoch} | {package['scenario']} | S{subgraph_id}",
                leave=False, disable=not self.is_main_process,
            )

            for batch_idx, (x_cpu, y_cpu, _) in enumerate(progress):
                x, target, wet, future_inputs = self._prepare_batch(x_cpu, y_cpu)
                # CPU batch tensors are no longer needed after transfer.
                del x_cpu, y_cpu
                self.optimizer.zero_grad(set_to_none=True)

                # ----- Main direct objective: one AMP forward -----
                with self._amp_context():
                    prediction = self._run_model(
                        x, bathymetry, manning, pos, edge_index, "direct"
                    )
                    direct_loss = _masked_weighted_smooth_l1(
                        prediction, target, wet,
                        beta=self.smooth_l1_beta,
                        horizon_alpha=self.horizon_weight_alpha,
                    )

                # Compute metrics before releasing the direct output.
                with torch.no_grad():
                    mae_value = float(_masked_mae(prediction, target, wet).item())
                    mape_value = float(_masked_mape(prediction, target, wet).item())
                    rmse_value = float(_masked_rmse(prediction, target, wet).item())
                direct_value = float(direct_loss.detach().item())

                self.gradscaler.scale(direct_loss).backward()
                # backward() has freed saved activations; remove output references too.
                del prediction, direct_loss

                do_short_rollout = (
                    self.short_rollout_weight > 0
                    and epoch >= self.short_rollout_start_epoch
                    and batch_idx % max(1, self.short_rollout_every) == 0
                )

                rollout_value = 0.0
                if do_short_rollout:
                    self.rollout_batches += 1
                    k = min(self.short_rollout_steps, target.shape[1])
                    if self.short_rollout_streaming:
                        rollout_loss_value = self._backward_short_rollout_streaming(
                            x, target, wet, bathymetry, manning, pos, edge_index,
                            future_inputs, k,
                        )
                        rollout_value = float(rollout_loss_value.item())
                    else:
                        # Legacy path: retained only for comparison; uses more memory.
                        rollout_amp_enabled = self.use_amp and not self.short_rollout_fp32
                        with amp.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype,
                            enabled=rollout_amp_enabled,
                        ):
                            rollout_prediction = self._run_model(
                                x, bathymetry, manning, pos, edge_index,
                                "rollout", future_inputs=future_inputs,
                                rollout_steps=k, detach_feedback=True,
                            )
                            rollout_loss = _masked_weighted_smooth_l1(
                                rollout_prediction, target[:, :k], wet[:, :k],
                                beta=self.smooth_l1_beta, horizon_alpha=0.0,
                            )
                        self.gradscaler.scale(
                            self.short_rollout_weight * rollout_loss
                        ).backward()
                        rollout_value = float(rollout_loss.detach().item())
                        del rollout_prediction, rollout_loss

                if self.gradscaler.is_enabled():
                    self.gradscaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
                self.gradscaler.step(self.optimizer)
                self.gradscaler.update()

                total_value = direct_value + self.short_rollout_weight * rollout_value
                totals += [
                    total_value, direct_value, rollout_value,
                    mae_value, mape_value, rmse_value,
                ]
                count += 1

                # Release large per-batch CUDA tensors promptly. Do not call
                # empty_cache() every iteration; it hurts performance and does not
                # reduce live tensor memory.
                del x, target, wet, future_inputs

        return totals, count

    @torch.no_grad()
    def _run_validation_package(self, package: dict[str, Any], epoch: int):
        self.model.eval()
        totals = np.zeros(5, dtype=np.float64)
        count = 0
        with torch.inference_mode():
            for position, subgraph_id in enumerate(sorted(package["subgraphs"])):
                subgraph = package["subgraphs"][subgraph_id]
                loader = self._make_loader(subgraph, False, epoch, position)
                bathymetry, manning, pos, edge_index = self._graph_tensors(subgraph)
                for x, y, _ in loader:
                    x, target, wet, future_inputs = self._prepare_batch(x, y)
                    with self._amp_context():
                        prediction = self._run_model(
                            x, bathymetry, manning, pos, edge_index,
                            self.val_mode,
                            future_inputs=future_inputs,
                        )
                    steps = prediction.shape[1]
                    totals += sums(prediction, target[:, :steps], wet[:, :steps]).sum((1, 2))
                    count += 1
        return totals, count

    def _release(self, package: dict[str, Any]) -> None:
        del package
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def train(self):
        if self.is_main_process:
            with open(os.path.join(self.log_dir, 'effective_config.json'), 'w') as f:
                json.dump(dict(requested=self.cfg_model, resolved_training={
                    key: getattr(self, key) for key in (
                        'short_rollout_weight', 'short_rollout_steps', 'short_rollout_every',
                        'short_rollout_start_epoch', 'short_rollout_fp32', 'short_rollout_streaming',
                        'short_rollout_detach_feedback', 'batch_size', 'drop_last', 'use_amp',
                        'input_len', 'horizon', 'train_mode', 'val_mode', 'resume_training',
                        'use_bathymetry', 'use_manning', 'use_time_of_day',
                        'use_day_of_week', 'cyclic_calendar')
                }), f, indent=2, default=str)
        self._log(f"Short rollout: weight={self.short_rollout_weight}, steps={self.short_rollout_steps}, "
                  f"every={self.short_rollout_every}, start={self.short_rollout_start_epoch}, "
                  f"fp32={self.short_rollout_fp32}")
        self._log("====================== START TRAINING ======================")
        self._log(f"Training hurricanes: {self.store.training_hurricanes}")
        self._log(f"Held-out tests: {self.store.test_hurricanes}")
        self._log(
            f"Model switches: persistence={self.unwrap_model().use_persistence}, "
            f"temporal={self.unwrap_model().use_temporal}, "
            f"fine_graph={self.unwrap_model().use_fine_graph}, "
            f"bathymetry={self.use_bathymetry}, manning={self.use_manning}, "
            f"TOD={self.use_time_of_day}, DOW={self.use_day_of_week}, "
            f"cyclic_calendar={self.cyclic_calendar}"
        )
        start_total = time.time()
        early_stopping = EarlyStopping(self.cfg_model.get("patience", 20))
        start_epoch, history, best_val = self._restore_training_state(early_stopping)

        for epoch in range(start_epoch, int(self.cfg_model.get("epochs", 80)) + 1):
            self.rollout_batches = 0
            epoch_start = time.time()
            train_total, train_count = np.zeros(6), 0
            for hurricane in self.store.training_hurricanes:
                package = self.store.load_training(hurricane, "train")
                values, n = self._run_training_package(package, epoch)
                train_total += values
                train_count += n
                self._release(package)

            val_total, val_count = np.zeros(5), 0
            for hurricane in self.store.training_hurricanes:
                package = self.store.load_training(hurricane, "validation")
                values, n = self._run_validation_package(package, epoch)
                val_total += values
                val_count += n
                self._release(package)

            train_metrics = []
            for value in train_total:
                total, n = self._reduce_sum_count(value, train_count)
                train_metrics.append(total / max(n, 1))
            pooled = metrics(np.array([self._reduce_sum_count(v, 0)[0] for v in val_total]))
            val_metrics = [pooled['MAE'], pooled['MAPE'], pooled['RMSE']]
            if not np.isfinite(pooled['MAE']):
                raise ValueError('Validation has no valid wet targets')
            rollout_batches = self._reduce_sum_count(self.rollout_batches, 0)[0]
            if train_count == 0:
                raise ValueError('No training batches; disable drop_last or reduce batch_size')
            if self.short_rollout_weight > 0 and epoch >= self.short_rollout_start_epoch and rollout_batches == 0:
                raise RuntimeError('Short rollout was scheduled but never executed')

            train_loss, train_direct, train_roll, train_mae, train_mape, train_rmse = train_metrics
            val_mae, val_mape, val_rmse = val_metrics
            self.lr_scheduler.step(val_mae)

            if self.is_main_process:
                row = {
                    "Epoch": epoch,
                    "Short_Rollout_Batches": int(rollout_batches),
                    "Val_Valid_Count": pooled['Valid_Count'],
                    "Train_Loss": train_loss,
                    "Train_Direct_Loss": train_direct,
                    "Train_Short_Rollout_Loss": train_roll,
                    "Train_MAE": train_mae,
                    "Train_MAPE": train_mape,
                    "Train_RMSE": train_rmse,
                    "Val_MAE": val_mae,
                    "Val_MAPE": val_mape,
                    "Val_RMSE": val_rmse,
                    "LR": self.optimizer.param_groups[0]["lr"],
                    "Time_sec": time.time() - epoch_start,
                    "GPU_Mem_MB": (
                        torch.cuda.max_memory_allocated(self.device) / 1024**2
                        if self.device.type == "cuda" else 0.0
                    ),
                }
                history.append(row)
                self._log(
                    f"Epoch {epoch:03d} | Train Loss {train_loss:.5f} | "
                    f"Train MAE {train_mae:.5f} | Val MAE {val_mae:.5f} | "
                    f"Val RMSE {val_rmse:.5f} | LR {row['LR']:.2e} | "
                    f"Time {row['Time_sec']:.1f}s"
                )
                if val_mae < best_val:
                    best_val = val_mae
                    torch.save(self.unwrap_model().state_dict(), self.model_file)
                    self._log(f"Saved best model: validation MAE={best_val:.6f}")
                early_stopping(val_mae)
                stop_now = early_stopping.early_stop
                pd.DataFrame(history).to_excel(
                    os.path.join(self.log_dir, "training_results.xlsx"),
                    sheet_name="Epoch_History",
                    index=False,
                )
            else:
                stop_now = False

            self._save_last_checkpoint(epoch, history, best_val, early_stopping)
            if self.distributed:
                dist.barrier()
                flag = torch.tensor(int(stop_now), device=self.device)
                dist.broadcast(flag, src=0)
                stop_now = bool(flag.item())
            if stop_now:
                break

        if self.distributed:
            dist.barrier()
        return self.test(start_time=start_total)

    @torch.no_grad()
    def test(
        self,
        start_time: float | None = None,
        save_npz: bool = False,
        load_best: bool = True,
        modes: tuple[str, ...] | None = None,
    ):
        return evaluate(self, load_best=load_best, modes=modes, save_npz=save_npz)
