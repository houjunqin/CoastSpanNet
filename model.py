import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMLPBlock(nn.Module):
    """Variable-node-safe residual MLP operating independently at each node.

    Input / output: [B, D, N]
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)                  # [B,N,D]
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2).contiguous()     # [B,D,N]
        return residual + x


class TemporalLite(nn.Module):
    """Cheap explicit temporal encoder shared by every node.

    Input:  history [B,T,N,C]
    Output: node temporal summary [B,N,D_t]

    Internally the node axis is folded into the batch axis so temporal
    convolutions are applied independently to each node.
    """

    def __init__(
        self,
        input_dim: int,
        temporal_dim: int = 64,
        layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, temporal_dim, kernel_size=1)
        blocks = []
        for i in range(layers):
            k = kernel_size if i % 2 == 0 else max(3, kernel_size + 2)
            if k % 2 == 0:
                k += 1
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        temporal_dim,
                        temporal_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=temporal_dim,
                    ),
                    nn.GELU(),
                    nn.Conv1d(temporal_dim, temporal_dim, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(temporal_dim)
        self.summary_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        b, t, n, c = history.shape
        x = history.permute(0, 2, 3, 1).contiguous()      # [B,N,C,T]
        x = x.view(b * n, c, t)                           # [B*N,C,T]
        x = self.input_proj(x)                            # [B*N,D,T]
        for block in self.blocks:
            x = x + block(x)

        # A light learned blend of recent-state and whole-window information.
        last = x[:, :, -1]                                # [B*N,D]
        mean = x.mean(dim=-1)                             # [B*N,D]
        gate = torch.sigmoid(self.summary_gate)
        summary = gate * last + (1.0 - gate) * mean
        summary = self.norm(summary)
        return summary.view(b, n, -1)                     # [B,N,D]


class PositionEmbedding(nn.Module):
    """Coordinate-derived spatial embedding, independent of node numbering."""

    def __init__(self, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or max(32, out_dim)
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pos: torch.Tensor, batch_size: int) -> torch.Tensor:
        if pos.dim() == 2:
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        elif pos.dim() != 3:
            raise ValueError(f"pos must be [N,2] or [B,N,2], got {tuple(pos.shape)}")
        return self.net(pos.float())                      # [B,N,Dp]


class FineGraphBlock(nn.Module):
    """Dependency-free mean-aggregation graph block.

    Input / output: x [B,N,D]
    edge_index: [2,E] with local node indices for the current span/subgraph.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg = nn.Linear(dim, dim, bias=False)
        self.self_proj = nn.Linear(dim, dim)
        self.update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)
        edge_index = edge_index.to(dtype=torch.long, non_blocking=True).contiguous()
        src, dst = edge_index[0], edge_index[1]
        b, n, d = x.shape

        # Fail synchronously before launching gather/aggregation kernels.
        if src.numel() > 0:
            e_min = int(torch.minimum(src.min(), dst.min()).item())
            e_max = int(torch.maximum(src.max(), dst.max()).item())
            if e_min < 0 or e_max >= n:
                raise IndexError(
                    f"FineGraph edge_index is out of range for local span: "
                    f"min={e_min}, max={e_max}, N={n}"
                )

        # index_select + index_add avoids the large expanded scatter index tensor
        # [B,E,D] that can be fragile and memory-heavy on large CUDA graphs.
        # Linear projection commutes with gathering. Project N nodes once,
        # then gather bounded chunks instead of projecting E edge messages.
        projected = self.msg(x)
        agg = torch.zeros_like(x)
        for start in range(0, src.numel(), 65536):
            stop = start + 65536
            agg.index_add_(1, dst[start:stop], projected.index_select(1, src[start:stop]))

        # Degree has no gradient requirement; bincount is simpler/safer than scatter.
        degree = torch.bincount(dst, minlength=n).to(device=x.device, dtype=x.dtype)
        agg = agg / degree.clamp_min(1.0).view(1, n, 1)

        h = self.self_proj(x) + agg
        return x + self.update(h)


class FineGraphCorrection(nn.Module):
    """Bottleneck fine-graph correction with a small learnable residual gate."""

    def __init__(
        self,
        model_dim: int,
        graph_dim: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        gate_init: float = -2.0,
        force_fp32: bool = True,
    ):
        super().__init__()
        self.down = nn.Linear(model_dim, graph_dim)
        self.blocks = nn.ModuleList([FineGraphBlock(graph_dim, dropout) for _ in range(layers)])
        self.up = nn.Linear(graph_dim, model_dim)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.force_fp32 = bool(force_fp32)

    def forward(self, hidden_bdn: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_bdn.dtype
        x = hidden_bdn.transpose(1, 2).contiguous()       # [B,N,D]

        # Graph gather/aggregation is intentionally kept in FP32 by default.
        # This is a small bottleneck branch and avoids half-precision index kernels
        # on very large span graphs.
        if self.force_fp32 and x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                g = self.down(x.float())
                for block in self.blocks:
                    g = block(g, edge_index)
                corr = self.up(g).transpose(1, 2).contiguous()
            corr = corr.to(original_dtype)
        else:
            g = self.down(x)
            for block in self.blocks:
                g = block(g, edge_index)
            corr = self.up(g).transpose(1, 2).contiguous()

        gate = torch.sigmoid(self.gate_logit).to(dtype=corr.dtype)
        return hidden_bdn + gate * corr


class CoastSpanNet(nn.Module):
    """CoastSpanNet upgraded for variable-size span/subgraph datasets.

    Main changes from the supplied CoastSpanNet implementation:
      1. persistence-residual output,
      2. explicit lightweight temporal encoder,
      3. optional bottleneck fine-graph correction,
      4. coordinate-based spatial embedding instead of fixed node-ID embedding,
      5. feature-dimension normalization so N can vary by subgraph.

    Expected history_data shape: [B,T,N,Cin_selected+calendar_channels]
    Output shape: [B,H,N,Cout]
    """

    def __init__(self, model_args: Dict[str, Any]):
        super().__init__()
        self.input_features = list(model_args.get("input_feature_idx", [0, 1, 2]))
        self.output_features = list(model_args.get("output_feature_idx", [0, 1, 2]))
        self.physical_input_dim = len(self.input_features)
        self.output_dim = len(self.output_features)
        self.input_time_steps = int(model_args.get("input_len", 12))
        self.output_time_steps = int(model_args.get("horizon", 48))
        self.num_layers = int(model_args.get("num_layers", 3))

        # Dynamic/state embedding from the full historical window.
        self.input_embedding_dim = int(model_args.get("input_embedding_dim", 256))
        self.temporal_dim = int(model_args.get("temporal_dim", 64))
        self.position_embedding_dim = int(model_args.get("position_embedding_dim", 64))
        self.bathy_embedding_dim = int(model_args.get("bathy_embedding_dim", 32))
        self.manning_embedding_dim = int(model_args.get("manning_embedding_dim", 32))

        # Optional inputs.  Calendar scalars are appended to the selected physical
        # channels by the trainer.  Cyclic encoding contributes sin/cos per field.
        self.use_bathymetry = bool(model_args.get("use_bathymetry", True))
        self.use_manning = bool(model_args.get("use_manning", True))
        self.use_time_of_day = bool(model_args.get("use_time_of_day", False))
        self.use_day_of_week = bool(model_args.get("use_day_of_week", False))
        self.cyclic_calendar = bool(model_args.get("cyclic_calendar", True))
        calendar_width = 2 if self.cyclic_calendar else 1
        self.calendar_dim = calendar_width * (
            int(self.use_time_of_day) + int(self.use_day_of_week)
        )
        self.input_dim = self.physical_input_dim + self.calendar_dim

        self.use_temporal = bool(model_args.get("use_temporal", True))
        self.use_fine_graph = bool(model_args.get("use_fine_graph", True))
        self.use_persistence = bool(model_args.get("use_persistence", True))
        self.learnable_persistence_gate = bool(model_args.get("learnable_persistence_gate", True))

        self.input_ts_emb_layer = nn.Conv1d(
            in_channels=self.input_dim * self.input_time_steps,
            out_channels=self.input_embedding_dim,
            kernel_size=1,
        )

        if self.use_temporal:
            self.temporal_encoder = TemporalLite(
                input_dim=self.input_dim,
                temporal_dim=self.temporal_dim,
                layers=int(model_args.get("temporal_layers", 2)),
                kernel_size=int(model_args.get("temporal_kernel", 3)),
                dropout=float(model_args.get("dropout", 0.1)),
            )

        self.pos_encoder = PositionEmbedding(self.position_embedding_dim)
        if self.use_bathymetry:
            self.bathy_encoder = nn.Sequential(
                nn.Linear(1, self.bathy_embedding_dim), nn.GELU()
            )
        if self.use_manning:
            self.manning_encoder = nn.Sequential(
                nn.Linear(1, self.manning_embedding_dim), nn.GELU()
            )

        self.model_dim = (
            self.input_embedding_dim
            + (self.temporal_dim if self.use_temporal else 0)
            + self.position_embedding_dim
            + (self.bathy_embedding_dim if self.use_bathymetry else 0)
            + (self.manning_embedding_dim if self.use_manning else 0)
        )

        dropout = float(model_args.get("dropout", 0.1))
        self.encoder = nn.Sequential(
            *[ChannelMLPBlock(self.model_dim, expansion=4, dropout=dropout) for _ in range(self.num_layers)]
        )

        if self.use_fine_graph:
            self.fine_graph = FineGraphCorrection(
                model_dim=self.model_dim,
                graph_dim=int(model_args.get("graph_hidden_dim", 64)),
                layers=int(model_args.get("graph_layers", 1)),
                dropout=dropout,
                gate_init=float(model_args.get("graph_gate_init", -2.0)),
            )

        self.output_head = nn.Conv1d(
            self.model_dim,
            self.output_time_steps * self.output_dim,
            kernel_size=1,
        )
        # Start exactly at persistence when persistence is enabled.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        if self.learnable_persistence_gate:
            self.persistence_gate_logit = nn.Parameter(
                torch.full((1, self.output_time_steps, 1, self.output_dim),
                           float(model_args.get("persistence_gate_init", 0.0)))
            )

        # Mapping from output feature IDs to positions inside selected input features.
        local = []
        for f in self.output_features:
            if f not in self.input_features:
                if self.use_persistence:
                    raise ValueError(
                        f"Persistence requires output feature {f} to be present in input_feature_idx={self.input_features}."
                    )
                local.append(-1)
            else:
                local.append(self.input_features.index(f))
        self.output_local_idx = local

    @staticmethod
    def _expand_node_feature(feat: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        if feat.dim() == 1:
            feat = feat[:, None]
        if feat.dim() == 2:
            feat = feat.unsqueeze(0).expand(batch_size, -1, -1)
        elif feat.dim() != 3:
            raise ValueError(f"Expected node feature [N,1] or [B,N,1], got {tuple(feat.shape)}")
        if feat.shape[1] != num_nodes:
            raise ValueError(f"Node count mismatch: feature has N={feat.shape[1]}, history has N={num_nodes}")
        return feat

    def _encode(
        self,
        history_data: torch.Tensor,
        bathymetry: Optional[torch.Tensor],
        manning: Optional[torch.Tensor],
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, n, c = history_data.shape
        if t != self.input_time_steps:
            raise ValueError(f"Expected input_len={self.input_time_steps}, got T={t}")
        if c != self.input_dim:
            raise ValueError(f"Expected Cin={self.input_dim}, got C={c}")

        pieces = []

        # Original STERCHA-style full-window embedding.
        flat = history_data.permute(0, 2, 1, 3).contiguous()            # [B,N,T,C]
        flat = flat.view(b, n, t * c).transpose(1, 2).contiguous()      # [B,T*C,N]
        pieces.append(self.input_ts_emb_layer(flat))                     # [B,Ein,N]

        # Explicit temporal representation.
        if self.use_temporal:
            temporal = self.temporal_encoder(history_data)              # [B,N,Dt]
            pieces.append(temporal.transpose(1, 2).contiguous())         # [B,Dt,N]

        pos_emb = self.pos_encoder(pos, b).transpose(1, 2).contiguous()  # [B,Dp,N]
        pieces.append(pos_emb)

        if self.use_bathymetry:
            if bathymetry is None:
                raise ValueError("use_bathymetry=True requires a bathymetry tensor")
            bath = self._expand_node_feature(bathymetry, b, n).float()
            pieces.append(self.bathy_encoder(bath).transpose(1, 2).contiguous())
        if self.use_manning:
            if manning is None:
                raise ValueError("use_manning=True requires a Manning tensor")
            mann = self._expand_node_feature(manning, b, n).float()
            pieces.append(self.manning_encoder(mann).transpose(1, 2).contiguous())

        hidden = torch.cat(pieces, dim=1)                                 # [B,D,N]
        hidden = self.encoder(hidden)                                     # [B,D,N]

        if self.use_fine_graph:
            hidden = self.fine_graph(hidden, edge_index)                  # [B,D,N]
        return hidden

    def _decode_delta(self, hidden: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        b, _, n = hidden.shape
        steps = self.output_time_steps if steps is None else int(steps)
        if not 1 <= steps <= self.output_time_steps:
            raise ValueError("forecast_steps must be between 1 and horizon")
        channels = steps * self.output_dim
        delta = F.conv1d(hidden, self.output_head.weight[:channels], self.output_head.bias[:channels])
        delta = delta.view(b, steps, self.output_dim, n)
        delta = delta.permute(0, 1, 3, 2).contiguous()                    # [B,H,N,Cout]
        return delta

    def _persistence_base(self, history_data: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor(self.output_local_idx, device=history_data.device, dtype=torch.long)
        last = history_data[:, -1:, :, :].index_select(-1, idx)          # [B,1,N,Cout]
        return last.expand(-1, self.output_time_steps, -1, -1)            # [B,H,N,Cout]

    def forward(
        self,
        history_data: torch.Tensor,
        bathymetry: Optional[torch.Tensor],
        manning: Optional[torch.Tensor],
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mode: str = "direct",
        rollout_steps: Optional[int] = None,
        future_inputs: Optional[torch.Tensor] = None,
        detach_feedback: bool = False,
        forecast_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Forecast with optional static and calendar inputs.

        history_data: selected physical channels followed by enabled TOD/DOW
            channels (cyclic sin/cos pairs by default), [B,T,N,Cin].
        output: direct -> [B,H,N,Cout], rollout -> [B,steps,N,Cout]
        """
        if mode == "rollout":
            steps = self.output_time_steps if rollout_steps is None else int(rollout_steps)
            return self.rollout(
                history_data, bathymetry, manning, pos, edge_index,
                steps=steps,
                future_inputs=future_inputs,
                detach_feedback=detach_feedback,
            )

        if mode not in {"direct", "current"}:
            raise ValueError(f"Unknown forecast mode: {mode}")
        hidden = self._encode(history_data, bathymetry, manning, pos, edge_index)
        delta = self._decode_delta(hidden, forecast_steps)
        if not self.use_persistence:
            return delta

        base = self._persistence_base(history_data)[:, :delta.shape[1]]
        if self.learnable_persistence_gate:
            gate = torch.sigmoid(self.persistence_gate_logit[:, :delta.shape[1]])
            return base + gate * delta
        return base + delta

    def rollout(
        self,
        history_data: torch.Tensor,
        bathymetry: Optional[torch.Tensor],
        manning: Optional[torch.Tensor],
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        steps: int,
        future_inputs: Optional[torch.Tensor] = None,
        detach_feedback: bool = False,
    ) -> torch.Tensor:
        """Autoregressive first-step rollout with optional TOD/DOW channels.

        future_inputs, if supplied, is [B,steps,N,Cin]. Known future
        physical/calendar channels are retained while predicted output channels
        are overwritten by the model's first-step prediction.
        """
        if steps < 1:
            raise ValueError("rollout steps must be positive")
        if any(i < 0 for i in self.output_local_idx):
            raise ValueError("Rollout requires all output features in input_feature_idx")
        if future_inputs is not None and future_inputs.shape[1] < steps:
            raise ValueError("future_inputs is shorter than rollout steps")
        cur_x = history_data
        preds = []

        for s in range(steps):
            pred_1 = self.forward(cur_x, bathymetry, manning, pos, edge_index,
                                  mode="direct", forecast_steps=1)
            preds.append(pred_1)
            feedback = pred_1.detach() if detach_feedback else pred_1

            if future_inputs is not None:
                next_frame = future_inputs[:, s:s + 1].clone()
            else:
                next_frame = cur_x[:, -1:].clone()

            for out_j, local_j in enumerate(self.output_local_idx):
                if local_j >= 0:
                    next_frame[..., local_j] = feedback[..., out_j]
            cur_x = torch.cat([cur_x[:, 1:], next_frame], dim=1)

        return torch.cat(preds, dim=1)


if __name__ == "__main__":
    # Minimal shape test with two different span sizes.
    cfg = {
        "input_feature_idx": [0, 1, 2],
        "output_feature_idx": [0, 1, 2],
        "input_len": 12,
        "horizon": 6,
        "input_embedding_dim": 64,
        "temporal_dim": 32,
        "position_embedding_dim": 16,
        "bathy_embedding_dim": 8,
        "manning_embedding_dim": 8,
        "use_bathymetry": False,
        "use_manning": True,
        "use_time_of_day": True,
        "use_day_of_week": True,
        "cyclic_calendar": True,
        "num_layers": 2,
        "use_temporal": True,
        "use_fine_graph": True,
        "graph_hidden_dim": 16,
        "graph_layers": 1,
        "use_persistence": True,
    }
    model = CoastSpanNet(cfg)
    for n in (20, 37):
        b, t, c = 2, 12, model.input_dim
        x = torch.randn(b, t, n, c)
        bath = None
        mann = torch.rand(n, 1)
        pos = torch.randn(n, 2)
        src = torch.arange(n - 1)
        dst = src + 1
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        y = model(x, bath, mann, pos, edge_index)
        yr = model(x, bath, mann, pos, edge_index, mode="rollout", rollout_steps=4)
        print(f"N={n}: direct={tuple(y.shape)}, rollout={tuple(yr.shape)}")
