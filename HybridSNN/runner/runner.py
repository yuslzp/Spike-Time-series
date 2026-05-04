from typing import List, Tuple, Optional, Union
import numpy as np

import torch
from torch import nn

from HybridSNN.runner.base import RUNNERS, BaseRunner


@RUNNERS.register_module("ts", inherit=True)
class TS(BaseRunner):
    """Generic runner for sequence classification and forecasting tasks."""

    def __init__(
        self,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
        out_size: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        aggregate: bool = True,
        mlp_head: bool = False,
        readout_mode: str = "pooled",
        readout_hidden: Optional[int] = None,
        readout_dropout: float = 0.1,
        decoder_num_heads: int = 4,
        forecast_residual: bool = False,
        forecast_residual_meta: Optional[dict] = None,
        forecast_residual_mode: str = "last",
        forecast_trend_window: int = 8,
        forecast_trend_blend: float = 0.5,
        **kwargs,
    ):
        """
        The model for general time-series prediction.

        Args:
            task: the prediction task, classification or regression.
            optimizer: which optimizer to use.
            lr: learning rate.
            weight_decay: L2 normlize weight
            loss_fn: loss function.
            metrics: metrics to evaluate model.
            observe: metric for model selection (earlystop).
            lower_is_better: whether a lower observed metric means better result.
            max_epoches: maximum epoch to learn.
            batch_size: batch size.
            early_stop: earlystop rounds.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            model_path: the path to existing model parameters for continued training or finetuning
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """
        self.hyper_paras = {
            "task": task,
            "out_ranges": out_ranges,
            "out_size": out_size,
            "forecast_horizon": forecast_horizon,
            "aggregate": aggregate,
            "mlp_head": mlp_head,
            "readout_mode": readout_mode,
            "readout_hidden": readout_hidden,
            "readout_dropout": readout_dropout,
            "decoder_num_heads": decoder_num_heads,
            "forecast_residual": forecast_residual,
            "forecast_residual_mode": forecast_residual_mode,
            "forecast_trend_window": forecast_trend_window,
            "forecast_trend_blend": forecast_trend_blend,
        }
        self._forecast_residual_meta = forecast_residual_meta
        super().__init__(**kwargs)

    def _build_network(
        self,
        network,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int, int], Tuple[int, int]]]] = None,
        out_size: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        aggregate: bool = True,
        mlp_head: bool = False,
        readout_mode: str = "pooled",
        readout_hidden: Optional[int] = None,
        readout_dropout: float = 0.1,
        decoder_num_heads: int = 4,
        forecast_residual: bool = False,
        forecast_residual_mode: str = "last",
        forecast_trend_window: int = 8,
        forecast_trend_blend: float = 0.5,
    ) -> None:
        """Initilize the network parameters

        Args:
            task: the prediction task, classification or regression.
            out_ranges: a list of final ranges to take as final output. Should have form [(start, end), (start, end, step), ...]
            out_size: the output size for multi-class classification or multi-variant regression task.
            aggregate: whether to aggregate across whole sequence.
        """

        self.network = network
        self.aggregate = aggregate
        self.readout_mode = readout_mode
        self.readout_dropout = readout_dropout
        self.forecast_horizon = forecast_horizon
        self.target_num_variables = None
        self.forecast_residual = False
        self.forecast_residual_mode = "last"
        self.forecast_trend_window = 2
        self.forecast_trend_blend = 0.0
        self._forecast_raw_label = False
        self._forecast_signal_indices: Optional[torch.Tensor] = None
        self._forecast_target_center: Optional[torch.Tensor] = None
        self._forecast_target_scale: Optional[torch.Tensor] = None

        # Output
        if task == "classification":
            self.act_out = nn.Sigmoid()
            out_size = 1
        elif task == "multiclassification":
            self.act_out = nn.LogSoftmax(-1)
        elif task == "regression":
            self.act_out = nn.Identity()
        else:
            raise ValueError(
                ("Task must be 'classification', 'multiclassification', 'regression'")
            )

        if out_ranges is not None:
            self.out_ranges = []
            for ran in out_ranges:
                if len(ran) == 2:
                    self.out_ranges.append(np.arange(ran[0], ran[1]))
                elif len(ran) == 3:
                    self.out_ranges.append(np.arange(ran[0], ran[1], ran[2]))
                else:
                    raise ValueError(f"Unknown range {ran}")
            self.out_ranges = np.concatenate(self.out_ranges)
        else:
            self.out_ranges = None

        d = network.output_size
        default_hidden = max(d * 4, 256)
        self.readout_hidden = readout_hidden if readout_hidden is not None else default_hidden
        self.seq_len = getattr(network, "max_length", None)
        self.sequence_pool = None
        self.horizon_decoder = None

        if out_size is not None:
            print('out_size', out_size)
            if self.readout_mode == "sequence_mlp":
                if self.seq_len is None:
                    raise ValueError("sequence_mlp readout requires network.max_length.")
                self.fc_out = nn.Sequential(
                    nn.LayerNorm(self.seq_len * d),
                    nn.Linear(self.seq_len * d, self.readout_hidden),
                    nn.GELU(),
                    nn.Dropout(readout_dropout),
                    nn.Linear(self.readout_hidden, out_size),
                )
            elif self.readout_mode == "horizon_decoder":
                if task != "regression":
                    raise ValueError("horizon_decoder is only supported for regression.")
                if forecast_horizon is None or forecast_horizon <= 0:
                    raise ValueError("horizon_decoder requires forecast_horizon > 0.")
                if out_size % forecast_horizon != 0:
                    raise ValueError(
                        f"out_size={out_size} is not divisible by forecast_horizon={forecast_horizon}."
                    )
                self.target_num_variables = out_size // forecast_horizon
                self.horizon_decoder = HorizonDecoder(
                    dim=d,
                    horizon=forecast_horizon,
                    target_num_variables=self.target_num_variables,
                    hidden_dim=self.readout_hidden,
                    dropout=readout_dropout,
                    num_heads=decoder_num_heads,
                )
                self.fc_out = nn.Identity()
                self.forecast_residual = forecast_residual
                self.forecast_residual_mode = str(forecast_residual_mode).lower()
                if self.forecast_residual_mode not in {"last", "trend"}:
                    raise ValueError(
                        "forecast_residual_mode must be 'last' or 'trend', "
                        f"got {forecast_residual_mode!r}."
                    )
                self.forecast_trend_window = max(2, int(forecast_trend_window))
                self.forecast_trend_blend = max(0.0, float(forecast_trend_blend))
                if self.forecast_residual:
                    self._init_forecast_residual_meta()
            elif self.readout_mode == "attention_pool":
                self.sequence_pool = AttentionPooling(d, dropout=readout_dropout)
                if mlp_head and task == "regression":
                    self.fc_out = nn.Sequential(
                        nn.LayerNorm(d),
                        nn.Linear(d, self.readout_hidden),
                        nn.GELU(),
                        nn.Dropout(readout_dropout),
                        nn.Linear(self.readout_hidden, out_size),
                    )
                else:
                    self.fc_out = nn.Linear(d, out_size)
            elif mlp_head and task == "regression":
                # Legacy pooled regression head.
                self.fc_out = nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.GELU(),
                    nn.Linear(d * 2, out_size),
                )
            else:
                self.fc_out = nn.Linear(d, out_size)
        else:
            self.fc_out = nn.Identity()

    def _init_forecast_residual_meta(self) -> None:
        """Prepare target-space metadata for residual forecasting."""
        meta = self._forecast_residual_meta or {}
        if self.target_num_variables is None or self.target_num_variables <= 0:
            raise ValueError("forecast_residual requires a valid target_num_variables.")

        signal_indices = meta.get("signal_indices")
        if signal_indices is None:
            signal_indices = np.arange(self.target_num_variables, dtype=np.int64)
        signal_indices = torch.as_tensor(signal_indices, dtype=torch.long)
        if signal_indices.numel() != self.target_num_variables:
            raise ValueError(
                "forecast_residual signal_indices must match target_num_variables "
                f"({signal_indices.numel()} != {self.target_num_variables})."
            )

        target_center = meta.get("target_center")
        if target_center is None:
            target_center = np.zeros(self.target_num_variables, dtype=np.float32)
        target_scale = meta.get("target_scale")
        if target_scale is None:
            target_scale = np.ones(self.target_num_variables, dtype=np.float32)

        self._forecast_signal_indices = signal_indices
        self._forecast_target_center = torch.as_tensor(target_center, dtype=torch.float32)
        self._forecast_target_scale = torch.as_tensor(target_scale, dtype=torch.float32)
        self._forecast_raw_label = bool(meta.get("raw_label", False))

    def _extract_forecast_residual_base(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the last observed value into the forecast target space."""
        history = self._extract_forecast_signal_history(inputs)
        return history[:, -1, :]

    def _extract_forecast_signal_history(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project the observed signal history into the forecast target space."""
        if self.target_num_variables is None:
            raise RuntimeError("forecast_residual requires target_num_variables to be set.")
        if inputs.ndim != 3:
            raise ValueError(f"forecast_residual expects rank-3 inputs, got shape={tuple(inputs.shape)}")

        signal_indices = self._forecast_signal_indices
        if signal_indices is None:
            signal_indices = torch.arange(self.target_num_variables, device=inputs.device)
        else:
            signal_indices = signal_indices.to(inputs.device)

        if signal_indices.numel() == 0:
            raise RuntimeError("forecast_residual signal_indices is empty.")
        if int(signal_indices.max().item()) >= inputs.shape[-1]:
            raise ValueError(
                "forecast_residual signal_indices exceed the available input channels "
                f"({int(signal_indices.max().item())} >= {inputs.shape[-1]})."
            )

        history = inputs.index_select(dim=-1, index=signal_indices)
        if self._forecast_raw_label:
            center = self._forecast_target_center.to(inputs.device)
            scale = self._forecast_target_scale.to(inputs.device)
            history = history * scale.view(1, 1, -1) + center.view(1, 1, -1)
        return history

    def _build_forecast_residual_baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Build a horizon-wise baseline that the decoder predicts residuals against."""
        if self.forecast_horizon is None or self.forecast_horizon <= 0:
            raise RuntimeError("forecast_residual requires a positive forecast_horizon.")

        base = self._extract_forecast_residual_base(inputs)
        baseline = base.unsqueeze(1).expand(-1, self.forecast_horizon, -1)
        if self.forecast_residual_mode != "trend" or self.forecast_trend_blend <= 0.0:
            return baseline

        history = self._extract_forecast_signal_history(inputs)
        window = min(self.forecast_trend_window, history.shape[1])
        if window < 2:
            return baseline

        recent = history[:, -window:, :]
        slope = recent[:, 1:, :] - recent[:, :-1, :]
        slope = slope.mean(dim=1) * self.forecast_trend_blend
        if not torch.isfinite(slope).all():
            slope = torch.nan_to_num(slope)

        steps = torch.arange(
            1, self.forecast_horizon + 1, device=inputs.device, dtype=history.dtype
        ).view(1, self.forecast_horizon, 1)
        return baseline + steps * slope.unsqueeze(1)

    def forward(self, inputs):
        """Run the backbone and configured readout head to produce predictions."""
        seq_out, emb_outs = self.network(inputs)

        if self.readout_mode == "sequence_mlp":
            out = seq_out.reshape(seq_out.shape[0], -1)
            preds = self.fc_out(out)
        elif self.readout_mode == "horizon_decoder":
            if self.horizon_decoder is None:
                raise RuntimeError("horizon_decoder readout requested but decoder is not initialized.")
            decoded = self.horizon_decoder(seq_out)
            if self.forecast_residual:
                decoded = decoded + self._build_forecast_residual_baseline(inputs)
            preds = decoded.reshape(seq_out.shape[0], -1)
        elif self.readout_mode == "attention_pool":
            if self.sequence_pool is None:
                raise RuntimeError("attention_pool readout requested but pool module is not initialized.")
            out = self.sequence_pool(seq_out)
            preds = self.fc_out(out)
        elif self.aggregate:
            out = emb_outs
            preds = self.fc_out(out)
        else:
            out = seq_out
            preds = self.fc_out(out)
        preds = self.act_out(preds.squeeze(-1))

        return preds


class AttentionPooling(nn.Module):
    """Pool sequence features with learned attention weights."""

    def __init__(self, dim: int, dropout: float = 0.1):
        """Build the scoring network used for attention pooling."""
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, seq_out: torch.Tensor) -> torch.Tensor:
        """Compute a weighted average over the sequence dimension."""
        weights = torch.softmax(self.score(seq_out).squeeze(-1), dim=1)
        return (seq_out * weights.unsqueeze(-1)).sum(dim=1)


class HorizonDecoder(nn.Module):
    """Decode sequence features into multi-step forecasting outputs."""

    def __init__(
        self,
        dim: int,
        horizon: int,
        target_num_variables: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        """Initialize learned horizon queries and the cross-attention decoder."""
        super().__init__()
        self.horizon = horizon
        self.target_num_variables = target_num_variables
        self.horizon_queries = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_num_variables),
        )

    def forward(self, seq_out: torch.Tensor) -> torch.Tensor:
        """Decode horizon-specific predictions from sequence embeddings."""
        batch_size = seq_out.shape[0]
        queries = self.horizon_queries.unsqueeze(0).expand(batch_size, -1, -1)
        global_context = seq_out.mean(dim=1, keepdim=True)
        queries = self.query_norm(queries + global_context)
        memory = self.context_norm(seq_out)
        attended, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        decoded = queries + attended
        decoded = decoded + self.ffn(decoded)
        return self.out_proj(decoded)


"""
HybridTS runner: extends TS with spike-rate regularization loss and visualization hooks.

Register this in YAML configs as:
    runner:
      type: hybrid_ts
      viz_every: 10       # run visualization every N epochs (0 = disabled)
      spike_lambda: 0.01  # weight for spike-rate reg loss (0 = disabled)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from utilsd import use_cuda

from HybridSNN.common.utils import to_torch
from HybridSNN.visualization.viz_runner import run_visualization


@RUNNERS.register_module("hybrid_ts", inherit=True)
class HybridTS(TS):
    """TS runner extended with:
    - Spike-rate regularization loss via network.get_spike_loss()
    - Periodic visualization via run_visualization()
    - wandb integration (inherited from BaseRunner)
    """

    def __init__(
        self,
        viz_every: int = 10,
        spike_lambda: float = 0.0,
        **kwargs,
    ):
        """Configure periodic visualization and spike regularization settings."""
        self.viz_every = viz_every
        self.spike_lambda = spike_lambda
        super().__init__(**kwargs)
        if self.spike_lambda > 0 and hasattr(self.network, "spike_lambda"):
            self.network.spike_lambda = self.spike_lambda

        # Store a small validation batch for visualization
        self._viz_batch: Optional[torch.Tensor] = None

    def _compute_extra_loss(self) -> float:
        """Add spike-rate regularization loss from HybridSNN."""
        if self.spike_lambda > 0 and hasattr(self.network, "get_spike_loss"):
            return self.network.get_spike_loss(epoch=getattr(self, "current_epoch", 0))
        return 0.0

    def _post_epoch(self, epoch: int, validset, train_metric_res: dict) -> None:
        """Run visualization every viz_every epochs."""
        if self.viz_every <= 0 or (epoch % self.viz_every) != 0:
            return
        if not hasattr(self.network, "record_mode"):
            return

        # Grab a small batch from validset for visualization
        batch = self._get_viz_batch(validset)
        if batch is None:
            return

        output_dir = str(self.output_dir) if getattr(self, "output_dir", None) else "/tmp/hybrid_snn_viz"
        try:
            run_visualization(
                network=self.network,
                data_batch=batch,
                output_dir=output_dir,
                epoch=epoch,
                wandb_run=self.wandb_run,
            )
        except Exception as e:
            print(f"Visualization failed at epoch {epoch}: {e}")

    def _get_viz_batch(self, validset) -> Optional[torch.Tensor]:
        """Get a small batch from validset for visualization."""
        if self._viz_batch is not None:
            return self._viz_batch
        if validset is None:
            return None
        try:
            validset.load()
            loader = DataLoader(validset, batch_size=4, shuffle=False, num_workers=0)
            data, _ = next(iter(loader))
            if torch.cuda.is_available():
                device = next(self.parameters()).device
                data = to_torch(data, device=str(device))
            self._viz_batch = data
            return self._viz_batch
        except Exception as e:
            print(f"Could not get viz batch: {e}")
            return None
