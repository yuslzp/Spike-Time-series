from typing import List, Tuple, Optional, Union
import numpy as np

from torch import nn

from HybridSNN.runner.base import RUNNERS, BaseRunner


@RUNNERS.register_module("ts", inherit=True)
class TS(BaseRunner):
    def __init__(
        self,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int], Tuple[int, int, int]]]] = None,
        out_size: Optional[int] = None,
        aggregate: bool = True,
        mlp_head: bool = False,
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
            "aggregate": aggregate,
            "mlp_head": mlp_head,
        }
        super().__init__(**kwargs)

    def _build_network(
        self,
        network,
        task: str,
        out_ranges: Optional[List[Union[Tuple[int, int, int], Tuple[int, int]]]] = None,
        out_size: Optional[int] = None,
        aggregate: bool = True,
        mlp_head: bool = False,
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
        if out_size is not None:
            print('out_size', out_size)
            d = network.output_size
            if mlp_head and task == "regression":
                # 2-layer MLP head: d -> 2d -> out_size
                # Addresses the severe bottleneck when out_size >> d (e.g. 128->8880)
                self.fc_out = nn.Sequential(
                    nn.Linear(d, d * 2),
                    nn.GELU(),
                    nn.Linear(d * 2, out_size),
                )
            else:
                self.fc_out = nn.Linear(d, out_size)
        else:
            self.fc_out = nn.Identity()

    def forward(self, inputs):
        seq_out, emb_outs = self.network(inputs)

        if self.aggregate:
            out = emb_outs
        else:
            out = seq_out
        preds = self.fc_out(out)
        preds = self.act_out(preds.squeeze(-1))

        return preds


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
        self.viz_every = viz_every
        self.spike_lambda = spike_lambda
        super().__init__(**kwargs)

        # Store a small validation batch for visualization
        self._viz_batch: Optional[torch.Tensor] = None

    def _compute_extra_loss(self) -> float:
        """Add spike-rate regularization loss from HybridSNN."""
        if self.spike_lambda > 0 and hasattr(self.network, "get_spike_loss"):
            return self.network.get_spike_loss()
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

        output_dir = str(self.checkpoint_dir.parent) if self.checkpoint_dir else "/tmp/hybrid_snn_viz"
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
