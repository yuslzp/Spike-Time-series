from typing import List, Optional
from pathlib import Path
import copy
import datetime
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from utilsd import use_cuda
from utilsd.config import Registry
from utilsd.earlystop import EarlyStop, EarlyStopStatus

from HybridSNN.common.function import get_loss_fn, get_metric_fn, printt
from HybridSNN.common.utils import AverageMeter, GlobalTracker, to_torch


class RUNNERS(metaclass=Registry, name="runner"):
    """Registry for available training runner implementations."""

    pass


@RUNNERS.register_module()
class BaseRunner(nn.Module):
    """Provide shared optimization, checkpointing, and evaluation utilities."""

    def __init__(
        self,
        loss_fn: str,
        metrics: List[str],
        observe: str,
        lr: float = 1e-3,
        lower_is_better: bool = True,
        max_epoches: int = 50,
        batch_size: int = 512,
        early_stop: int = 10,
        optimizer: str = "Adam",
        weight_decay: float = 1e-5,
        network: Optional[nn.Module] = None,
        model_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        wandb_enabled: bool = False,
        wandb_project: str = "HybridSNN",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        drop_last_train: bool = True,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        scheduler_type: str = "warmup_cosine",
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.05,
        ema_decay: float = 0.0,
        ema_eval: bool = False,
    ) -> None:
        """Initialize the network, optimizer stack, logging, and checkpoint state."""
        super().__init__()
        if not hasattr(self, "hyper_paras"):
            self.hyper_paras = {}
        self._build_network(network, **self.hyper_paras)
        self._init_optimization(
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
            metrics=metrics,
            observe=observe,
            lower_is_better=lower_is_better,
            max_epoches=max_epoches,
            batch_size=batch_size,
            early_stop=early_stop,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last_train=drop_last_train,
            grad_clip=grad_clip,
            amp_enabled=amp_enabled,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
            ema_decay=ema_decay,
            ema_eval=ema_eval,
        )
        self._init_logger(output_dir)
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.lower_is_better = lower_is_better
        if model_path is not None:
            self.load(model_path)
        if torch.cuda.is_available():
            print("Using GPU")
            self.cuda(device=0)
            torch.backends.cudnn.benchmark = True
        self._reset_targets = [module for module in self.network.modules() if hasattr(module, "reset")]
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_cuda() and self.amp_enabled)
        self.ema_state = None
        self._ema_backup_state = None
        self.current_epoch = 0
        self.best_uses_ema = False
        self.wandb_run = None
        if wandb_enabled:
            self.wandb_run = self._init_wandb_run(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
            )

    def _init_wandb_run(self, project: str, entity: Optional[str], run_name: Optional[str]):
        """Start a Weights & Biases run when the dependency is available."""
        try:
            import wandb
        except ImportError:
            print("wandb not installed, skipping wandb init")
            return None

        try:
            return wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=self.hyper_paras,
                reinit=True,
            )
        except BaseException as exc:
            # W&B auth/network failures should not abort model construction.
            print(f"wandb init failed, continuing without wandb: {exc}")
            return None

    def _build_network(self, network, *args, **kwargs) -> None:
        """Assign the runner network and let subclasses finish construction."""
        self.network = network
        raise NotImplementedError()

    def _init_optimization(
        self,
        optimizer: str,
        lr: float,
        weight_decay: float,
        loss_fn: str,
        metrics: List[str],
        observe: str,
        lower_is_better: bool,
        max_epoches: int,
        batch_size: int,
        early_stop: Optional[int] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        drop_last_train: bool = True,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
        scheduler_type: str = "warmup_cosine",
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.05,
        ema_decay: float = 0.0,
        ema_eval: bool = False,
    ) -> None:
        """Configure loss, metrics, optimizer, scheduler, and loader settings."""
        for key, value in locals().items():
            if key not in ["self", "metrics", "observe", "lower_is_better", "loss_fn"]:
                self.hyper_paras[key] = value
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = {metric: get_metric_fn(metric) for metric in metrics}
        self.metrics = metrics
        self.early_stop = EarlyStop(
            patience=early_stop if early_stop is not None else max_epoches,
            mode="min" if lower_is_better else "max",
        )
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.observe = observe
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = max(0, num_workers)
        self.persistent_workers = persistent_workers and self.num_workers > 0
        self.pin_memory = use_cuda() if pin_memory is None else pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last_train = drop_last_train
        self.grad_clip = grad_clip
        self.amp_enabled = amp_enabled
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.ema_decay = ema_decay
        self.ema_eval = ema_eval
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _init_logger(self, log_dir: Path) -> None:
        """Create the TensorBoard writer used by the runner."""
        self.writer = SummaryWriter(log_dir)
        self.writer.flush()

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build a dataloader with the runner's worker and memory settings."""
        kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "drop_last": drop_last,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)

    def forward(self, inputs: torch.Tensor):
        """Define the runner forward pass in subclasses."""
        pass

    def _init_scheduler(self, loader_length):
        """Initialize the configured learning-rate scheduler."""
        if self.scheduler_type == "warmup_cosine":
            total_steps = max(1, self.max_epoches * max(1, loader_length))
            warmup_steps = min(total_steps - 1, max(0, self.warmup_epochs * max(1, loader_length)))
            min_ratio = max(0.0, min(1.0, self.min_lr_ratio))

            def lr_lambda(step: int) -> float:
                """Scale the learning rate during warmup and cosine decay."""
                if warmup_steps > 0 and step < warmup_steps:
                    return max(1e-8, float(step + 1) / float(warmup_steps))
                if total_steps <= warmup_steps + 1:
                    return 1.0
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.scheduler_type == "plateau":
            patience = self.early_stop.patience if hasattr(self.early_stop, "patience") else 30
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self.lower_is_better else "max",
                factor=0.5,
                patience=max(1, patience // 3),
                min_lr=self.lr * 0.01,
            )
        elif self.scheduler_type in {"none", ""}:
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler_type={self.scheduler_type!r}")

    def _init_ema(self) -> None:
        """Initialize exponential moving averages for trainable parameters."""
        if self.ema_decay <= 0:
            self.ema_state = None
            return
        trainable_names = {name for name, _ in self.named_parameters()}
        self.ema_state = {}
        for name, tensor in self.state_dict().items():
            if (
                name in trainable_names
                and torch.is_tensor(tensor)
                and tensor.dtype.is_floating_point
            ):
                self.ema_state[name] = tensor.detach().clone()

    def _update_ema(self) -> None:
        """Update the EMA snapshot after an optimizer step."""
        if self.ema_state is None:
            return
        decay = self.ema_decay
        current_state = self.state_dict()
        for name, ema_tensor in self.ema_state.items():
            current = current_state[name].detach()
            ema_tensor.mul_(decay).add_(current, alpha=1.0 - decay)

    def _apply_ema_weights(self) -> bool:
        """Swap EMA weights into the model for evaluation."""
        if not self.ema_eval or self.ema_state is None:
            return False
        self._ema_backup_state = {}
        current_state = self.state_dict()
        for name, ema_tensor in self.ema_state.items():
            self._ema_backup_state[name] = current_state[name].detach().clone()
            current_state[name].copy_(ema_tensor)
        return True

    def _restore_ema_weights(self) -> None:
        """Restore the original non-EMA weights after evaluation."""
        if self._ema_backup_state is None:
            return
        current_state = self.state_dict()
        for name, backup_tensor in self._ema_backup_state.items():
            current_state[name].copy_(backup_tensor)
        self._ema_backup_state = None

    def _snapshot_best_states(self, use_ema: bool) -> tuple[dict, dict]:
        """Deep-copy the full runner and network weights for the current best model."""
        if use_ema and self.ema_state is not None:
            applied = self._apply_ema_weights()
            try:
                return copy.deepcopy(self.state_dict()), copy.deepcopy(self.network.state_dict())
            finally:
                if applied:
                    self._restore_ema_weights()
        return copy.deepcopy(self.state_dict()), copy.deepcopy(self.network.state_dict())

    def _post_batch(
        self,
        iterations: int,
        epoch,
        train_loss,
        train_global_tracker,
        validset,
        testset,
    ):
        """Hook for subclasses to run logic after each optimization step."""
        pass

    def _compute_extra_loss(self) -> float:
        """Return any subclass-specific loss term added to the main objective."""
        return 0.0

    def _post_epoch(self, epoch: int, validset, train_metric_res: dict) -> None:
        """Hook for subclasses to run logic after each training epoch."""
        pass

    def _load_weight(self, params):
        """Load a full runner state dictionary."""
        self.load_state_dict(params, strict=True)

    def _early_stop(self):
        """Return whether early-stop decisions should terminate training."""
        return True

    def fit(
        self,
        trainset: Dataset,
        validset: Optional[Dataset] = None,
        testset: Optional[Dataset] = None,
    ) -> nn.Module:
        """Train the model, track the best checkpoint, and optionally test it."""
        trainset.load()
        if validset is not None:
            validset.load()

        loader = self._build_loader(trainset, shuffle=True, drop_last=self.drop_last_train)
        self._init_scheduler(len(loader))
        self._init_ema()
        self.best_params = copy.deepcopy(self.state_dict())
        self.best_network_params = copy.deepcopy(self.network.state_dict())
        iterations = 0
        start_epoch, best_res = self._resume()
        best_epoch = best_res.pop("best_epoch", 0)
        best_score = self.early_stop.best
        current_epoch = start_epoch
        self.current_epoch = current_epoch

        try:
            for epoch in range(start_epoch, self.max_epoches):
                current_epoch = epoch
                self.current_epoch = epoch
                self.train()
                train_loss = AverageMeter()
                train_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
                tracked = 0
                optimizer_steps = 0
                skipped_nonfinite_loss = 0
                skipped_nonfinite_grad = 0
                skipped_nonfinite_grad_norm = 0
                start_time = time.time()

                for data, label in loader:
                    if use_cuda():
                        data = to_torch(data, device="cuda:0")
                        label = to_torch(label, device="cuda:0")
                    for module in self._reset_targets:
                        module.reset()
                    with torch.cuda.amp.autocast(enabled=use_cuda() and self.amp_enabled):
                        pred = self(data)
                        if self.out_ranges is not None:
                            pred = pred[:, self.out_ranges]
                            label = label[:, self.out_ranges]
                        loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1)) + self._compute_extra_loss()
                    if not torch.isfinite(loss):
                        print(f"Skipping non-finite loss at epoch={epoch} iter={iterations}")
                        skipped_nonfinite_loss += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grads_finite = True
                    for param in self.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grads_finite = False
                            break
                    if not grads_finite:
                        print(f"Skipping optimizer step due to non-finite gradients at epoch={epoch} iter={iterations}")
                        skipped_nonfinite_grad += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    if self.grad_clip and self.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                        if not torch.isfinite(grad_norm):
                            print(f"Skipping optimizer step due to non-finite grad norm at epoch={epoch} iter={iterations}")
                            skipped_nonfinite_grad_norm += 1
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._update_ema()
                    if self.scheduler is not None and self.scheduler_type == "warmup_cosine":
                        self.scheduler.step()

                    optimizer_steps += 1
                    train_loss.update(loss.item(), label.numel())
                    train_global_tracker.update(label, pred)
                    tracked += label.shape[0]
                    iterations += 1
                    self._post_batch(iterations, epoch, train_loss, train_global_tracker, validset, testset)

                if optimizer_steps == 0:
                    raise RuntimeError(
                        "No valid optimizer steps completed for "
                        f"epoch={epoch}. skipped_nonfinite_loss={skipped_nonfinite_loss}, "
                        f"skipped_nonfinite_grad={skipped_nonfinite_grad}, "
                        f"skipped_nonfinite_grad_norm={skipped_nonfinite_grad_norm}. "
                        "Treat this run as invalid rather than writing misleading zero train metrics."
                    )

                train_time = time.time() - start_time
                start_time = time.time()
                train_global_tracker.concat()
                metric_res = train_global_tracker.performance()
                metric_time = time.time() - start_time
                metric_res["loss"] = train_loss.performance()

                printt(f"{epoch}\t'train'\tTime:{train_time:.2f}\tMetricT: {metric_time:.2f}")
                for metric, value in metric_res.items():
                    printt(f"{metric}: {value:.4f}")
                print(f"{datetime.datetime.today()}")
                for key, value in metric_res.items():
                    self.writer.add_scalar(f"{key}/train", value, epoch)
                self.writer.flush()
                if self.wandb_run is not None:
                    try:
                        self.wandb_run.log({f"train/{k}": v for k, v in metric_res.items()}, step=epoch)
                    except Exception as exc:
                        print(f"wandb train logging failed, disabling wandb: {exc}")
                        self.wandb_run = None

                self._post_epoch(epoch, validset, metric_res)

                if validset is not None:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    using_ema = self._apply_ema_weights()
                    with torch.no_grad():
                        eval_res = self.evaluate(validset, epoch)
                    if using_ema:
                        self._restore_ema_weights()
                    value = eval_res[self.observe]
                    if self.scheduler is not None and self.scheduler_type == "plateau":
                        self.scheduler.step(value)
                    es = self.early_stop.step(value)
                    if es == EarlyStopStatus.BEST:
                        best_score = value
                        best_epoch = epoch
                        self.best_params, self.best_network_params = self._snapshot_best_states(using_ema)
                        self.best_uses_ema = bool(using_ema)
                        best_res = {"train": metric_res, "valid": eval_res}
                        torch.save(self.best_params, self.checkpoint_dir / "model_best.pkl")
                        torch.save(self.best_network_params, self.checkpoint_dir / "network_best.pkl")
                    elif es == EarlyStopStatus.STOP and self._early_stop():
                        break
                else:
                    value = metric_res[self.observe]
                    es = self.early_stop.step(value)
                    if es == EarlyStopStatus.BEST:
                        best_score = value
                        best_epoch = epoch
                        self.best_params = copy.deepcopy(self.state_dict())
                        self.best_network_params = copy.deepcopy(self.network.state_dict())
                        self.best_uses_ema = False
                        best_res = {"train": metric_res}
                        torch.save(self.best_params, self.checkpoint_dir / "model_best.pkl")
                        torch.save(self.best_network_params, self.checkpoint_dir / "network_best.pkl")
                    elif es == EarlyStopStatus.STOP and self._early_stop():
                        break
                self._checkpoint(epoch, {**best_res, "best_epoch": best_epoch})
        except Exception:
            if self.checkpoint_dir is not None:
                self._checkpoint(current_epoch, {**best_res, "best_epoch": best_epoch})
            raise

        trainset.freeup()
        if validset is not None:
            validset.freeup()

        self._load_weight(self.best_params)
        if testset is not None:
            testset.load()
            print("Begin evaluate on testset ...")
            using_ema = False
            if self.ema_eval and not self.best_uses_ema:
                using_ema = self._apply_ema_weights()
            with torch.no_grad():
                test_res = self.evaluate(testset)
            if using_ema:
                self._restore_ema_weights()
            for key, value in test_res.items():
                self.writer.add_scalar(f"{key}/test", value, epoch)
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log({f"test/{k}": v for k, v in test_res.items()})
                    self.wandb_run.finish()
                except Exception as exc:
                    print(f"wandb test logging failed, disabling wandb: {exc}")
                    self.wandb_run = None
            best_score = test_res[self.observe]
            best_res["test"] = test_res
            testset.freeup()

        torch.save(self.best_params, self.checkpoint_dir / "model_best.pkl")
        torch.save(self.best_network_params, self.checkpoint_dir / "network_best.pkl")
        with open(self.checkpoint_dir / "res.json", "w") as file:
            json.dump(best_res, file, indent=4, sort_keys=True)
        print(best_res)
        keys = list(self.hyper_paras.keys())
        for key in keys:
            if type(self.hyper_paras[key]) not in [int, float, str, bool, torch.Tensor]:
                self.hyper_paras.pop(key)
        self.writer.add_hparams(self.hyper_paras, {"result": best_score, "best_epoch": best_epoch})
        return self

    def _checkpoint(self, cur_epoch, best_res, checkpoint_dir=None):
        """Persist resume state, best weights, and optimizer state to disk."""
        target_dir = self.checkpoint_dir if checkpoint_dir is None else checkpoint_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "earlystop": self.early_stop.state_dict(),
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "epoch": cur_epoch,
                "best_res": best_res,
                "best_params": self.best_params,
                "best_network_params": self.best_network_params,
                "best_uses_ema": self.best_uses_ema,
                "ema_state": self.ema_state,
            },
            target_dir / "resume.pth",
        )
        print(f"Checkpoint saved to {target_dir / 'resume.pth'}", __name__)

    def _resume(self):
        """Resume training state from `resume.pth` when it exists."""
        resume_path = self.checkpoint_dir / "resume.pth"
        if resume_path.exists():
            print(f"Resume from {resume_path}", __name__)
            checkpoint = torch.load(resume_path, weights_only=False)
            self.early_stop.load_state_dict(checkpoint["earlystop"])
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            if self.scheduler is not None and checkpoint.get("scheduler") is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_params = checkpoint["best_params"]
            self.best_network_params = checkpoint["best_network_params"]
            self.best_uses_ema = checkpoint.get("best_uses_ema", False)
            self.ema_state = checkpoint.get("ema_state")
            return checkpoint["epoch"] + 1, checkpoint["best_res"]
        print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
        return 0, {}

    def evaluate(self, validset: Dataset, epoch: Optional[int] = None) -> dict:
        """Evaluate the model on one dataset split and compute tracked metrics."""
        validset.load()
        loader = self._build_loader(validset, shuffle=False, drop_last=False)
        self.eval()
        eval_loss = AverageMeter()
        eval_global_tracker = GlobalTracker(self.metrics, self.metric_fn)
        start_time = time.time()
        with torch.no_grad():
            for data, label in loader:
                if use_cuda():
                    data = to_torch(data, device="cuda:0")
                    label = to_torch(label, device="cuda:0")
                for module in self._reset_targets:
                    module.reset()
                with torch.cuda.amp.autocast(enabled=use_cuda() and self.amp_enabled):
                    pred = self(data)
                    if self.out_ranges is not None:
                        pred = pred[:, self.out_ranges]
                        label = label[:, self.out_ranges]
                    loss = self.loss_fn(label.squeeze(-1), pred.squeeze(-1))
                eval_loss.update(loss.item(), label.numel())
                eval_global_tracker.update(label, pred)

        eval_time = time.time() - start_time
        start_time = time.time()
        eval_global_tracker.concat()
        metric_res = eval_global_tracker.performance()
        metric_time = time.time() - start_time
        metric_res["loss"] = eval_loss.performance()

        if epoch is not None:
            printt(f"{epoch}\t'valid'\tTime:{eval_time:.2f}\tMetricT: {metric_time:.2f}")
            for metric, value in metric_res.items():
                printt(f"{metric}: {value:.4f}")
            print(f"{datetime.datetime.today()}")
            for key, value in metric_res.items():
                self.writer.add_scalar(f"{key}/valid", value, epoch)
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log({f"valid/{k}": v for k, v in metric_res.items()}, step=epoch)
                except Exception as exc:
                    print(f"wandb valid logging failed, disabling wandb: {exc}")
                    self.wandb_run = None

        return metric_res

    def load(self, model_path: str, strict=True):
        """Load a serialized runner state dictionary from disk."""
        self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=strict)

    def predict(self, dataset: Dataset, name: str):
        """Run inference on a dataset split and save predictions as a pickle file."""
        self.eval()
        dataset.load()
        data_length = len(dataset.get_index())
        loader = self._build_loader(dataset, shuffle=False, drop_last=False)
        prediction: Optional[np.ndarray] = None
        ptr = 0
        with torch.no_grad():
            for data, _ in loader:
                if use_cuda():
                    data = to_torch(data, device="cuda:0")
                for module in self._reset_targets:
                    module.reset()
                pred = self(data)
                if self.out_ranges is not None:
                    pred = pred[:, self.out_ranges]
                pred = pred.squeeze(-1).cpu().detach().numpy()
                if prediction is None:
                    prediction = np.empty((data_length, pred.reshape(pred.shape[0], -1).shape[-1]), dtype=np.float32)
                n_samples = pred.shape[0]
                prediction[ptr : ptr + n_samples] = pred.reshape(n_samples, -1)
                ptr += n_samples
        prediction = pd.DataFrame(data=prediction, index=dataset.get_index())
        prediction.to_pickle(self.checkpoint_dir / f"{name}_pre.pkl")
        return prediction
