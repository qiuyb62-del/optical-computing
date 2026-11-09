"""CPU float64 PyTorch baseline; training never receives the test set."""
from dataclasses import dataclass, asdict
import numpy as np
from .ai_data import features, labels, integer
from .ai_inference import ClassifierWeights, digital_forward


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    max_epochs: int = 500
    patience: int = 50
    min_delta: float = 1e-8

    def __post_init__(self):
        integer(self.max_epochs, "max_epochs", 1)
        integer(self.patience, "patience", 1)
        for name in ("learning_rate", "min_delta"):
            v = getattr(self, name)
            if isinstance(v, bool) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def train_classifier(X_train, y_train, X_validation, y_validation, *, seed, config=None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training needs PyTorch: python -m pip install -r requirements-ai.txt") from exc
    config = config or TrainingConfig()
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be TrainingConfig")
    seed = integer(seed, "seed")
    if seed >= 2**63:
        raise ValueError("training seed must be less than 2**63")
    X_train, X_validation = features(X_train), features(X_validation)
    y_train, y_validation = labels(y_train, len(X_train)), labels(y_validation, len(X_validation))
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4, device="cpu", dtype=torch.float64),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 3, device="cpu", dtype=torch.float64),
            )
            tx, vx = torch.from_numpy(X_train), torch.from_numpy(X_validation)
            ty, vy = torch.from_numpy(y_train), torch.from_numpy(y_validation)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, foreach=False)
            criterion = torch.nn.CrossEntropyLoss()

            def losses():
                model.eval()
                with torch.no_grad():
                    return float(criterion(model(tx), ty)), float(criterion(model(vx), vy))

            train_loss, val_loss = losses()
            history = [{"epoch": 0, "train_loss": train_loss, "validation_loss": val_loss}]
            best_loss, best_epoch, waiting = val_loss, 0, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            for epoch in range(1, config.max_epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                criterion(model(tx), ty).backward()
                optimizer.step()
                train_loss, val_loss = losses()
                if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                    raise RuntimeError("nonfinite training loss")
                history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
                if val_loss < best_loss - config.min_delta:
                    best_loss, best_epoch, waiting = val_loss, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    waiting += 1
                if waiting >= config.patience:
                    break
            model.load_state_dict(best_state)
            model.eval()
            weights = ClassifierWeights(
                best_state["0.weight"].numpy(), best_state["0.bias"].numpy(),
                best_state["2.weight"].numpy(), best_state["2.bias"].numpy())
            with torch.no_grad():
                torch_logits = model(vx).numpy()
            export_error = float(np.max(np.abs(digital_forward(X_validation, weights)["logits"] - torch_logits)))
            return weights, {
                "seed": seed, "config": asdict(config), "optimizer": "Adam",
                "dtype": "float64", "device": "cpu", "threads": 1,
                "torch_version": torch.__version__, "best_epoch": best_epoch,
                "epochs_run": history[-1]["epoch"], "best_validation_loss": best_loss,
                "torch_numpy_max_absolute_error": export_error,
                "history": history,
            }
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism, warn_only=old_warn)
