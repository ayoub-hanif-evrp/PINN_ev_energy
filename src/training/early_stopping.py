"""Early stopping on an inner-training validation split (never the outer test trip)."""

from __future__ import annotations

from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience: int = 40, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.bad_epochs = 0

    def step(self, epoch: int, value: float, model) -> bool:
        """Return True if training should stop."""
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.best_state = deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
