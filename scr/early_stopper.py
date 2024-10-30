import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 1, delta: int = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.max_score = -np.inf

    def __call__(self, score) -> bool:
        if score > self.max_score:
            self.max_score = score
            self.counter = 0
        elif score <= (self.max_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False