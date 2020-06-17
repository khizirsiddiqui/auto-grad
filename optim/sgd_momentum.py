import numpy as np
from .base import Optimizer

class SGDMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.9):
        super().__init__(params, lr)
        self.gamma = gamma
        self.velocity = {param.name: np.zeros_like(param) for param in params}

    def step(self):
        for param in self.params:
            self.velocity[param.name] = self.gamma * self.velocity[param.name] - self.lr * param.grad
            param += self.velocity[param.name]
