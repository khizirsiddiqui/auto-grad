from .base import Optimizer
import numpy as np

class SGDNesterov(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.9):
        super().__init__(params, lr)
        self.gamma = gamma
        self.velocity = {param.name: np.zeros_like(param) for param in params}

    def step(self):
        for param in self.params:
            prev_v = np.copy(self.velocity[param.name])
            self.velocity[param.name] = self.gamma * self.velocity[param.name] - self.lr * param.grad
            param += -self.gamma * prev_v + (1 + self.gamma) * self.velocity[param.name]
