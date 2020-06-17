from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
