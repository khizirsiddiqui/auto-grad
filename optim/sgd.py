class SGD:
    def __init__(self, params, lr=1e-3):
        self.lr = lr
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
