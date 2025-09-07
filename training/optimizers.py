import torch
import torch.optim as optim

class Adam:
    """Adam optimizer implementation using PyTorch's Adam."""
    def __init__(self, model_parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.optimizer = optim.Adam(model_parameters, lr=learning_rate, betas=(beta1, beta2), eps=epsilon)

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups
