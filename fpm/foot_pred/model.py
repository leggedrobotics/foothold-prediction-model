import torch
import torch.nn as nn


class FootPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dims: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers, last = [], input_size
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            last = h
        layers.append(nn.Linear(last, output_size))
        self.net = nn.Sequential(*layers)
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class NormalizedFootPredictor(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("in_mean", mean.reshape(-1))
        self.register_buffer("in_std", std.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.in_mean) / self.in_std)
