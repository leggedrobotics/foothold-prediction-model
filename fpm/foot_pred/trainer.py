import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .labels import foothold_labels
from .model import FootPredictor, NormalizedFootPredictor


def fit_foot_predictor(
    state_hist: torch.Tensor,
    contact_hist: torch.Tensor,
    pose_hist: torch.Tensor,
    save_path: str | None = None,
    device: str = "cuda",
    hidden_dims: tuple[int, ...] = (128, 128),
    lr: float = 5e-3,
    batch_size: int = 1024,
    num_epochs: int = 25,
    loss_delta: float = 0.01,
    train_fraction: float = 0.8,
    reset_xy_jump: float = 0.5,
    num_feet: int = 2,
) -> tuple[nn.Module, float]:
    with torch.inference_mode(False), torch.enable_grad():
        state_hist, contact_hist, pose_hist = state_hist.clone(), contact_hist.clone(), pose_hist.clone()

        x = state_hist[:, :-1].reshape(-1, state_hist.shape[-1])
        y = foothold_labels(contact_hist, pose_hist, num_feet, reset_xy_jump).reshape(x.shape[0], -1)

        dataset = TensorDataset(x, y)
        num_train = int(train_fraction * len(dataset))
        train_set, val_set = random_split(dataset, [num_train, len(dataset) - num_train])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        x_train = x[torch.as_tensor(train_set.indices)]
        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0).clamp_min(1e-6)
        core = FootPredictor(x.shape[-1], y.shape[-1], hidden_dims)
        model = NormalizedFootPredictor(core, mean, std).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss(delta=loss_delta)

        val_loss = float("nan")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    val_loss += criterion(model(batch_x.to(device)), batch_y.to(device)).item()
            train_loss /= max(len(train_loader), 1)
            val_loss /= max(len(val_loader), 1)
            print(f"[FPM] epoch {epoch + 1}/{num_epochs}  train {train_loss:.5f}  val {val_loss:.5f}")

        model.eval()
        if save_path is not None:
            torch.jit.save(torch.jit.script(model.cpu()), save_path)
            model.to(device)
            print(f"[FPM] saved {save_path}")
    return model, val_loss
