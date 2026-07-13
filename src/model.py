import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────

class DeepSetPredictor(nn.Module):
    def __init__(
        self,
        point_dim: int,
        phi_hidden: int,
        phi_out: int,
        rho_hidden: int,
        dropout: float,
    ):
        super().__init__()

        # phi: shared per-point MLP, applied independently to each of the n points
        self.phi = nn.Sequential(
            nn.Linear(point_dim, phi_hidden),
            nn.BatchNorm1d(phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(phi_hidden, phi_out),
            nn.BatchNorm1d(phi_out),
            nn.ReLU(),
        )

        # rho: post-pooling MLP, operates on the aggregated set representation
        self.rho = nn.Sequential(
            nn.Linear(phi_out * 2, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(rho_hidden, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(rho_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_points, point_dim)
        b, n, d = x.shape
        # apply phi to every point independently (shared weights)
        h = self.phi(x.reshape(b * n, d)).reshape(b, n, -1)  # (batch, n, phi_out)
        # symmetric pooling over the point dimension
        pooled = torch.cat([h.sum(dim=1), h.max(dim=1).values], dim=-1)
        return self.rho(pooled)  # (batch, 2): depth score, size score


def predict_values(
    model: nn.Module,
    x: np.ndarray,
    predict_batch_size: int,
) -> np.ndarray:
    model.to(DEVICE)
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), predict_batch_size):
            xb = torch.as_tensor(
                x[start : start + predict_batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            predictions.append(model(xb).cpu().numpy())

    return np.concatenate(predictions).astype(np.float32)

