import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────

def unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Expands packed rows (rows, row_bytes) into ±1 float32 (rows, count).

    Bit order matches np.unpackbits(bitorder="little"): bit b of the padded
    little-endian row stands for the float 2*b - 1.
    """
    assert packed.dtype == torch.uint8, packed.dtype
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(len(packed), -1)[:, :count]
    return bits.to(torch.float32) * 2.0 - 1.0


class DeepSetPredictor(nn.Module):
    def __init__(
        self,
        batches: int,
        points_in_batch: int,
        point_dim: int,
        phi_hidden: int,
        phi_out: int,
        rho_hidden: int,
        dropout: float,
    ):
        super().__init__()

        self.batches = batches
        self.points_in_batch = points_in_batch
        self.point_dim = point_dim

        # phi: one dedicated per-batch MLP for each batch of points (no weight sharing)
        self.phi = nn.ModuleList(
            nn.Sequential(
                nn.Linear(points_in_batch * point_dim, phi_hidden),
                nn.BatchNorm1d(phi_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(phi_hidden, phi_out),
                nn.BatchNorm1d(phi_out),
                nn.ReLU(),
            )
            for _ in range(batches)
        )

        # rho: MLP over the concatenated per-batch phi outputs
        self.rho = nn.Sequential(
            nn.Linear(batches * phi_out, rho_hidden),
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
        # x: float32 (batch, batches * points_in_batch, point_dim), or the same
        # points as packed uint8 rows (batch, row_bytes) unpacked here on-device
        if x.dtype == torch.uint8:
            num_points = self.batches * self.points_in_batch
            x = unpack_bits(x, num_points * self.point_dim).reshape(-1, num_points, self.point_dim)
        batches, num_points, point_dim = x.shape
        assert num_points == self.batches * self.points_in_batch, (num_points, self.batches, self.points_in_batch)
        # apply each batch's dedicated phi to its own batch of points
        groups = x.reshape(batches, self.batches, self.points_in_batch * point_dim)
        h = torch.cat(
            [phi(groups[:, i]) for i, phi in enumerate(self.phi)],
            dim=-1,
        )  # (batch, batches * phi_out)
        return self.rho(h)  # (batch, 2): depth score, size score


def predict_values(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
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

