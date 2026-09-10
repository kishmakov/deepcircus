import numpy as np
import torch
import torch.nn as nn

from src.config import ModelConfig, OrderedConfig, TrainConfig

assert torch.cuda.is_available(), "CUDA is required for training"
DEVICE = "cuda"

# ── Model ─────────────────────────────────────────────────────────────────────

def statistics_shape(bitness: int) -> tuple[int, int]:
    """The pooled products: `3n + 1` left operands by the `n + 1` right ones."""
    return 3 * bitness + 1, bitness + 1


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
    """Pools a shared net over the sampled batches, and another over the products."""

    def __init__(
        self,
        batches: int,
        points_in_batch: int,
        point_dim: int,
        phi_hidden: int,
        phi_out: int,
        psi_hidden: int,
        psi_out: int,
        rho_hidden: int,
        rho_out: int,
        dropout: float,
    ):
        super().__init__()
        assert (point_dim - 2) % 3 == 0, point_dim
        self.bitness = (point_dim - 2) // 3
        self.batches = batches
        self.points_in_batch = points_in_batch
        self.points = batches * points_in_batch
        self.point_dim = point_dim

        # phi: shared over the batches, reading the raw points of one
        self.phi = nn.Sequential(
            nn.Linear(points_in_batch * point_dim, phi_hidden),
            nn.BatchNorm1d(phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(phi_hidden, phi_out),
            nn.BatchNorm1d(phi_out),
            nn.ReLU(),
        )

        # psi: shared over the rows of the pooled product matrix
        self.psi = nn.Sequential(
            nn.Linear(statistics_shape(self.bitness)[1], psi_hidden), nn.ReLU(),
            nn.Linear(psi_hidden, psi_out), nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(2 * phi_out + 2 * psi_out, rho_hidden), nn.ReLU(),
            nn.Linear(rho_hidden, rho_out), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(rho_out, 2),
        )

    def unpacked(self, x: torch.Tensor) -> torch.Tensor:
        # packed uint8 rows (batch, row_bytes) -> ±1 float32 points, on-device
        assert x.dtype == torch.uint8, x.dtype
        return unpack_bits(x, self.points * self.point_dim).reshape(-1, self.points, self.point_dim)

    def statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Every product of two different blocks of a point, averaged over them.

        The three blocks stack into one `(3n + 1, n + 1)` matrix, since each is
        against `f` or `g`: the input bits against `f`, against `g`, then `f`
        against `g`.
        """
        assert x.shape[1:] == (self.points, self.point_dim), x.shape
        n = self.bitness
        inputs = x[:, :, :n]
        g = x[:, :, n : 2 * n + 1]
        f = x[:, :, 2 * n + 1 :]
        blocks = (torch.einsum("bpi,bpj->bij", inputs, f),
                  torch.einsum("bpi,bpj->bij", inputs, g),
                  torch.einsum("bpi,bpj->bij", f, g))
        return torch.cat(blocks, dim=1) / self.points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unpacked(x)
        groups = x.reshape(len(x) * self.batches, self.points_in_batch * self.point_dim)
        phi_pooled = self.phi(groups).reshape(len(x), self.batches, -1)
        psi_pooled = self.psi(self.statistics(x))
        # (batch, 2): depth score, size score
        return self.rho(torch.cat((phi_pooled.mean(1), phi_pooled.amax(1),
                                   psi_pooled.mean(1), psi_pooled.amax(1)), dim=-1))


def make_predictor(
    config: TrainConfig, bitness: int | None = None, model_name: str | None = None,
) -> nn.Module:
    model_bitness = config.bitness if bitness is None else bitness
    if isinstance(config.model, OrderedConfig):
        from src.ordered import OrderedRestrictionPredictor

        name = config.model_name if model_name is None else model_name
        return OrderedRestrictionPredictor(model_bitness, config.sampling.points, name, config.model)
    assert isinstance(config.model, ModelConfig), config.model
    return DeepSetPredictor(
        point_dim=3 * model_bitness + 2,
        batches=config.sampling.batches,
        points_in_batch=config.sampling.points_in_batch,
        phi_hidden=config.model.phi_hidden,
        phi_out=config.model.phi_out,
        psi_hidden=config.model.psi_hidden,
        psi_out=config.model.psi_out,
        rho_hidden=config.model.rho_hidden,
        rho_out=config.model.rho_out,
        dropout=config.model.dropout,
    )


def predict_values(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    predict_batch_size: int,
) -> np.ndarray:
    assert not model.training

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), predict_batch_size):
            xb = torch.as_tensor(x[start : start + predict_batch_size], device=DEVICE)
            predictions.append(model(xb).cpu().numpy())

    result = np.concatenate(predictions)
    assert result.dtype == np.float32, result.dtype
    return result
