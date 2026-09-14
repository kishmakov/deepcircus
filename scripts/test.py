"""The Python half of `test.sh`, which runs it against the files that script
generated: the offline server driven through its client, then tiny M1 and M2
runs of the predictor. Takes the data directory as its only argument, and
needs `OFFLINE_SERVER` pointing at the sanitized binary.
"""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.bootstrap import combine_helper_predictions, combine_primary_predictions
from src.config import (OptimizerConfig, OrderedConfig, SamplingConfig,
                        TrainConfig, TrainingConfig)
from src.daemon.client import VALIDATION_EPOCH, Client
from src.model import OrderedRestrictionPredictor, OrderedTopology, make_predictor
from src.train import run_training


def check_ordered_inference_cache(model_name: str, device: str) -> None:
    rng = np.random.default_rng(239)
    model = OrderedRestrictionPredictor(
        9, 32, model_name, OrderedConfig(hidden=8, branch_hidden=32, head_hidden=64, orders=4),
    ).to(device)
    previous_topology = None
    for _ in range(2):
        inputs = ((rng.choice(512, 32, replace=False)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
        rows = rng.integers(0, 2, (2, 32, 29), dtype=np.uint8)
        rows[:, :, :9] = inputs
        packed = torch.as_tensor(np.packbits(rows.reshape(2, -1), axis=1, bitorder="little"), device=device)
        model.eval()
        with torch.inference_mode():
            expected = model(packed)
        topology = model._topology
        assert topology is not previous_topology
        model.train()
        model.zero_grad(set_to_none=True)
        predicted = model(packed)
        assert model._topology is topology
        torch.testing.assert_close(predicted, expected)
        predicted.square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()), model_name
        previous_topology = topology


def check_ordered_checkpoint(model_name: str, device: str) -> None:
    rng = np.random.default_rng(239)
    rows = rng.integers(0, 2, (3, 32, 32), dtype=np.uint8)
    rows[1, :, :10] = rows[0, :, :10]
    packed = torch.as_tensor(np.packbits(rows.reshape(3, -1), axis=1, bitorder="little"), device=device)
    model = OrderedRestrictionPredictor(
        10, 32, model_name, OrderedConfig(hidden=8, branch_hidden=32, head_hidden=64, orders=4),
    ).to(device)
    predicted = model(packed)
    predicted.square().mean().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    assert all(grad.isfinite().all() for grad in gradients), model_name
    model.zero_grad(set_to_none=True)
    # Compare against the original update with all activations retained.
    with patch("src.model.checkpoint", lambda function, *args, **kwargs: function(*args)):
        expected = model(packed)
        expected.square().mean().backward()
    torch.testing.assert_close(predicted, expected)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)


client = Client("m1", 8, Path(sys.argv[1]), 239, 2, 4)
assert client.sizes.train_known == 4, client.sizes
assert client.sizes.validation_known == 2, client.sizes
validation = client.fetch(VALIDATION_EPOCH)
epoch = client.fetch(1)
assert validation.values.shape == (2, 26), validation.values.shape
assert epoch.values.shape == (4, 26), epoch.values.shape
assert epoch.targets.shape == (4, 2), epoch.targets.shape
# Closing hangs up, and the daemon's exit status -- LeakSanitizer included --
# is asserted on the way out.
client.close()

client = Client("m1", 13, Path(sys.argv[1]), 239, 2, 4)
assert client.sizes.unknown_train == 1, client.sizes
assert client.primary_reductions(0, 1).values.shape == (26, 38)
assert client.helper_reductions(0, 1).values.shape == (2, 41)
client.set_unknown_targets(np.array([[1.25, 2.5]], dtype=np.float32))
epoch = client.fetch(1)
assert epoch.values.shape == (2, 41), epoch.values.shape
assert np.array_equal(epoch.targets[1], np.array([1.25, 2.5], dtype=np.float32)), epoch.targets
client.close()

primary = np.array([[11, 10], [9, 8], [7, 6], [7, 6]], dtype=np.float32)
combined = combine_primary_predictions(primary, parents=1, candidates=2, child_bitness=12)
assert np.isclose(combined[0, 0], 9), combined
assert np.isclose(combined[0, 1], np.log2(2**10 + 2**8 - 1)), combined
helper = combine_helper_predictions(np.full((2, 2), 13, dtype=np.float32), parents=1, bitness=13)
assert np.isclose(helper[0, 0], 12), helper
assert np.isclose(helper[0, 1], np.log2(2**13 - 1)), helper
print("served known and bootstrapped training cases, daemon exited cleanly")

for model_name in ("m1", "m2"):
    check_ordered_inference_cache(model_name, "cpu")
    check_ordered_inference_cache(model_name, "cuda")
    check_ordered_checkpoint(model_name, "cpu")
    check_ordered_checkpoint(model_name, "cuda")
    config = TrainConfig(
        model_name=model_name,
        bitness=8,
        data_dir=Path(sys.argv[1]),
        work_dir=Path(sys.argv[1]) / "work",
        seed=239,
        sampling=SamplingConfig(batches=2, points_in_batch=256),
        training=TrainingConfig(epochs=1, batch_size=4, rmse_threshold=0.0001),
        model=OrderedConfig(hidden=8, branch_hidden=32, head_hidden=64),
        optimizer=OptimizerConfig(
            lr=0.001, scheduler_patience=1, scheduler_factor=0.5, scheduler_min_lr=0.0001
        ),
    )
    run_training(config)
    assert config.checkpoint_path().is_file(), config.checkpoint_path()
    assert config.best_checkpoint_path().is_file(), config.best_checkpoint_path()
    assert config.metrics_path().is_file(), config.metrics_path()
    client = Client(model_name, 8, config.data_dir, config.seed, 2, 256)
    cases = client.fetch(VALIDATION_EPOCH)
    client.close()
    ordered = make_predictor(config).cuda()
    packed = torch.as_tensor(cases.values, device="cuda")
    bits = np.unpackbits(cases.values, axis=1, bitorder="little").reshape(-1, 512, 26)

    def repack(points):
        return torch.as_tensor(
            np.packbits(points.reshape(len(points), -1), axis=1, bitorder="little"), device="cuda")

    complemented = bits.copy()
    complemented[:, :, 5] ^= 1
    predicted = ordered(packed)
    predicted.square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in ordered.parameters()), model_name
    with torch.no_grad():
        # Points are a set, and the coordinates carry no order or polarity.
        torch.testing.assert_close(ordered(repack(bits[:, ::-1])), predicted)
        torch.testing.assert_close(ordered(repack(complemented)), predicted)
    # Sparse coverage has no complete-table requirement, including duplicate points.
    partial = make_predictor(replace(config, sampling=SamplingConfig(2, 16))).cuda()
    partial.load_state_dict(ordered.state_dict())
    with torch.no_grad():
        sparse = partial(repack(bits[:, :32]))
        torch.testing.assert_close(partial(repack(bits[:, :32][:, ::-1])), sparse)
        assert partial(repack(np.repeat(bits[:, :1], 32, axis=1))).isfinite().all()

# At bitness 8 this ordering budget happens to retain the whole lattice, which is
# why its accuracy matches the exhaustive model; the bound it obeys is elsewhere.
coordinates = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
extended = np.concatenate((np.pad(coordinates, ((0, 0), (0, 1))),
                           np.pad(coordinates, ((0, 0), (0, 1)), constant_values=1)), 0)
full = OrderedTopology(extended, orders=64, seed=239, device=torch.device("cpu"))
assert full.node_count == 3**9, full.node_count
sparse = OrderedTopology(extended, orders=4, seed=239, device=torch.device("cpu"))
assert sparse.node_count < 3**9, sparse.node_count
print(f"trained M1 and M2; checked invariance, sparse coverage, and {full.node_count} lattice nodes")
