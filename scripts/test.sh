#!/usr/bin/env bash
set -euo pipefail

# Builds the whole C++ tree with ASan/UBSan and every assert alive -- Debug, so
# no `NDEBUG` anywhere -- then runs the test suite and, after it, each
# executable once on a small input of its own. The executables' error handling
# is their asserts, and the suite does not reach those; a run does. Extra
# arguments go to GoogleTest.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/cpp/build-asan"

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -G Ninja \
    -DDEEPCIRCUS_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build "$BUILD_DIR"

export ASAN_OPTIONS="detect_leaks=1:${ASAN_OPTIONS:-}"

"$BUILD_DIR/test/deepcircus_tests" "$@"

# Everything the executables write goes here and leaves with the run.
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo
echo "== data_generator: tt and general entries at bitness 8 and 13"
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m1 8 239 2 2 2
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m2 8 239 2 2 2
"$BUILD_DIR/prep/data_generator" "$WORK_DIR" m1 13 239 1 1 1
ls "$WORK_DIR"

echo
echo "== validation: reconstruct a random 10-input scheme"
"$BUILD_DIR/validation/validation" 10 122 | tail -2

echo
echo "== offline_server: serve the files above, driven by the Python client"
cd "$ROOT"
OFFLINE_SERVER="$BUILD_DIR/offline_server" "$ROOT/.venv/bin/python" - "$WORK_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np

from src.bootstrap import combine_helper_predictions, combine_primary_predictions
from src.config import ModelConfig, OptimizerConfig, SamplingConfig, TrainConfig, TrainingConfig
from src.daemon.client import VALIDATION_EPOCH, Client
from src.train import run_training

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
    config = TrainConfig(
        model_name=model_name,
        bitness=8,
        data_dir=Path(sys.argv[1]),
        work_dir=Path(sys.argv[1]) / "work",
        seed=239,
        sampling=SamplingConfig(batches=2, points_in_batch=4),
        training=TrainingConfig(epochs=1, batch_size=4, rmse_threshold=0.0001),
        model=ModelConfig(phi_hidden=8, phi_out=4, rho_hidden=8, dropout=0.0),
        optimizer=OptimizerConfig(lr=0.001, scheduler_patience=1, scheduler_factor=0.5),
    )
    run_training(config)
    assert config.checkpoint_path().is_file(), config.checkpoint_path()
    assert config.best_checkpoint_path().is_file(), config.best_checkpoint_path()
    assert config.metrics_path().is_file(), config.metrics_path()
print("trained M1 and M2 for one epoch each")
PY

echo
echo "all executables ran with assertions enabled"
