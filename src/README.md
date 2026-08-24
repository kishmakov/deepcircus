# src

The Python side: the client of the C++ generator daemon, the model, and the
training loop. Nothing here generates input bits or evaluates functions -- that
is [`cpp/`](../cpp/README.md)'s job, reached over the daemon protocol.

| File | What it owns |
| --- | --- |
| `config.py` | config parsing (`conf/*.conf` through OmegaConf) plus snapshot, state, and resume details |
| `generator.py` | the generator daemon client: spawning the server, the task protocol, shared-memory task views |
| `model.py` | `DeepSetPredictor` and the on-device unpacking |
| `sampler.py` | the thin generator wrapper and pipelined dataset orchestration |
| `train.py` | the bitness training loop, model construction/loading/saving, per-epoch optimization |

## config.py

`TrainConfig` and its `training` / `model` / `optimizer` sections, loaded from
[`conf/`](../conf/README.md). It also owns the bitness snapshot -- the record of
how far a run got -- including atomic saves, normalization, resume, and weight
pruning. Training goes through the config's methods (`iterations_range`,
`bitness_range`, `snapshot_path`, `model_key`, ...) rather than reading snapshot
internals itself.

## generator.py

`load_generator` spawns `generator_server`, `Generator.initialize` sends the
training shape, and `next_task` borrows the next coordinate. A `GeneratedTask`
copies every tensor out of shared memory at parse time, so it stays valid after
`release()`.

Tensors stay bit-packed all the way to the GPU: value tensors keep their compact
uint8 rows `(cases, row_bytes)`, and the large restrictions matrix is streamed as
packed row chunks through `restriction_chunks()` with its logical shape exposed
as `restrictions_shape`. Only exact targets arrive as float32. Expansion to ±1
float32 happens on-device in `model.unpack_bits`.

`expand_inputs` is the client helper over the `expand_inputs` CLI, so Python
reuses the exact C++ input walk instead of mirroring it.

## model.py

`DeepSetPredictor`: input `(batch, batches * points_in_batch, point_dim)` is
split into `batches` groups, each processed by its own dedicated 2-layer `phi`
MLP (an `nn.ModuleList`, no weight sharing), the outputs concatenated into a
`batches * phi_out` vector and fed to a single `rho` head.

`unpack_bits` expands packed rows into ±1 float32 (little-endian, bit `b` ->
`2*b - 1`) on the device. `forward` unpacks uint8 input itself; float32 input
passes through.

## sampler.py

`GeneratorProxy` wraps the daemon client; `Sampler` walks the
`(iteration, bitness)` coordinates in the order the daemon produces them. Since
a validation set depends only on its bitness, the daemon emits one
validation-only task per bitness ahead of the training tasks, and `Sampler`
pulls them all up front and keeps them for the whole run. Above the solvable
bitness a training task also carries approximate values, whose targets come from
the previous bitness's model over the task's restrictions.

## train.py

`run_training` walks the coordinates, building or loading the model for each
bitness, running epochs against the RMSE threshold, checkpointing through the
snapshot, and reporting per-iteration metrics.
