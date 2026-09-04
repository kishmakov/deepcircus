# src

The Python side: for now the model alone, until a training loop over the
prepared offline data is brought in. Nothing here generates input bits or
evaluates functions -- that is [`cpp/`](../cpp/README.md)'s job.

| File | What it owns |
| --- | --- |
| `model.py` | `DeepSetPredictor` and the on-device unpacking |

## model.py

`DeepSetPredictor`: input `(batch, batches * points_in_batch, point_dim)` is
split into `batches` groups, each processed by its own dedicated 2-layer `phi`
MLP (an `nn.ModuleList`, no weight sharing), the outputs concatenated into a
`batches * phi_out` vector and fed to a single `rho` head.

`unpack_bits` expands packed rows into ±1 float32 (little-endian, bit `b` ->
`2*b - 1`) on the device. `forward` unpacks uint8 input itself; float32 input
passes through.
