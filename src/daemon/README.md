# src/daemon

The boundary between Python training and the C++ `offline_server`.

| File | Responsibility |
| --- | --- |
| `__init__.py` | `TrainingData`, the context-managed validation/epoch interface used by the training loop; it also installs bootstrapped targets before serving starts |
| `client.py` | daemon process lifecycle, binary commands, shared-memory copying, and dataset metadata |

Only this package knows the transport. Callers receive NumPy arrays and close
the `TrainingData` context when training finishes. The client copies every
published segment before requesting another one, so returned arrays do not
depend on shared-memory lifetime.
