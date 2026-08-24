# cpp/server

The `generator_server` daemon. It accepts training requests, obtains generated
tasks from [`../producer/`](../producer/README.md), and publishes their tensors
through shared memory.

The Python client is [`../../src/generator.py`](../../src/generator.py).
