# cpp/common/offline

Reader and writer for packed offline training data. One envelope serves both
models: it is specified in [`docs/data_m1.md`](../../../docs/data_m1.md), and
[`docs/data_m2.md`](../../../docs/data_m2.md) says what `M_2` puts in it.

[`../../prep/`](../../prep/README.md) produces these files, and
the test suite covers their round trip.
