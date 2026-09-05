# data/circuits

Reference ISCAS benchmark circuits. Each case is kept in its original
human-readable `.bench` form and as a binary `.aig` file with the same stem.
Offline data preparation and model training do not read this corpus.

To regenerate one AIG file after changing its BENCH source, run:

```bash
data/circuits/convert_bench_to_aig.sh data/circuits/<set>/<case>.bench
```

The converter requires the `abc` executable. Refresh the tracked input/output
summary after changing the corpus with:

```bash
scripts/dimensions.py
```

This rewrites [`dimensions.txt`](dimensions.txt) deterministically from the AIG
headers. Inputs include primary inputs and latches; outputs include primary
outputs and bad-state signals.
