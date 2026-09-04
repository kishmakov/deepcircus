# cpp/server

The `offline_server` daemon: the one process Python trains against. It reads the
prepared files of [`docs/data_m1.md`](../../docs/data_m1.md) and hands their
cases over, an epoch at a time. Its client is
[`../../src/daemon/`](../../src/daemon/), and nothing else on either side needs
to know it exists.

| File | What is in it |
| --- | --- |
| [`daemon.h`](daemon.h) | the protocol: one client accepted once, two commands, one shared-memory segment alive at a time |
| [`dataset.h`](dataset.h) | one offline file as cases |

## The point layout

An entry is a pair of functions plus a target; a case is that pair sampled at
`batches * points_in_batch` inputs, one point per input:

```
[ x_1..x_n | g(x), g(x^e_1)..g(x^e_n) | f(x), f(x^e_1)..f(x^e_n) ]
```

so a point is `3n + 2` bits wide. The two targets are the scores of
[`tools/score.h`](../common/tools/score.h), not the raw depth and size the file
stores. Entries carrying the unknown marker are skipped -- bootstrapping them
through the lower-arity models belongs to the training loop, and nothing here
can score them.

## Epochs

Epoch 0 is the validation file; every epoch above it is the training one. The
epoch id enters the seed, so each one draws its own inputs for the same pairs
and the same epoch asked for twice draws the same ones -- which is also why the
sampling can be spread over threads without changing what comes out.
