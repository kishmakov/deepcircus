# preparation.conf

- `work_dir` -- staging directory used by generator.
- `seed` -- the generator's only randomness source.
- `entries` -- number of entries written to *each* of the four files per
  bitness.
- `bitness_from`, `bitness_to` -- inclusive range of arities `n` to prepare; one
  set of four files per bitness.
- `small_size_from`, `small_size_to` -- inclusive range of witness-tree internal
  node counts for the `_small` source.
