# Project

This is a research project to study ML approach to handle decision trees.

## Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- In C++ use plain asserts `assert(condition)`
- Keep C++ generation deterministic from `(bitness, seed)`
- Prefer to fail on assert than quiet ignoring of error
- Do not add package-presence guards (e.g. `assert torch is not None`)
- Do not generalize code for running in other environments, it is only run on this machine
- The generator API is bitness-based: use `uint16_t bitness`, not series ids or bit masks


## Code Layout

Layout is documented directory by directory: every directory holding code has a
`README.md` saying what lives in it, and a directory whose content is really its
subdirectories links them instead of repeating them. This file maps the top
level.

| Directory | What is in it |
| --- | --- |
| [`conf/`](conf/README.md) | run configuration -- training runs and offline data preparation |
| [`cpp/`](cpp/README.md) | C++ part of the project |
| `data/` | the benchmark circuits under `data/circuits/` (`*.aig`/`*.bench`, sizes recorded in `data/dimensions.txt`) and the generated offline training files `s{1,2}_<bitness>_{rand,small}.bin`. Gitignored apart from `data/circuits/` |
| `docs/` | the paper (`paper.tex`, `paper.bib`) and format documentation -- [`docs/data_m1.md`](docs/data_m1.md) specifies the offline training files; `docs/old/` keeps retired notes |
| `execs/` | symlinks to built binaries under their published names; gitignored |
| [`scripts/`](scripts/README.md) | thin entrypoints: build, test, run, benchmark, plot, prepare data |
| [`src/`](src/README.md) | Python part of the project |
