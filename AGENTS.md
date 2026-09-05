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
- Provide shorter comments. Do not repeat information in comments in different files.


## Code Layout

Layout is documented directory by directory: every directory holding code has a
`README.md` saying what lives in it, and a directory whose content is really its
subdirectories links them instead of repeating them. This file maps the top
level.

| Directory | What is in it |
| --- | --- |
| [`conf/`](conf/README.md) | run configuration -- training runs and offline data preparation |
| [`cpp/`](cpp/README.md) | C++ part of the project |
| [`data/circuits/`](data/circuits/README.md) | the benchmark circuits (`*.aig`/`*.bench`, sizes in `dimensions.txt`); generated offline files `data/m{1,2}_<bitness>.{train,val}` are gitignored and training reads but never writes them |
| `docs/` | the paper (`paper.tex`, `paper.bib`) and format documentation -- [`docs/data_m1.md`](docs/data_m1.md) and [`docs/data_m2.md`](docs/data_m2.md) specify the offline training files |
| `execs/` | symlinks to built binaries under their published names; gitignored |
| [`scripts/`](scripts/README.md) | thin entrypoints: build, test, prepare data, train |
| [`src/`](src/README.md) | Python part of the project |
