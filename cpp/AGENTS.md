# C++ generation architecture

The generator, producer, and server sit over the generator-independent
`cpp/common/` and `cpp/tools/` libraries.

## Common

`cpp/common/` holds code shared by otherwise independent executables. The exact
truth-table solvers live in `common/tools/solver.{h,cpp}`, retain the `tools`
namespace, and are included as `"tools/solver.h"`. `SolveForDepth` and
`SolveForSize` are called from `table.cpp`; `kMaxSolvableBitness` is aliased as
`gen::kSolvableTableBitness`.

`common/tools/random.{h,cpp}` is the shared random toolbox: the SplitMix64
finalizer and state step, one-shot mixing, unbiased bounded draws, and fair
boolean draws. Generator seed/case selection and preparation samplers both use
it rather than keeping private mixers or random-stream classes.

The `...Given` and `...Restricted` pairs score the two models of
`docs/paper.tex` over a second truth table -- the same two readings a
`docs/offline_data.md` entry's second table has, `f` for series 1 and the
indicator of `X` for series 2.

`SolveForDepthGiven`/`SolveForSizeGiven` are `S_1[g | f]`: the tree may also
query a helper function of the same inputs, once per path, seeing the value
table of `x_1..x_n` plus `y = f(x)`. They run the one DP over three layers of
the same `3^bitness` states -- helper unqueried, helper fixed to 0, helper fixed
to 1 -- since a path either knows `f`'s value or does not. The fixed-helper
layers depend only on themselves, so they are solved first and then offered to
the unqueried layer as one more transition at every state.

`SolveForDepthRestricted`/`SolveForSizeRestricted` are `S_2[g | X]`: the tree
only has to be right on the subset, and everything outside it is a don't-care.
That is the plain DP over states seeded to hold `X` alone, so it is `Solve` with
a subset rather than a fourth code path.

Both rest on states that can hold no assignment at all -- `seen == 0`, a dead
branch, which costs nothing. Depth and size differ only in how a node combines
its two branches, so all six entry points are one templated DP over a policy;
`cpp/test/test_solver.cpp` pins golden targets and cross-checks every entry
point against a brute-force reference.

## Tools

`cpp/tools/` is its own library in namespace `tools`, holding what is separable
from case bookkeeping: bit walks and input sampling, with no notion of a
case, table, tree, or circuit. The dependency runs strictly one way -- `gen`
links `tools` and `generator.h` includes `sample.h`, while nothing here may name
`gen::`, include a `generator/` header, or link `gen`. That is what lets
`expand_inputs` link `tools` alone; preserve it when adding files.

`sample.{h,cpp}` owns `InputShape` (aliased as `gen::InputShape`) and the whole
path from randomness to input points: the bit primitives (`BitsFromChars`,
`SplitBitsInGroups`, `NextSequence`), `GenerateSequence` (one base sequence off
a `BitSource` -- `Case` passes its own `GenerateBool`), `ExpandInputs` (bases ->
block-and-random walk: batch 0 follows the Gray-code `NextSequence` walk, later
batches flip bit groups keyed by `(batch, point id)`), and `SampleInputs`
composing the two. `Case::Sample` is then only "get the points, evaluate at
each". The `expand_inputs` CLI exposes `ExpandInputs` over stdin/stdout so
Python callers reuse the same walk.

## Generator

`cpp/generator/` is synchronous and contains no worker pool: each call runs to
completion on its calling thread, and independent calls are safe from different
producer threads. `Values` and `Restrictions` are dense, bit-packed matrices
whose rows are cases; exact targets stay separate float vectors of
`gen::kTargetsPerCase` interleaved values per case -- the depth score
`bitness - depth` and the size score `log2(2^bitness - size)`.

`Case` (`case.{h,cpp}`) owns each case's deterministic randomness keyed by
`(bitness, case_id)`: its `mt19937` plus the buffered fair-coin `GenerateBool`
stream. It exposes `SampleValues`/`SampleRestrictions` over an `InputShape`
(`batches` independent samplings, each expanded into `batch_size` power-of-two
points); `TableCase`/`TreeCase` only supply the virtual
`Evaluate(std::vector<bool>)`.

Case-ID sampling is split from generation: `TreeSampleCaseIds`/
`TableSampleCaseIds` sample a bitness x cases pair once, then
`TreeValuesForCases`/`TableValuesForCases`/`TableRestrictionsForCases` generate
a batch for an explicit, pre-sampled chunk. Since each case is deterministic
from `(bitness, case_id)` alone, generating disjoint chunks on different threads
and merging with `Values::Concat` reproduces a single non-chunked call exactly.

## Producer

`cpp/producer/` owns the FIFO `ThreadPool`, the iteration/bitness `TaskQueue`,
and the task types (`TrainingShape`, `Task`, `TaskResult`) the daemon publishes;
wire-level tensor kinds belong to the daemon. `TaskQueue` has no socket or
shared-memory dependency, so it is its own library, exercised directly by
`cpp/test`.

Coordinates (`iteration` x `bitness`) are produced strictly sequentially by a
dedicated producer thread that prefetches results into a bounded buffer
(`TaskQueue::kPrefetchDepth`, currently 8); `Take()` blocks until the next
result in task order is ready, pops it, and thereby unblocks the producer to top
the buffer back up. Within a coordinate the producer samples that tensor's case
ids once, splits them evenly across `ThreadPool::WorkerCount()` into contiguous
chunks, and fans them out. Chunks merge back into one `gen::Values` (for
recursive tables, `Values::Concat` plus `Restrictions::Concat` into a
`gen::GeneratedRestrictions`), so published tensors do not depend on worker
count.

A task's validation set depends only on its bitness, so it is generated once per
bitness (iteration 0) ahead of that bitness's training tasks rather than per
iteration. Iteration-0 tasks carry only a validation tensor with exact targets.
Training tasks (iteration >= 1) carry a train tensor with exact targets (tables
up to the solvable bitness, trees above it) plus, above the solvable bitness, a
recursive-table tensor with restrictions instead of targets.

## Server

`cpp/server/` owns the socket protocol, command loop, and POSIX shared-memory
publication. Only the current coordinate is published; its value and restriction
tensors stay bit-packed (rows padded to whole bytes, little-endian bit order,
bit `b` standing for the float `2*b - 1`) while exact targets are written as
float32, keeping a task's segment ~32x smaller than a float32 expansion. The
Python client unpacks on its side. `server.cpp` owns `main`, constructing a
`TaskQueue` and handing it to the daemon as the task source. Both
`scripts/bench.py` and the training client (`src/generator.py`) speak this
protocol.

## Tests

`cpp/test/` is a Google Test suite (fetched via CMake `FetchContent`, same
pattern as the `aiger` dependency) linked against the `producer` library.

## Validation

`cpp/validation/` is the standalone `validation` executable that checks what
training produces: `scheme.{h,cpp}` owns the operation table -- each operation
carrying a `Preimage` inverse -- and the gate scheme, `tree_scorer.{h,cpp}` owns
everything that costs a walk over every row (`Evaluation` -- the function's
values at a set of rows, and what `Score` takes -- `Tabulate`, and the exact
solvers behind `Score`), `reconstruct.{h,cpp}` searches for a
scheme reproducing a decision tree, and `main.cpp` drives the search and
verifies the result over the validation rows. Keep the dense side on the
`tree_scorer.h` side of that line: it is what caps the reachable bitness, and a
score predicted from a sampled fingerprint would need none of it. A
`ReconstructionState` is its scheme and its score and nothing else -- what it
has left to compute is worked out afresh through `Evaluate`, walking each row of
its unbound slots back to an input reaching it (`ConstructInputs`, inverting one
operation per level) and reading the target there; `Grow` always builds the next
state and cheaply, while
`Validate(original)` is const and only counts the rows on which the state stopped
telling the target apart, zero meaning it is still completable; working out its
residual and scoring it is a separate step
(`Assess` in `reconstruct.cpp`) the search takes for the states it inspects and
no others. The scheme being rebuilt is passed in wherever it is needed and read
a row at a time, so the search holds no tabulation of it at all. Everything the
search judges a state by goes through the one row set `ValidationRows` draws --
internal to `reconstruct.cpp`, with `kValidationBudget` the only part of it
`main.cpp` names: `Validate`, `IsAssembled` (a completed scheme agreeing with the original over
those rows), and `SlotsFingerprint`, the per-slot hash the search dedups on in
place of the columns it used to keep. That is `kValidationBudget` rows --
exhaustive while `2^bitness` fits in it, that many random rows above, where a
dropped distinction can slip through unseen: nothing walks every assignment any
more, so above the budget the result is verified only on the sampled rows.
`Reconstruct` returns the assembled
`ReconstructionState` it ended on and reports what the run cost only through
`PrintStep`'s per-step line; checking the rebuilt scheme against the original is
`main.cpp`'s job, not the search's -- it calls `Validate` on the returned state. One counter bounds a run: past
`kMaxProcessed` states pushed the search stops adding to the heap and finishes
on what it holds, aborting if that drains without assembling. A `TreeScore`'s `log_size` is on the model's
own scale -- `log2(2^slots - size)`, a hand-kept copy of `gen::SizeScore`, since
validation cannot link `gen` -- so a *cheaper* tree scores *higher* there, which
is why the search's ordering negates it.

It links `common` for the exact solvers and nothing else -- like `expand_inputs`,
it stays on the generator-independent side, so no `gen::` names, no
`generator/` includes, no `gen` link edge.

`main.cpp`'s asserts are the checks, not debug scaffolding, so its CMake entry
undefines `NDEBUG` for that one file; the rest of the target builds Release like
everything else. `scripts/build.sh` builds it with the whole project;
`cpp/validation/build.sh` builds just this target out of the same `cpp/build`
tree.
