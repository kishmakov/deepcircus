# Experiments log

Every architecture tried for `M_1`/`M_2`, what it scored, and what the score
settled. Unless a line says otherwise, numbers are RMSE on the 1,024 held-out
`m1 8` pairs, the two scores reported separately, measured against the exact
solvers of [`tools/solver.h`](../cpp/common/tools/solver.h).

The target is both scores under 0.1. The cubic ordered lattice reaches it on
both training and validation; section 6 records that follow-up and section 7
the independent check of it. Sections 1-5 preserve the earlier investigation of
the quadratic architecture.

**What the tree still contains.** Only the ordered architecture ships, as the
default `conf/train.yaml`, together with the earlier `DeepSetPredictor` under
`conf/train_sampled.yaml`. The complete restriction lattice, the certificate
architecture and the cubic pooling experiment were removed once the ordered
lattice passed: they are recorded here, not carried. Numbers below that name a
removed configuration describe runs made when it existed.

| Architecture | states per case | parameters | val depth | val size | config |
| --- | ---: | ---: | ---: | ---: | --- |
| sampled DeepSet | -- | 1,022,402 | 0.5345 over both scores | | `train_sampled.yaml` |
| complete restriction lattice | 19,683 | 1,514 | **0.0463** | **0.0706** | removed |
| certificates | 9 x 1,024 x 10 | 101,687 | 0.1379 | **0.0176** | removed |
| ordered restriction lattice | 19,683 at 8 bits; cubic bound | 1,514 | **0.0418** | **0.0525** | `train.yaml` |

## What everything is measured against

**The targets are exactly the restriction recursion.** Reimplementing the
dynamic program over the 19,683 states reproduces every stored depth exactly and
every stored size score to 5e-7. There is nothing to learn beyond that
recursion, and nothing else to blame for an error.

**The two scores are not equally hard.** On validation the depth score has
spread 2.554 and the size score 0.184; on training, 2.510 and 0.308. Depth is an
integer between 0 and 8, so RMSE 0.1 on it means naming it exactly on about 99%
of the pairs, and the lattice model's 0.046 means 99.5%. Size has no such
threshold behaviour.

**Minimum decision-tree depth is NP-hard** for a function given compactly, so no
model whose cost is polynomial in the bitness computes it. Everything below is
about how close an affordable surrogate gets on this distribution.

## 1. Architectures

### 1.1 Sampled DeepSet -- `conf/train_sampled.yaml`

Flattens sampled groups and pools cross-block products; no recursion. Best
validation RMSE 0.534487 over both scores after 128 epochs, learning rate still
at 0.001. 1,022,402 parameters.

*Conclusion.* Globally pooled statistics do not reach the targets at all.

### 1.2 Complete restriction lattice -- `conf/train_restrictions.yaml`

```bash
python -m src.train --model m1 --bitness 8 --config conf/train_restrictions.yaml
```

Reconstructs complete truth tables from the sampled points and their single-bit
neighbours, then works bottom-up over every input restriction: a shared MLP
takes the maximum and sum of two child embeddings for each possible next query,
minimum pooling compares the candidates, constant or empty restrictions have
zero embedding, and a learned root head predicts both scores. The helper is one
more queryable coordinate for M1; M2 keeps only the assignments its subset
indicator marks.

Three epochs from scratch, about 126 seconds on the RTX 5070, 1,514 parameters:

| Epoch | Train RMSE | Validation RMSE |
| ---: | ---: | ---: |
| 1 | 0.979170 | 0.100856 |
| 2 | 0.082326 | 0.077492 |
| 3 | **0.065326** | **0.059715** |

At epoch 3, training depth/size were 0.049740/0.077852 and validation
0.046312/0.070619. Evaluated afterwards without updates:

| Evaluation | Sampled checkpoint | Lattice checkpoint |
| --- | ---: | ---: |
| All training pairs | 0.441525 | **0.052511** |
| Training independent tables | 0.331104 | **0.042806** |
| Training tree-over-table pairs | 0.529395 | **0.060683** |
| Validation, sampling seeds 239/240/241 | 0.534487/0.536489/0.539387 | **0.059715** each |

Predictions agree across sampling seeds because every sampling recovers the same
complete tables.

*Conclusion.* Both scores under 0.1, and the only architecture that gets there
on depth. It costs `3^(n+1)` states for M1 and `3^n` for M2, caps bitness at 10,
and asserts complete assignment coverage -- which is the whole objection to it,
and what the rest of this log is about.

### 1.3 Certificates -- `conf/train.yaml`, the default

Keep the recursion; stop indexing its states by restriction.

A restriction's state is collapsed onto the cells it contains, so the sampled
points carry it. Each sampled point is a cell -- two for M1, since the helper is
one more coordinate and only one of its values is reachable. Points one flip
apart are linked; a link the sample missed loses only a state, because the point
row already carries `g` and `f` at every one of its neighbours.

`dimensions` rounds of message passing follow. Each round pools messages over
the query coordinates and over the cells, so what a round decides is one choice
rather than a choice per cell -- a tree queries one coordinate at a node. A cell
settled in a round stays settled, so the rounds it spends unsettled count the
coordinates its certificate fixes, and the worst cell's count is the depth up to
a residual the head predicts over the possible corrections.

Each cell also carries one state per coordinate: the subcube grown from that
cell along that coordinate, which each round grows by joining this cell's half to
the half across it. They talk only along their own direction, which is what
keeps the cost quadratic, and what they settle at is the certificate of the
restriction on that coordinate -- read directly rather than inferred. Section
3.6 is why they are there.

The exact top two levels of the recursion, over restrictions of one and of two
coordinates, read the same states: the test for a constant restriction stays
exact there, and branches combine as the lattice model combines them.

Cost per case is `O(points * bitness^2)` for the rounds and the two exact
levels, plus `O(points^2 * bitness)` to find the links. Nothing allocates
anything of size `2^n`, so the bitness cap and the coverage assertion are gone.

```bash
./scripts/train/train_model.sh m1 8
```

512 sampled points, batch 32, Adam at 0.001 with the plateau scheduler and the
gradient norm capped at 1; 30 epochs at about 330 seconds each, 101,687
parameters. Best epoch, the 27th:

| | depth | size |
| --- | ---: | ---: |
| training | 0.2646 | 0.0226 |
| validation | **0.1379** | **0.0176** |

Evaluated afterwards without updates, on the held-out pairs resampled under
three sampling seeds and on two freshly sampled training epochs:

| Evaluation | depth | size | depth named exactly |
| --- | ---: | ---: | ---: |
| validation, sampling seed 239 | 0.137885 | 0.017598 | 97.66% |
| validation, sampling seed 240 | 0.138803 | 0.017496 | 97.56% |
| validation, sampling seed 241 | 0.139399 | 0.017362 | 97.56% |
| training epoch 1 | 0.258024 | 0.020297 | 90.61% |
| training epoch 7 | 0.258027 | 0.020306 | 90.63% |

*Conclusion.* Size is answered better than the exponential model answered it, at
quadratic cost. Depth is not: 0.138 against 0.046.

**Where the depth error lives.** On a sampled training epoch the 32,768
`tree-over-table` entries -- what validation is drawn from -- score depth RMSE
0.1268 with the depth named exactly on 97.9% of them, while the 32,768 `general`
entries score 0.3421 and 83.4%. Those are pairs of independent random tables,
mean depth 7.37, and telling depth 7 from depth 8 is where the error sits: a
random function has no short certificate to find, so the surrogate the network
computes has nothing to separate the two cases with. The lattice separates them
by construction.

## 2. What cheap surrogates for depth are worth

Each row is a computation run against the exact solver on the same 1,024 pairs.
The last two columns are its RMSE and the fraction of depths it names exactly.

| Surrogate | RMSE | exact |
| --- | ---: | ---: |
| greedy top-down split, beam 1 to 16, impurity | 2.90 - 3.63 | 13 - 21% |
| greedy top-down split, beam 1 to 16, entropy | 2.02 - 2.46 | 25 - 34% |
| greedy oblivious tree, one shared query per level | 2.92 | 35% |
| naive per-cell relaxation, hard presence gate | 4.66 | 13% |
| greedy certificate growth per cell (greedy set cover) | 1.24 | 54% |
| ANF degree of `g` | -- | 51% |
| certificate complexity `max_c delta(c)`, exact | 0.392 | 85% |
| `max(certificate, ceil(log2(size + 1)))`, exact size | 0.375 | 87% |
| `max(certificate, 1 + min_v max_b certificate(rho))`, exact | 0.349 | 88% |
| MLP on exact certificate histogram + exact size | 0.205 | 95% |
| the same MLP, plus subcube parities | 0.219 | 94% |
| subcube parities alone | 0.494 | 70% |

Notes on individual rows:

- **Top-down beams overshoot**, by +1.6 to +2.8 on average, and a *wider* beam
  is *worse*: the beam optimises the adversary's branch while the builder's
  choice of query stays a heuristic. Picking a query greedily is exactly what
  the recursion's minimum is for.
- **The naive per-cell relaxation collapses.** Growing each cell's subcube by a
  hard `argmin` over directions, with only a presence pair as state, finds every
  cell "constant" almost immediately, because the two halves it joins may come
  from different free sets. Predicted depth goes to 0 on 88% of the pairs. This
  is why the shipped model gates softly and learns the combination.
- **Certificate complexity is the right view.** `delta(c)`, the number of
  coordinates a certificate for cell `c` fixes, names the depth on 85% of the
  pairs through a plain maximum over cells, and the depth is never below it.
  That is what put the model's states on cells.
- **The ANF degree is a genuinely different signal** -- it correlates 0.854 with
  the depth and is an upper bound on 99.5% of the pairs -- and its top
  coefficients are cheap, since the parity of a restriction is its reachable-one
  count modulo two, which the pooling already computes. On top of the
  certificates it adds nothing (0.219 against 0.205). Degrees of `f`, `g xor f`
  and `g and f` correlate 0.09, 0.13 and 0.42, so it is `g`'s own structure that
  carries the signal.
- **The 0.205 row is the ceiling this bounds.** A strong MLP over *exact*
  certificate statistics and the *exact* size score stops at 95% exact. The
  shipped network gets past it -- 97.9% on the distribution validation is drawn
  from -- so its cell states carry more than the certificate histogram; but 0.046
  needs 99.5%.

## 3. Modifications tried on the certificate architecture

### 3.1 Complete sampling coverage -- kept

The cells are the distinct inputs among `batches * points_in_batch` points. At
bitness 8: `batches: 8, points_in_batch: 128` gives 1,024 points covering 255 of
256 assignments on average; `2, 128` gives 256 points and only 192; `4, 128`
gives 240; `3, 128` gives 224; `4, 64` gives 175. **`2, 256` covers all 256 in
every case, with half the points of the old default**, and is what `train.yaml`
now asks for.

*Result.* An early run on the 192-cell sampling was still at validation depth
0.44 after eight epochs, where the complete sampling was past that by the third.
*Conclusion.* Kept -- coverage matters more than point count.

### 3.2 The exact top two levels of the recursion -- kept

Restrictions of one and of two coordinates, `O(n^2)` of them, pooled from the
cell states with the constant-restriction test kept exact.

*Result.* At matched epochs on the 192-cell sampling, 0.410 against 0.436 at
epoch 8. *Conclusion.* A modest, consistent gain; kept. It also makes depths 0,
1 and 2 exactly decidable, which is 36% of the validation pairs.

### 3.3 Discrete residual head -- kept

The depth is the worst cell's certificate count plus a residual read as the
expectation of a softmax over the `2 * dimensions + 1` possible integer
corrections, rather than a free scalar.

*Conclusion.* Kept; it converged more cleanly than the plain linear head.

### 3.4 Leaning the loss onto depth -- no effect

Size is seven times better than it needs to be, so `training.size_weight: 0.2`
spends the gradient on depth instead.

*Result.* Eight epochs held validation depth between 0.143 and 0.166 and
training depth at 0.258. *Conclusion.* No effect; the default is 1.0.

### 3.5 More computation per certificate level -- no effect

`model.steps: 2` runs two message-passing steps inside each of the `dimensions`
rounds, so information travels 18 hops instead of 9, at twice the cost. The
deeper unrolled recurrence **diverged at epoch 17** of the first attempt --
training depth jumped from 0.307 to 0.604 in one epoch -- which is why
`training.gradient_clip` exists.

*Result.* Trained the same way with clipping, its best epoch scored validation
depth 0.1418 and size 0.0204: the same place, for twice the compute.
*Conclusion.* What limits the model is the representation, not the budget it is
given to compute in. The default is 1.

### 3.6 Per-coordinate axis states -- kept

Collapsing restrictions onto cells forgets *which* subcube a cell's certificate
came from, and that is exactly the gap between `max_c delta(c)` and the depth.
So each cell carries one state per coordinate, joined each round to the half
across that coordinate -- the recursion's own step, with the free set kept.
Because a state only ever talks along its own direction, the cost stays
`O(points * bitness^2)`.

*Result.* At a matched schedule on full data, validation depth 0.350 / 0.264 /
0.228 over the first three epochs against 0.556 / 0.332 / 0.267 without them.
Over a full run: **0.1379 / 0.0176 in 27 epochs, against 0.1422 / 0.0199 in 38**.
*Conclusion.* The largest single gain of the session and the current default,
though it lands in the same region as everything else.

### 3.7 Continuing a converged run at a lower learning rate -- no effect

*Result.* From its own weights at `lr: 0.00025`, eight further epochs held
validation depth between 0.143 and 0.166 while the learning rate halved twice
again. *Conclusion.* The runs are converged for this width, not stopped early.

### 3.8 A beam over the recursion -- rejected before building

The obvious remaining idea: keep the recursion exact but expand only the best
few coordinates at each node, which would be quadratic for a constant beam. It
needs a cheap ranking that reliably contains the optimal query.

*Result.* Ranking the root's nine coordinates and scoring the children exactly,
an optimal split is in the top one on 52.1% of the non-constant pairs by
impurity and 53.5% by entropy; in the top four, 68.6% and 68.9%.
*Conclusion.* That is the root alone, and the loss compounds down the levels. A
beam of constant width cannot hold a recursion whose minimum it misses a third
of the time at the first node. Not built.

## 4. Bugs the invariance checks caught

The model must be invariant to point order, to renaming input bits, and to
complementing one. Each of these was a real defect found by asserting that:

1. **Slot pairing in the two exact levels.** Restriction slots are laid out as
   `coordinate * 2 + value`; a reshape read them as `value * dimensions +
   coordinate`, so the recursion combined branches that were not siblings.
2. **A coordinate-indexed usage head.** Reading "how much does fixing coordinate
   `v` help" from output row `v` of a linear layer ties the answer to the bit's
   *name*. It now reads one message, so it follows the coordinate.
3. **Coordinate-ordered features.** Per-coordinate certificates were flattened
   in coordinate order. They are now sorted, with a key pairing both of a
   coordinate's values so that ties still rank the same way.
4. **The axis twin under incomplete coverage.** A direction the sample missed
   made the gather read whichever cell the link fell back on, which depends on
   point order. Masked, like the cell messages already were. This one only bites
   when links are missing -- never at bitness 8 with complete coverage, but
   exactly the regime the architecture exists for.

`scripts/test.sh` covers all three invariances for M1 and M2 alongside the 43
C++ tests under ASan/UBSan, and trains a tiny run of each of the three
architectures.

## 5. Where this leaves it

The size score is answered better than the exponential model answered it, at a
cost that grows quadratically. The depth score is not: 0.138 against 0.046.

The reason is not a shortfall of tuning. The exact recursion is what the lattice
model was buying with its `3^(n+1)` states; every affordable surrogate measured
in section 2 tops out near 95% of the depths named exactly, which is RMSE 0.2;
and four independent attempts to move the number -- leaning the loss onto depth,
continuing at a tenth of the learning rate, doubling the message passing, and
per-coordinate axis states -- all land between 0.138 and 0.142. A beam over the
recursion loses the optimal split too often at the root to stand in for the
minimum. The network is already past the certificate ceiling, at 97.9% on the
distribution validation is drawn from, but 0.046 needs 99.5%.

So the trade is: the exponential model is exact, capped at bitness 10, and needs
complete coverage; the certificate model costs `O(points * bitness^2)`, has
neither cap nor coverage requirement, wins on size, and gives up a factor of
three on depth.

Both export under the same `data/m1_08.best.pt`, so that file belongs to
whichever run finished last; each configuration keeps its own `work_dir`.

Convergence is verified for M1 at 8 bits. M2 has smoke coverage only.

## 6. Cubic follow-up: sparse ordered restrictions (2026-09-06)

The new target is **both depth and size RMSE below 0.1**, with a scheme that
can execute through 256 bits. The successful change is `ordered.py`, selected
by `conf/train_ordered.yaml`. The training and validation files were unchanged:
65,536 training pairs and the same 1,024 held-out pairs. Sampling uses 512
points, seed 239. Results below concern M1 at 8 bits unless stated otherwise.

### 6.1 Retain a cubic family of restrictions

Choose a fixed number `K` of random variable orderings. For each ordering,
allow fixed-coordinate sets that are a prefix with at most two coordinates
postponed. Join those families; an edge queries a coordinate whose child set
is also allowed. This retains explicit query choices and child restrictions.
The network learns the same max/sum branch combination and minimum pooling
as the complete lattice, with 1,514 parameters.

For `d` queryable coordinates, one ordering has at most
`1 + d + C(d,2) + C(d,3)` distinct fixed-coordinate sets. Each set contains at
most `Q` nonempty sampled assignments (`Q <= 2 * points` for M1), giving
`O(K * Q * d^3)` nodes. The number of next-query candidates is at most `3K`
per node. `K` is a configuration constant independent of bitness.

The implementation discovers groups of sampled cells with integer bitsets;
it never allocates a complete `2^d` table or `3^d` lattice. Singleton groups
stop the recursion. Constant and duplicate/complemented query columns on the
sample are collapsed before constructing topology. Only the most recent
geometry is cached. The topology is built on CPU; the learned recurrence and
gradients run on GPU. Direct sampled `g(x), f(x)` values are the inputs;
neighbor-value columns and exact-solver labels are not used by this predictor.

The polynomial bound has a substantial constant. It is not a claim that the
same ordering budget is cheap at every bitness, or that sampled observations
determine the true answer under incomplete coverage. A finite ordering family
can depend on the names of coordinates; it remains invariant to point order
and input-bit complements. Averaging many orderings improves coverage.

### 6.2 Checkpoint transfer establishes the architectural effect

The complete-lattice checkpoint's learned combination and head weights load
directly into this architecture. No optimization or new labels were involved
in this first comparison:

| Orderings | Train depth | Train size | Validation depth | Validation size |
| ---: | ---: | ---: | ---: | ---: |
| 16 | — | — | 0.348189 | 0.069231 |
| 32 | 0.060870 | 0.052065 | 0.083619 | 0.064035 |
| 64 | 0.029133 | 0.051988 | 0.037716 | 0.063970 |

Training values here are frozen evaluations over all 65,536 pairs, not
averages while weights were changing. With 64 orderings, the TT/general depth
RMSE was 0.039192/0.012706, and size was 0.062970/0.037951.

At this small bitness and complete sampled coverage, 64 orderings happen to
cover the entire lattice: 19,683 nodes. With 32, there are 19,675 nodes.
Consequently this result does **not** establish that a much smaller fraction
of the lattice preserves accuracy at high bitness. What changes is the
construction's polynomial upper bound and its ability to accept sparse input
at high bitness. The comparison also explains why the root-ranking beam in
section 3.8 was not decisive: this method keeps a much richer union of query
orders and shares their child states.

### 6.3 Training from scratch

```bash
./scripts/train/train_model.sh m1 8
```

That configuration is now the default `conf/train.yaml`; the run recorded here
used the identical settings under the name `conf/train_ordered.yaml`. The run
directory is `/tmp/circus-ordered`. Adam uses learning rate
0.001, batch size 64, and gradient clipping at 1. The early-stop threshold now
applies to each head on each split, rather than only their combined RMSE.
Training stopped after five epochs (about 7 minutes 44 seconds); the best
validation checkpoint came from epoch 3. Frozen evaluations of that checkpoint:

| Evaluation | Depth RMSE | Size RMSE |
| --- | ---: | ---: |
| All 65,536 training pairs | **0.038191** | **0.078132** |
| Training tree-over-table pairs | 0.042946 | 0.051283 |
| Training independent tables | 0.032752 | 0.097873 |
| Validation, seed 239 | **0.041767** | **0.052526** |
| Validation, seed 240 | 0.041767 | 0.052526 |
| Validation, seed 241 | 0.041767 | 0.052526 |

Both heads meet 0.1 on training and validation. Size error is higher than the
certificate baseline's 0.0176 validation RMSE, while depth falls from 0.1379
to 0.0418. The three validation samplings agree because they cover the same
complete input domain at this bitness. The epoch-3 training-loop depth RMSE
was 0.1032 while weights were changing; the frozen training value above is the
measurement of the saved best checkpoint.

Weights: `/tmp/circus-ordered/m1_08.best.pt`. The complete schedule, frozen
evaluations, configuration, and SHA-256 hashes of the checkpoint, data files,
and predictor source are in [`ordered_m1_08_results.json`](ordered_m1_08_results.json).

### 6.4 Execution at 256 bits

Both formats now accept bitness 256. The existing one-byte header uses zero
to encode it, preserving the layout and encodings for 8–255. An M1 witness
can query internal coordinate 256, so its tree accepts 257 coordinates.
Deterministic generation, header round trips, and helper-coordinate
serialization are covered by tests. Tiny M1 and M2 files at 256 were generated
and served through the ordinary daemon.

On those cases, the integrated ordered predictor completed forward and
backward passes with finite gradients, using 512 sampled points, one ordering,
batch size one, and the default hidden widths:

| Model | Sampled query columns after compression | Nodes | Forward + backward | Peak GPU allocation |
| --- | ---: | ---: | ---: | ---: |
| M1, 256 bits | 27 | 966,614 | 5.91 s | 769 MiB |
| M2, 256 bits | 26 | 404,717 | 1.78 s | 421 MiB |

These are execution measurements on the RTX 5070, not accuracy results.
There is no trained 256-bit checkpoint or claim of RMSE below 0.1 there.
Sampling determines which columns can be distinguished; collapsing equal
sampled columns does not prove the underlying functions treat them equally.
Higher-bit training still requires the existing bootstrap dependency chain.

### 6.5 Cubic pooling attempts that did not reach the target

Before retaining explicit restrictions, three-coordinate conditional pools
were added on top of the frozen certificate backbone. A 100-epoch head fit
stayed near depth RMSE 0.137. Adding conditional neighbor disagreements and
an auxiliary loss on exact three-coordinate restriction targets likewise
stayed near 0.136. Those exact targets were offline training supervision only;
they were never model inputs and are not used in the ordered predictor.

A `cubic.py` predictor also supported joint training of the certificate backbone
with three top pooled levels. One full epoch from the certificate checkpoint reached
validation depth/size 0.162436/0.020945 and took 439 seconds. That run was
stopped after the sparse lattice met the target. Recomputing single-restriction
certificates with the old weights, without retraining, was worse still
(validation depth 0.589302). These results do not establish a lower bound on
what a better pooled architecture could learn.

Scratch experiment artifacts are under `/tmp/circus-cubic-experiments`.

## 7. Independent check of section 6, and the minimised tree (2026-09-06)

Everything in this section was measured after section 6, with its own
evaluation loop rather than the one that produced the numbers above.

### 7.1 The result reproduces

Loading `/tmp/circus-ordered/m1_08.best.pt` and scoring it independently:

| Evaluation | depth | size |
| --- | ---: | ---: |
| validation, sampling seed 239 | 0.041767 | 0.052526 |
| validation, sampling seed 241 | 0.041767 | 0.052526 |
| training epoch 1, first 8,192 pairs | 0.043956 | 0.051599 |
| training epoch 5, first 8,192 pairs | 0.043956 | 0.051599 |

Both scores are under 0.1 on both splits. Retraining from scratch after the
tree was minimised -- `./scripts/train/train_model.sh m1 8`, the ordered
configuration now being the default -- reproduced the run exactly: the same
per-epoch numbers, the stop on the per-head threshold at epoch 5, best
validation RMSE 0.047452, and a checkpoint **bit-identical** to the recorded
one (SHA-256 `8a14651f...`).

### 7.2 The ordering budget covers the whole lattice at 8 bits

Building the topology on the M1 extended cube at bitness 8 and counting nodes
against the complete `3^9 = 19,683`:

| Orderings | Nodes | Share of the lattice | Distinct fixed sets |
| ---: | ---: | ---: | ---: |
| 4 | 14,927 | 75.8% | 293 |
| 8 | 17,933 | 91.1% | 400 |
| 16 | 19,259 | 97.8% | 484 |
| 32 | 19,675 | 100.0% | 510 |
| 64 | **19,683** | **100.0%** | 511 |

So the shipped `orders: 64` retains *every* restriction at this bitness. The
accuracy in section 6.3 is the exhaustive recursion's, obtained through a
construction that obeys a polynomial bound -- not evidence that a cubic subset
of the lattice suffices.

Training with a genuinely sparse budget settles it. At `orders: 8`, which keeps
91% of the lattice, eight epochs reach validation depth **0.3715** and size
0.0342, against 0.0418 and 0.0525 at `orders: 64`:

| Epoch | 1 | 2 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation depth | 0.5326 | 0.5093 | 0.4700 | 0.4014 | 0.3715 |

Dropping 9% of the restrictions costs an order of magnitude on depth. What
carries the result at bitness 8 is completeness, and the cubic bound is a
property of the construction rather than the reason it scores well.

### 7.3 How the construction actually grows

Topology size for 64 orderings over 512 sampled cells, as the number of
distinct query coordinates rises:

| Coordinates | 8 | 12 | 16 | 20 | 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| distinct fixed sets | 255 | 3,708 | 20,173 | 53,278 | 77,822 |
| nodes | 6,503 | 390,617 | 4,542,544 | 6,175,934 | 6,105,750 |
| `3^d` | 6,561 | 531,441 | 43,046,721 | 3.5e9 | 2.8e11 |

The fixed-set count grows about cubically, and the node count saturates near
six million because the recursion stops at singleton groups: the sample bounds
it, not `3^d`. Nothing of size `2^d` or `3^d` is allocated at any point. Six
million nodes for one case is still a large constant, and the topology took
about 24 seconds to build on CPU at 24 coordinates.

### 7.4 The topology cache only hits when cases share an input set

The geometry is cached on the case's set of distinct sampled inputs, and only
the most recent one is kept. Among the first 64 validation cases:

| Bitness | Distinct input sets | Topologies built |
| ---: | ---: | ---: |
| 8 | 1 | 1 per 64 cases |
| 13 | 64 | 64 per 64 cases |

At bitness 8 the sample covers the cube, so every case shares one topology and
it is built once for the whole run -- which is why five epochs take under eight
minutes. Above that the sample no longer covers the cube, every case has its
own geometry, and the build cost returns per case. The five-epoch figure does
not carry to higher bitness.

### 7.5 What was removed

The tree now carries the ordered architecture and the earlier
`DeepSetPredictor`, and nothing else. Removed: the complete restriction lattice
and its configuration, the certificate architecture and its configuration, the
`cubic.py` pooling experiment and its configuration, the `size_weight` loss
knob no shipped configuration set, and the 256-bit offline-format work of
section 6.4, none of which is needed to put both scores under 0.1 at `m1 8`.
Sections 1-6 keep their measurements; the code behind the removed rows is in
the history, not the tree.
