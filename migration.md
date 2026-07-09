Yes. I’d do it as a staged API migration, not a rewrite. Keep Python/PyTorch as the training owner, and gradually make `bool-bench` own “sample production”.

**Current expensive boundary**

Right now Python does:

- sample case IDs in `src/sampler.py`
- generate input bit strings
- route work through Python multiprocessing in `src/generator_proxy.py`
- call scalar-ish C APIs repeatedly
- receive ASCII `0/1` strings
- convert those strings to `float32` NumPy tensors

With `train_samples=65536` and `points_per_sample=128`, that is a lot of Python string/list/process overhead.

**Recommended transition**

1. **Add batch C APIs, but keep Python-generated inputs first**

   First add low-risk functions in `bool-bench` that accept many case IDs and packed inputs, then fill a caller-owned `float*`.

   Example shape:

   ```cpp
   void bb_tree_value_tensor(
       uint16_t bitness,
       const size_t* case_ids,
       size_t cases,
       const char* packed_inputs,
       size_t reps,
       float* out
   );
   ```

   Output shape:

   ```text
   cases x reps x (2 * bitness + 1)
   ```

   This removes the biggest per-sample overhead: repeated ctypes calls, ASCII return buffers, and Python-side ASCII-to-float conversion.

2. **Change Python wrapper to pass NumPy output buffers**

   In `bool-bench/bool_bench.py` or `src/generator_proxy.py`, allocate:

   ```python
   x = np.empty((cases, reps, sample_point_dim(bitness)), dtype=np.float32)
   ```

   Then pass `x.ctypes.data` into C++.

   This avoids extra copies. Python still controls the training loop and tensor ownership.

3. **Move input generation into C++**

   Once the batch tensor API works, change it so Python passes only:

   ```text
   bitness, case_ids, cases, reps, seed, input_policy
   ```

   C++ generates the same two-part input scheme currently in `Sampler._input_bits`:

   - half block-inversion inputs
   - half random inputs

   I would make this explicit:

   ```cpp
   enum bb_input_policy {
       BB_INPUT_BLOCK_AND_RANDOM = 0,
       BB_INPUT_RANDOM_ONLY = 1,
   };
   ```

   Use a deterministic C++ PRNG you own, not `std::uniform_int_distribution` if you care about exact cross-compiler reproducibility. A tiny `splitmix64`-based helper is enough.

4. **Move depth/node batch queries into C++**

   Add:

   ```cpp
   void bb_tree_depth_tensor(
       uint16_t bitness,
       const size_t* case_ids,
       size_t cases,
       float* out
   );

   void bb_table_depth_tensor(...);
   ```

   Then replace `tree_depth_tensors` / `table_depth_tensors`.

   This is simple and gives you another correctness checkpoint.

5. **Move ID selection into C++**

   After tensor generation is solid, move the train/val case selection rules from `src/sampler.py` into C++.

   Python should ask for logical datasets:

   ```text
   split = train/val
   bitness
   iteration
   seed
   train_samples / validation_samples
   reps
   ```

   C++ decides:

   - table vs tree split
   - solvable vs recursive table split
   - sampled case IDs

   Use deterministic sampling without replacement. For huge ranges like `2^32`, use Floyd’s algorithm or another sparse algorithm rather than allocating a full range.

6. **Keep recursive target inference split across the boundary**

   This part should stay hybrid for now:

   ```python
   x_restricted = generator.table_recursive_restriction_tensor(...)
   predictions = predict_values(previous_model, x_restricted, ...)
   y = min/max reduction over predictions
   ```

   C++ can generate the restriction tensor, but Python should still run the model. Do not make C++ call PyTorch yet; that would make the migration much messier.

7. **Replace Python multiprocessing last**

   Only after batch APIs are correct, move process/thread handling into C++.

   Important: current tree code has global cache state like `g_decision_trees`, so naive C++ threads would need locking or per-worker state. I’d prefer introducing a C++ sampler/context object:

   ```cpp
   struct BbSampler;

   BbSampler* bb_sampler_create(size_t workers);
   void bb_sampler_destroy(BbSampler*);
   void bb_sampler_fill_train_values(...);
   ```

   Internally each worker can own its own cache/context. That mirrors your current “persistent worker keeps cache warm” behavior without Python multiprocessing.

8. **Collapse Python classes**

   At the end:

   - `GeneratorProxy` becomes a thin ctypes handle wrapper
   - `Sampler` mostly asks C++ for train/val tensors
   - `train.py` stays almost unchanged

   That is the right final boundary: C++ produces datasets, Python trains models.

**Order I’d actually implement**

1. `bb_*_value_tensor` batch APIs with Python-provided inputs.
2. Python wrapper uses caller-owned NumPy buffers.
3. C++ input generation with seed/reps/policy.
4. C++ restriction tensor generation.
5. C++ case ID selection.
6. C++ persistent worker/context object.
7. Delete most of `src/generator_proxy.py`.

This gives you useful speedups early, while each step is testable against the current implementation. The key is to avoid jumping directly to “C++ owns everything”; the recursive-model target path makes that boundary naturally hybrid for now.
