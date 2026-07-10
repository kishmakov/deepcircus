# C++ dataset-generation migration

The generator boundary is now C++-owned and pipelined. Python/PyTorch still
owns model inference and training, but it no longer selects case IDs, generates
input bits, routes multiprocessing workers, or assembles table/tree tensors.

## Implemented architecture

1. **C++-owned typed data**

   `bb_tree_value_tensor` and `bb_table_value_tensor` enqueue typed batches and
   return opaque `bb_data` handles. Each handle owns its sampled case IDs, case
   representations, values, exact targets, and case-local caches.

2. **Deterministic sampling and inputs**

   C++ samples IDs without replacement using Floyd's sparse algorithm and a
   SplitMix64 generator. Value inputs retain the half block-inversion, half
   random scheme as a private implementation detail.

3. **Transferred zero-copy views**

   Python calls `bb_data_acquire` and takes ownership of the C++ buffers as
   NumPy/PyTorch views. NumPy finalizers delete transferred buffers when the last
   view dies. Exact case representations are dropped immediately after handoff;
   exact targets are generated together with values as `bitness - depth`.

4. **Recursive table targets**

   Recursive targets remain hybrid. C++ owns the table case source while Python
   asks for restrictions, but each generated restriction buffer transfers to
   NumPy immediately. Python runs the previous model, owns the reduced target
   array, and releases the recursive source after the final chunk.

5. **C++ worker pool**

   A persistent `bb_generator` owns the worker threads. Different cases are
   generated concurrently while the Python caller remains the single frontend
   thread. The old Python multiprocessing fleet and case routing are removed.

6. **One-stage look-ahead**

   After Python acquires the current bitness, it requests the next stage before
   training. C++ therefore prepares the next bitness concurrently. Recursive
   restriction generation similarly keeps the current and next chunk in flight.

7. **Bounded lifetime**

   At steady state the generator prepares the current and prefetched-next
   bitness. Transferred output buffers are owned only by Python views; released
   recursive sources drop their representations and caches. No global generation
   cache retains cases across completed stages.

## Python responsibilities

Python continues to:

- choose table/tree batch counts according to bitness;
- request typed train and validation data;
- run recursive-target model inference;
- build loaders over borrowed tensors;
- train, validate, checkpoint, and explicitly release each stage.
