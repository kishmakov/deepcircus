# Performance Inspection Plan

This plan is for auditing:

```bash
uv run scripts/run.py
```

The script creates `GeneratorProxy(16)`, calls `run_training`, uses a persistent
process-backed worker fleet, calls C++ through `libbb.so` via ctypes, builds
NumPy/PyTorch tensors, and trains/evaluates a PyTorch model.

## 1. Establish a Baseline

Start with a plain wall-clock and resource baseline.

```bash
/usr/bin/time -v uv run scripts/run.py
```

Look at:

- elapsed time
- max resident set size
- CPU percentage
- voluntary and involuntary context switches
- major/minor page faults

Repeat this before and after every meaningful change.

## 2. Profile Python Orchestration

Use `py-spy` first because it can observe Python without modifying code.

```bash
py-spy top --subprocesses -- uv run scripts/run.py
py-spy record --rate 20 --subprocesses -o tmp/profile.svg -- uv run scripts/run.py
```

Use `--subprocesses` because `src/generator_proxy.py` creates worker processes.

This should show whether time is going into:

- `Sampler._set_train`
- `Sampler._approximate_recursive_targets`
- `GeneratorProxy._dispatch`
- result collection from worker pools
- `predict_values`
- `train_epoch`
- `evaluate_epoch`

## 3. Profile Native C++ Work

The ctypes calls into `libbb.so` are native work, so use `perf`.

```bash
perf record -g --call-graph dwarf -- uv run scripts/run.py
perf report
```

If the C++ stack traces are poor, rebuild `bool-bench` with debug symbols and
frame pointers while keeping optimization enabled:

```bash
CXXFLAGS="-O3 -g -fno-omit-frame-pointer" bool-bench/build.sh
```

Then rerun `perf`.

Look especially for hot functions in:

- `bool-bench/decision_tree.cpp`
- `bool-bench/small_bitness.cpp`
- `bool-bench/bool_bench.cpp`

## 4. Inspect Multiprocessing Behavior

`GeneratorProxy` routes each `(bitness, case_id)` pair to a fixed worker. Audit:

- whether all 16 workers are busy
- whether buckets are balanced
- whether parent process waits on slow workers
- whether result payloads are too large to pickle/copy cheaply
- whether `apply_async(...).get()` causes serialization bottlenecks

Useful commands:

```bash
htop
pidstat -dur -p ALL 1
```

For process-level detail during a run:

```bash
ps -o pid,ppid,pcpu,pmem,rss,stat,comm,args -C python
```

## 5. Profile PyTorch Training

Use the PyTorch profiler around `train_epoch`, `evaluate_epoch`, and the
recursive-target inference path in `predict_values`.

Capture CPU, CUDA, shapes, and memory:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("tmp/profiler"),
) as profiler:
    ...
```

Then inspect with:

```bash
tensorboard --logdir tmp/profiler
```

Look for:

- low GPU utilization
- frequent small CPU-to-GPU transfers
- tensor conversion overhead from NumPy
- time in `BatchNorm1d`, `Linear`, `Dropout`, and pooling
- excessive evaluation time relative to training time

## 6. Monitor GPU Utilization

Quick live checks:

```bash
nvidia-smi dmon
nvidia-smi pmon
```

If the GPU is mostly idle while the run is slow, focus on generation, IPC,
NumPy allocation, and host-to-device transfer.

For a deeper timeline:

```bash
nsys profile -o tmp/nsys_run uv run scripts/run.py
```

## 7. Inspect Memory Allocation and Copies

For mixed Python/native time and memory pressure:

```bash
scalene scripts/run.py
```

For Python allocation flamegraphs:

```bash
memray run -o tmp/memray.bin scripts/run.py
memray flamegraph tmp/memray.bin
```

Pay attention to:

- large `np.empty` allocations in `GeneratorProxy`
- concatenation in `Sampler._set_train`
- copies from worker results into parent arrays
- `torch.as_tensor(..., device=DEVICE)` transfers
- `.cpu().numpy()` in `predict_values`

## 8. Add Lightweight Phase Timers

Profilers are useful, but this project also needs experiment-aware timers.
Add timing around these boundaries:

- `sampler.val_loader(bitness)`
- `sampler.reset_train(bitness, iteration, previous_model)`
- `Sampler._approximate_recursive_targets`
- `GeneratorProxy._dispatch`
- `_worker`
- `GeneratorProxy.restrictions_tensors`
- `predict_values`
- `train_epoch`
- `evaluate_epoch`
- checkpoint save

Log at least:

- iteration
- bitness
- operation name
- number of cases
- input/output tensor shape
- elapsed seconds

The `table_restrictions` path is a prime suspect because it combines C++
generation, multiprocessing IPC, NumPy allocation, CPU-to-GPU transfer, and
model inference.

## 9. First Recommended Audit Run

Run these in order:

```bash
/usr/bin/time -v uv run scripts/run.py
py-spy record --subprocesses -o tmp/profile.svg -- uv run scripts/run.py
perf record -g --call-graph dwarf -- uv run scripts/run.py
```

After that, add phase timers to the hottest code paths and repeat the baseline.

