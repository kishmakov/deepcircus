# Performance Inspection Plan

This plan is for auditing:

```bash
uv run scripts/run.py
```

The script creates `GeneratorProxy(16)`, which spawns the C++ daemon
`cpp/build/generator_server` as a child process. The daemon generates task
data eagerly on a thread pool, publishes each task's tensors through POSIX
shared memory, and the Python side (`src/generator.py`) borrows zero-copy
NumPy views over a socket protocol. Training (`src/train.py`) consumes one
`(iteration, bitness)` stage at a time, builds PyTorch tensors, and
trains/evaluates a DeepSet model.

There is no Python worker fleet anymore: all native generation lives in the
single `generator_server` process and its worker threads.

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

Note that shared-memory segments show up in both the daemon's and the Python
process's RSS; don't double-count them.

Repeat this before and after every meaningful change.

## 2. Profile Python Orchestration

Use `py-spy` first because it can observe Python without modifying code.
Only the main process is Python now, so no `--subprocesses` is needed.

```bash
py-spy top -- uv run scripts/run.py
py-spy record --rate 20 -o tmp/profile.svg -- uv run scripts/run.py
```

This should show whether time is going into:

- `Generator.next_task` / `_receive_exact` (blocked waiting on the daemon —
  a generation-side bottleneck, not a Python one)
- `Sampler.take_stage` (tensor construction, `torch.cat`)
- `Sampler._approximate_targets` (inference over restriction chunks)
- `predict_values`
- `train_epoch`
- `evaluate_epoch`

Time spent inside socket reads in `next_task` means training outpaces the
daemon; time spent everywhere else means the daemon's eager generation keeps
up and Python/PyTorch is the bottleneck.

## 3. Profile Native C++ Work

Generation runs inside the `generator_server` child process, so profile that
process, not the Python driver. The record/report flow is scripted:

```bash
scripts/perf_binary_run.sh [duration-seconds]   # default 30
scripts/perf_binary_inspect.sh [perf-report-args]
```

`perf_binary_run.sh` builds a `cpp/build-perf` tree with debug symbols and
frame pointers (the Release `cpp/build` stays untouched), launches
`scripts/run.py` against it via `GENERATOR_SERVER`, waits until the daemon
starts burning CPU, records DWARF call graphs into `/tmp/circus/perf.data`,
and tears the run down. `perf_binary_inspect.sh` opens that recording in
`perf report`.

One-time prerequisite: this machine ships with
`kernel.perf_event_paranoid = 4`, which blocks perf entirely for
unprivileged users. Allow self-profiling with:

```bash
sudo sysctl kernel.perf_event_paranoid=1
# optional, resolves kernel symbols in call graphs:
sudo sysctl kernel.kptr_restrict=0
# persist across reboots:
echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/60-perf.conf
```

Look especially for hot functions in:

- `cpp/generator/tree.cpp` (tree building, evaluation, exact solving)
- `cpp/generator/dataset.cpp` (case-ID sampling, data/restriction handles)
- `cpp/generator/utils.cpp` (value-input generation, `FlippingSampler`)
- `cpp/producer/task_queue.cpp` (chunking, `Concat` merges)
- `cpp/server/daemon.cpp` (socket writes, shared-memory publication)

For ad-hoc checks against an already-running daemon (`pgrep` needs `-f`
because the kernel truncates process names to 15 characters):

```bash
perf top -p "$(pgrep -n -f generator_server)"                # live view
perf stat -p "$(pgrep -n -f generator_server)" -- sleep 30   # hw counters
```

## 4. Inspect Daemon Concurrency

`cpp/producer` produces coordinates strictly sequentially, chunking each
coordinate's case ids across the thread pool and merging the chunks. Audit:

- whether all worker threads are busy during generation
- whether the sequential merge (`gen::Values::Concat` /
  `gen::Restrictions::Concat`) serializes a meaningful share of the time
- whether the daemon idles once it has generated ahead of training
  (fine), or training stalls in `next_task` waiting on it (not fine)

`perf` only shows on-CPU time, so cross-check with per-thread utilization:

```bash
pidstat -t -p "$(pgrep -n -f generator_server)" 1
htop   # tree view: python + generator_server threads
```

If per-thread CPU% is well below 100 per worker while training waits, the
bottleneck is queueing/merging, not the code `perf report` shows.

## 5. Profile PyTorch Training

Use the PyTorch profiler around `train_epoch`, `evaluate_epoch`, and the
recursive-target inference path (`Sampler._approximate_targets` →
`predict_values`).

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

If the GPU is mostly idle while the run is slow, focus on generation,
shared-memory materialization, NumPy-to-tensor conversion, and
host-to-device transfer.

For a deeper timeline:

```bash
nsys profile -o tmp/nsys_run uv run scripts/run.py
```

## 7. Inspect Memory Allocation and Copies

Task arrays arrive as zero-copy views into shared memory, but every
`torch.tensor(...)` call in `Sampler` copies them out. For mixed
Python/native time and memory pressure:

```bash
scalene scripts/run.py
```

For Python allocation flamegraphs:

```bash
memray run -o tmp/memray.bin scripts/run.py
memray flamegraph tmp/memray.bin
```

Pay attention to:

- `torch.tensor(task.train_values)` / `torch.cat` copies in
  `Sampler.take_stage` and `_take_validation_datasets`
- the `chunk.reshape(...)` and `np.empty` in `_approximate_targets`
- `torch.as_tensor(..., device=DEVICE)` transfers inside `predict_values`
- `.cpu().numpy()` on the way back from inference
- shared-memory segment lifetime (each task's segment must die at
  `task.release()`; leaked segments show up in `/dev/shm`)

## 8. Add Lightweight Phase Timers

Profilers are useful, but this project also needs experiment-aware timers.
Add timing around these boundaries:

- `Generator.next_task` (time blocked = daemon behind training)
- `Sampler.take_stage` (excluding `next_task`: tensor build cost)
- `Sampler._approximate_targets`
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

The restrictions path is a prime suspect because it combines C++ generation,
shared-memory publication, chunked model inference in
`_approximate_targets`, and CPU-to-GPU transfer.

## 9. First Recommended Audit Run

Run these in order:

```bash
/usr/bin/time -v uv run scripts/run.py
py-spy record --rate 20 -o tmp/profile.svg -- uv run scripts/run.py
scripts/perf_binary_run.sh 30
```

The py-spy flamegraph decides where to look next: if the main process is
blocked in `next_task`, go deep on sections 3–4 (daemon); otherwise go deep
on sections 5–7 (training and copies).

After that, add phase timers to the hottest code paths and repeat the
baseline.
