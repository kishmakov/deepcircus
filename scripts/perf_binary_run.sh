#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/cpp/build-perf"
PERF_DATA="/tmp/circus/perf.data"
DURATION="${1:-30}"

PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
if [ "$PARANOID" -gt 2 ]; then
    echo "kernel.perf_event_paranoid=$PARANOID blocks profiling for unprivileged users" >&2
    echo "fix once with: sudo sysctl kernel.perf_event_paranoid=1" >&2
    exit 1
fi

if pgrep -f "scripts/run.py" >/dev/null; then
    echo "another scripts/run.py is already running; it would clash on /tmp/circus and shared memory" >&2
    exit 1
fi

cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -G Ninja \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEEPCIRCUS_BUILD_TESTS=OFF \
    -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -g -fno-omit-frame-pointer" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -g -fno-omit-frame-pointer"
cmake --build "$BUILD_DIR" --config Release --target generator_server

mkdir -p "$(dirname "$PERF_DATA")"

SERVER_PID=""
GENERATOR_SERVER="$BUILD_DIR/generator_server" uv run "$ROOT/scripts/run.py" &
RUN_PID=$!
trap 'kill "$RUN_PID" $SERVER_PID 2>/dev/null || true' EXIT

until SERVER_PID="$(pgrep -n -f "$BUILD_DIR/generator_server")"; do
    if ! kill -0 "$RUN_PID" 2>/dev/null; then
        echo "run.py exited before the generator daemon appeared" >&2
        exit 1
    fi
    sleep 0.2
done

# Wait until the daemon starts generating: the client spends a while importing
# torch and loading the snapshot before it sends the initialize command, and
# recording that idle stretch would produce an empty profile.
ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null; }
START_TICKS="$(ticks "$SERVER_PID")"
while :; do
    NOW="$(ticks "$SERVER_PID")" || true
    if [ -z "$NOW" ]; then
        echo "generator daemon exited before generation started" >&2
        exit 1
    fi
    [ "$NOW" -ge $((START_TICKS + 20)) ] && break
    sleep 0.2
done

perf record -g --call-graph dwarf -p "$SERVER_PID" -o "$PERF_DATA" -- sleep "$DURATION"
echo "recorded ${DURATION}s of generator_server into $PERF_DATA"
echo "inspect with: scripts/perf_binary_inspect.sh"
