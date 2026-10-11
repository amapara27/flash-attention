#!/usr/bin/env bash
# Builds benchmarks/fixtures/: copies Q/K/V from ../matrices (written by reference/dump_trace.py) and computes
# fp64 references with make_refs. Copying isolates the benchmark from dump_trace.py, which wipes ../matrices.
cd "$(dirname "$0")"
set -e

copy() {   # copy <fixture> <source case in ../matrices> <B N H D_K D_V>
    local dst=fixtures/$1 src=../matrices/$2
    [[ -f $src/Q.f32 ]] || { echo "missing $src - run 'python reference/dump_trace.py' first"; exit 1; }
    mkdir -p "$dst"
    cp "$src"/{Q,K,V}.f32 "$dst"/
    cp "$src/O.f32" "$dst/O_original.f32"   # only used to validate make_refs against the PyTorch reference
    echo "$3" > "$dst/shape.txt"
}

copy single_4096 massive_causal       "1 4096 1 64 64"
copy batch4_4096 batch_massive_causal "1 4096 4 64 64"
copy large_odd   large_odd            "1 517 1 64 48"
copy batch_odd   batch_odd            "2 517 3 64 48"

mkdir -p bin
if [[ ! -f bin/make_refs.exe ]]; then
    cmd //c ".\\nvcc.bat -O3 -arch=sm_86 -Xcompiler /openmp,/O2 make_refs.cu -o bin/make_refs.exe" > bin/build_make_refs.log 2>&1 \
        || { echo "make_refs build failed - see bin/build_make_refs.log"; exit 1; }
fi
./bin/make_refs.exe fixtures/single_4096 fixtures/batch4_4096 fixtures/large_odd fixtures/batch_odd
