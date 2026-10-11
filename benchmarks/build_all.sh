#!/usr/bin/env bash
# builds one bench binary per stage in parallel; logs to bin/build_<name>.log
cd "$(dirname "$0")"
mkdir -p bin
FLAGS="-O3 -arch=sm_86 -lineinfo -std=c++17"

build() {   # build <out-name> <extra nvcc args...>
    local name=$1; shift
    if cmd //c ".\\nvcc.bat $FLAGS $* bench.cu -o bin/$name.exe" > "bin/build_$name.log" 2>&1; then
        echo "OK   $name"
    else
        echo "FAIL $name (see bin/build_$name.log)"
    fi
}

for s in "${@:-1 2 3 4 5 6 7 8 9 91}"; do
    for st in $s; do build "bench_s$st" "-DSTAGE=$st" & done
done
[[ $# -eq 0 ]] && build bench_s8_odd -DSTAGE=8 -DODD_SHAPES &
wait
