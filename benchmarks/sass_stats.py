"""Static SASS instruction counts per kernel instantiation (cuobjdump), to explain profiler results.

usage: python sass_stats.py <stage> [filter]      e.g. python sass_stats.py 4 'Li4ELi128ELi32E'
Counts are static (instructions in the binary), not dynamic executions.
"""
import re
import subprocess
import sys
from collections import Counter

stage = sys.argv[1]
filt = sys.argv[2] if len(sys.argv) > 2 else ""
sass = subprocess.run(["cuobjdump", "-sass", f"bin/bench_s{stage}.exe"], capture_output=True, text=True).stdout

for block in sass.split("Function : ")[1:]:
    name = block.split("\n", 1)[0].strip()
    if filt not in name:
        continue
    ops = Counter()
    for line in block.splitlines():
        m = re.search(r"\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)", line)
        if m:
            ops[m.group(1)] += 1
    lds = {k: v for k, v in ops.items() if k.startswith("LDS")}
    pick = lambda pre: sum(v for k, v in ops.items() if k.split(".")[0] == pre)
    print(name)
    print(f"  LDS by width: {dict(sorted(lds.items()))}")
    print(f"  LDL={pick('LDL')} STL={pick('STL')} FFMA={pick('FFMA')} HMMA={pick('HMMA')} LDSM={pick('LDSM')} "
          f"LDGSTS={pick('LDGSTS')} SHFL={pick('SHFL')} MUFU={pick('MUFU')} BAR={pick('BAR')} total={sum(ops.values())}")
