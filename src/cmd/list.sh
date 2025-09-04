#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

check_deps list

echo "=== Nœuds du cluster et nombre de runs enregistrés ==="
mapfile -t NODES < <(sinfo -h -N -o '%N')
for NODE in "${NODES[@]}"; do
  f="$RES_DIR/$NODE.csv"
  if [[ -f "$f" ]]; then
    runs=$(awk -F, 'FNR==1{next} $2=="mono"{c++} END{print c+0}' "$f")
  else
    runs=0
  fi
  printf "%s %d\n" "$NODE" "$runs"
done | sort -s -k1,1
