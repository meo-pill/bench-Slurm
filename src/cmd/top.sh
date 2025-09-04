#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

TOP_MODE=${TOP_MODE:-unique}

check_deps top

if ! ls "$RES_DIR"/*.csv >/dev/null 2>&1; then
  echo "Aucun résultat trouvé dans $RES_DIR" >&2
  exit 1
fi

case "$TOP_MODE" in
  unique)
    echo "=== TOP Monothread (meilleur run par nœud) ==="
    awk -F, 'FNR==1{next} $2=="mono" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== TOP Multithread (meilleur run par nœud) ==="
    awk -F, 'FNR==1{next} $2=="multi" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    ;;
  unique-last)
    echo "=== TOP Monothread (dernier run par nœud) ==="
    for f in "$RES_DIR"/*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="mono"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== TOP Multithread (dernier run par nœud) ==="
    for f in "$RES_DIR"/*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="multi"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
    ;;
  top10)
    echo "=== TOP 10 Monothread (toutes runs) ==="
    awk -F, 'FNR==1{next} $2=="mono" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
    echo
    echo "=== TOP 10 Multithread (toutes runs) ==="
    awk -F, 'FNR==1{next} $2=="multi" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
    ;;
  by-node-mean)
    echo "=== Classement Monothread par moyenne de toutes les runs (par nœud) ==="
    awk -F, 'FNR==1{next} $2=="mono" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== Classement Multithread par moyenne de toutes les runs (par nœud) ==="
    awk -F, 'FNR==1{next} $2=="multi" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    ;;
  *)
    echo "TOP_MODE inconnu: $TOP_MODE" >&2; exit 1 ;;
esac
