#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/bench_common.sh"

TOP_MODE=${TOP_MODE:-unique}

check_deps top

if ! ls "$RES_DIR"/cpu_*.csv >/dev/null 2>&1; then
  echo "Aucun résultat trouvé dans $RES_DIR" >&2
  exit 1
fi

has_gpu_csv=0
ls "$RES_DIR"/gpu_*.csv >/dev/null 2>&1 && has_gpu_csv=1 || true

case "$TOP_MODE" in
  unique)
    echo "=== TOP Monothread (meilleur run par nœud) ==="
  awk -F, 'FNR==1{next} $2=="mono" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== TOP Multithread (meilleur run par nœud) ==="
  awk -F, 'FNR==1{next} $2=="multi" {k=$1; a=$6; s=$7; if(!(k in max)||a>max[k]){max[k]=a; std[k]=s}} END{for(k in max) printf "%s %.3f ± %.3f\n", k, max[k], std[k]}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    if (( has_gpu_csv == 1 )); then
      echo
      echo "=== TOP GPU Mono (moyenne des backends, meilleur run par nœud) ==="
      awk -F, '
      FNR==1{
        delete monoAvg; delete monoStd; delete multiAvg; delete multiStd;
        for(i=1;i<=NF;i++){
          if($i ~ /_mono_avg$/) monoAvg[i]=1;
          else if($i ~ /_mono_std$/) monoStd[i]=1;
          else if($i ~ /_multi_avg$/) multiAvg[i]=1;
          else if($i ~ /_multi_std$/) multiStd[i]=1;
        }
        next
      }
      {
        node=$1;
        sum=cnt=0; for(i in monoAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt}else{avg=""}
        sum=cnt=0; for(i in monoStd){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){std=sum/cnt}else{std=""}
        if(avg!="" && (!(node in maxM) || avg>maxM[node])){maxM[node]=avg; stdM[node]=std}
      }
      END{for(n in maxM) printf "%s %.3f ± %.3f\n", n, maxM[n], (stdM[n]==""?0:stdM[n])}
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '

      echo
      echo "=== TOP GPU Multi (moyenne des backends, meilleur run par nœud) ==="
      awk -F, '
      FNR==1{
        delete monoAvg; delete monoStd; delete multiAvg; delete multiStd;
        for(i=1;i<=NF;i++){
          if($i ~ /_mono_avg$/) monoAvg[i]=1;
          else if($i ~ /_mono_std$/) monoStd[i]=1;
          else if($i ~ /_multi_avg$/) multiAvg[i]=1;
          else if($i ~ /_multi_std$/) multiStd[i]=1;
        }
        next
      }
      {
        node=$1;
        sum=cnt=0; for(i in multiAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt}else{avg=""}
        sum=cnt=0; for(i in multiStd){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){std=sum/cnt}else{std=""}
        if(avg!="" && (!(node in maxM) || avg>maxM[node])){maxM[node]=avg; stdM[node]=std}
      }
      END{for(n in maxM) printf "%s %.3f ± %.3f\n", n, maxM[n], (stdM[n]==""?0:stdM[n])}
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    fi
    ;;
  unique-last)
    echo "=== TOP Monothread (dernier run par nœud) ==="
  for f in "$RES_DIR"/cpu_*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="mono"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== TOP Multithread (dernier run par nœud) ==="
  for f in "$RES_DIR"/cpu_*.csv; do n=$(basename "$f" .csv); awk -F, -v n="$n" 'FNR==1{next} $2=="multi"{a=$6;s=$7} END{if(a!="") printf "%s %.3f ± %.3f\n", n, a, s}' "$f"; done | sort -s -k2,2nr | nl -w2 -s'. '
    if (( has_gpu_csv == 1 )); then
      echo
      echo "=== TOP GPU Mono (moyenne des backends, dernier run par nœud) ==="
  for f in "$RES_DIR"/gpu_*.csv; do n=$(basename "$f" .csv);
        awk -F, -v n="$n" '
          FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
          {
            sum=cnt=0; for(i in monoAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0) lma=sum/cnt; else lma="";
            sum=cnt=0; for(i in monoStd){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0) lms=sum/cnt; else lms="";
            lastMono=lma; lastMonoStd=lms
          }
          END{ if(lastMono!="") printf "%s %.3f ± %.3f\n", n, lastMono, (lastMonoStd==""?0:lastMonoStd) }
        ' "$f"
      done | sort -s -k2,2nr | nl -w2 -s'. '

      echo
      echo "=== TOP GPU Multi (moyenne des backends, dernier run par nœud) ==="
  for f in "$RES_DIR"/gpu_*.csv; do n=$(basename "$f" .csv);
        awk -F, -v n="$n" '
          FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
          {
            sum=cnt=0; for(i in multiAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0) lma=sum/cnt; else lma="";
            sum=cnt=0; for(i in multiStd){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0) lms=sum/cnt; else lms="";
            lastMulti=lma; lastMultiStd=lms
          }
          END{ if(lastMulti!="") printf "%s %.3f ± %.3f\n", n, lastMulti, (lastMultiStd==""?0:lastMultiStd) }
        ' "$f"
      done | sort -s -k2,2nr | nl -w2 -s'. '
    fi
    ;;
  top10)
    echo "=== TOP 10 Monothread (toutes runs) ==="
  awk -F, 'FNR==1{next} $2=="mono" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
    echo
    echo "=== TOP 10 Multithread (toutes runs) ==="
  awk -F, 'FNR==1{next} $2=="multi" {printf "%s %.3f ± %.3f\n", $1, $6, $7}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
    if (( has_gpu_csv == 1 )); then
      echo
      echo "=== TOP 10 GPU Mono (moyenne des backends, toutes runs) ==="
      awk -F, '
        FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
        {
          sum=cnt=0; for(i in monoAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt; } else next;
          sum=cnt=0; for(i in monoStd){v=$i; if(v!=""){sum+=v; cnt++}}; std=(cnt>0)?sum/cnt:0;
          printf "%s %.3f ± %.3f\n", $1, avg, std
        }
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '

      echo
      echo "=== TOP 10 GPU Multi (moyenne des backends, toutes runs) ==="
      awk -F, '
        FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
        {
          sum=cnt=0; for(i in multiAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt; } else next;
          sum=cnt=0; for(i in multiStd){v=$i; if(v!=""){sum+=v; cnt++}}; std=(cnt>0)?sum/cnt:0;
          printf "%s %.3f ± %.3f\n", $1, avg, std
        }
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | head -10 | nl -w2 -s'. '
    fi
    ;;
  by-node-mean)
    echo "=== Classement Monothread par moyenne de toutes les runs (par nœud) ==="
  awk -F, 'FNR==1{next} $2=="mono" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    echo
    echo "=== Classement Multithread par moyenne de toutes les runs (par nœud) ==="
  awk -F, 'FNR==1{next} $2=="multi" {k=$1; sum[k]+=$6; ss[k]+=$6*$6; n[k]++} END{for(k in n){m=sum[k]/n[k]; v=(ss[k]/n[k])-m*m; if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)}}' "$RES_DIR"/cpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    if (( has_gpu_csv == 1 )); then
      echo
      echo "=== Classement GPU Mono (moyenne des backends, moyenne sur toutes les runs par nœud) ==="
      awk -F, '
        FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
        {
          k=$1;
          sum=cnt=0; for(i in monoAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt}else next;
          sumMono[k]+=avg; ssMono[k]+=avg*avg; nMono[k]++
        }
        END{ for(k in nMono){ m=sumMono[k]/nMono[k]; v=(ssMono[k]/nMono[k])-(m*m); if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)} }
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '

      echo
      echo "=== Classement GPU Multi (moyenne des backends, moyenne sur toutes les runs par nœud) ==="
      awk -F, '
        FNR==1{delete monoAvg; delete monoStd; delete multiAvg; delete multiStd; for(i=1;i<=NF;i++){if($i~/_mono_avg$/) monoAvg[i]=1; else if($i~/_mono_std$/) monoStd[i]=1; else if($i~/_multi_avg$/) multiAvg[i]=1; else if($i~/_multi_std$/) multiStd[i]=1;} next}
        {
          k=$1;
          sum=cnt=0; for(i in multiAvg){v=$i; if(v!=""){sum+=v; cnt++}}; if(cnt>0){avg=sum/cnt}else next;
          sumMul[k]+=avg; ssMul[k]+=avg*avg; nMul[k]++
        }
        END{ for(k in nMul){ m=sumMul[k]/nMul[k]; v=(ssMul[k]/nMul[k])-(m*m); if(v<0)v=0; printf "%s %.3f ± %.3f\n", k, m, sqrt(v)} }
  ' "$RES_DIR"/gpu_*.csv | sort -s -k2,2nr | nl -w2 -s'. '
    fi
    ;;
  *)
    echo "TOP_MODE inconnu: $TOP_MODE" >&2; exit 1 ;;
esac
