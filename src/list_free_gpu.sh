#!/bin/bash
# Script : list_free_gpus.sh
# Objectif : lister les nœuds Slurm dont tous les GPU sont libres

echo "Noeud(s) avec tous les GPU libres :"

# On boucle sur tous les nœuds connus de Slurm
for node in $(sinfo -h -N -o "%N"); do
    # Vérifie si le nœud a des GPU configurés
    if scontrol show node "$node" | grep -q "Gres=gpu"; then
        # Récupère nombre total et nombre utilisés
        total=$(scontrol show node "$node" | awk -F= '/CfgTRES/{print $2}' | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)
        used=$(scontrol show node "$node" | awk -F= '/AllocTRES/{print $2}' | grep -o "gres/gpu=[0-9]*" | cut -d= -f2)

        # Si aucun GPU n'est utilisé et qu'il y en a au moins 1 → on affiche le nœud
        if [[ -n "$total" && "$total" -gt 0 && ( -z "$used" || "$used" -eq 0 ) ]]; then
            echo "$node (GPU total: $total, utilisés: 0)"
        fi
    fi
done
