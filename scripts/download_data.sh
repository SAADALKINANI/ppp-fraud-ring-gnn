#!/usr/bin/env bash
# Real-data benchmark: Amazon fraudulent-reviewer graph as released with CARE-GNN (Dou et al., CIKM 2020)
# https://github.com/YingtongDou/CARE-GNN  (the PPP data itself is simulated by src/pppgraph/simulate.py)
set -euo pipefail
mkdir -p data/raw && cd data/raw
curl -sSfLO https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip
unzip -o Amazon.zip && ls -la
