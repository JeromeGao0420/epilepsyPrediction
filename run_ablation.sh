#!/bin/bash
cd /Users/gaojiaran.123/epilepsyPrediction
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate shortPaper
python generate_ablation_figure.py > /tmp/ablation_out.txt 2>&1
echo "exit: $?" >> /tmp/ablation_out.txt