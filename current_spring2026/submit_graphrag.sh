#!/bin/bash -l
#$ -P sparkgrp
#$ -l h_rt=12:00:00
#$ -N bpl-graph-build
#$ -j y
#$ -o graph-build.log
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -pe omp 8

source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/ml-bpl-rag-spring-2026/ml-bpl-rag/current_spring2026

echo "Starting graph build at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# python -m graph.graph_builder --all
#!/bin/bash -l
#$ -P sparkgrp
#$ -l h_rt=48:00:00
#$ -N bpl-graph-build
#$ -j y
#$ -o logs/graph-build.log
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -l gpu_c=7.5
#$ -pe omp 8

source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/ml-bpl-rag-spring-2026/ml-bpl-rag/current_spring2026

echo "Starting graph build at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# for year in 1900 1901; do
#     echo "Processing year: $year"
python -m graph.graph_builder --all
# done

echo "Graph build complete at $(date)"
