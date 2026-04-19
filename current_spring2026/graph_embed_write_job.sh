#!/bin/bash -l
#$ -P ivc-ml
#$ -l h_rt=8:00:00
#$ -N bpl-graph-embed-write
#$ -j y
#$ -o logs/graph-embed-write.log
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -pe omp 4

source /etc/profile.d/modules.sh
module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/ml-bpl-rag-spring-2026/ml-bpl-rag/current_spring2026

echo "Starting embed + write at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 3. Embed GPT-4o-mini entities
python -m graph.embed_entities --suffix all_gpt

# 4. Write to Neo4j
python -m graph.write_graph --suffix all_gpt

echo "Embed + write complete at $(date)"
