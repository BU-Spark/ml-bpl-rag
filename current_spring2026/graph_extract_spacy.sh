#!/bin/bash -l
#$ -P sparkgrp
#$ -l h_rt=12:00:00
#$ -N graph-metadata-extract
#$ -j y
#$ -o logs/graph-metadata-extract.log
#$ -pe omp 8

module load miniconda
conda activate spark-rag

cd /projectnb/sparkgrp/ml-bpl-rag-data-subset/temp/ml-bpl-rag/current_spring2026

mkdir -p logs data/graph

echo "Starting metadata entity extraction at $(date)"

python -m graph.extract_entities_spacy --metadata-only --workers 8

echo "Done at $(date)"