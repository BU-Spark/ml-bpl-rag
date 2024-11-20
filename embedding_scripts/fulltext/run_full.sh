#!/bin/bash -l


#$ -P ds549           # Assign to project ds549

#$ -l h_rt=2:00:00   # Set a hard runtime limit

#$ -N real_embed     # Give the job a name other than the shell script name

#$ -j y               # merge the error and regular output into a single file


source .venv/bin/activate

module load python3/3.12.4

echo "Print python version"

python --version
python3 --version

python3 embed.py 0 100
