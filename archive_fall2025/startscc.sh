# copy and paste this into your terminal
module load miniconda 
conda activate ./env
cd ml-bpl-rag

# useful commands 
cd current_fall2025/scripts
pip install -r requirements.txt
git checkout dev 
git pull 
git checkout <your-branch-name>