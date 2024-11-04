# How to Conduct the MVP

Make sure you pip install -r requirements.txt

Basically run all of the cells in the Jupyter Notebook sequentially, making sure that you include your own OPENAI_API_KEY

From the cell where the chunks are made, copy the saved directory of the vector store.

Then, copy this path into server_librag.py

Run server_librag.py in your terminal and go to http://localhost:8000/chain/playground on your browser

# Data Needed

Make sure you download the the following files from the current paths on the SCC and change the needed directories in the Jupyter Notebook:

/projectnb/sparkgrp/ml-bpl-rag-data/ft_13_checkpoint_10_133.json

/projectnb/sparkgrp/ml-bpl-rag-data/full_data/bpl_data.json

These are data files that are needed for the code to even work
