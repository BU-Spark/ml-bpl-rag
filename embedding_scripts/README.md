Each directory (metadata and fulltext) contains the respective python script and corresponding shell script used for loading in an [i:n) chunk of the either the fulltext or metadata.

Be sure to change the directories within the python scripts to be those where your un-embedded data is located and where you would like to store your embeddings.

Each python script is called in the shell script as "python (scriptname).py <i> <n>" Where i and n are the index boundaries of the chunk you would like to embed. By making n = -1, the script will embed documents from i to the very end of the corpus of data.
