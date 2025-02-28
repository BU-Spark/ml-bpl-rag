#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
import nest_asyncio
import uvicorn
import numpy as np

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.runnables import RunnablePassthrough

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content + "URL: " + str(doc.metadata["URL"]) for doc in docs)

def sources(docs):
    return "\n\n".join(doc.metadata["URL"] for doc in docs)

vectorstore = Chroma(persist_directory="./chroma_path", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#print(retriever.invoke("What is LangChain?"))

# 1. Create prompt template
system_template = """Answer the question based only on the following context:

{context}
Question: {question}
Helpful response:
Then, list the url's that come after 'URL' and say 'Here are the sources:'
"""

prompt_template = PromptTemplate.from_template(system_template)

user_template = ChatPromptTemplate.from_messages([
    ('user', '{query}'),
])

#query_prompt(user_template)
#q = user_template[0].prompt.template
#print(q)
#print(query_prompt(q))


# 2. Create model
model = ChatOpenAI(model="gpt-3.5-turbo")


# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | model #| parser

#print(chain.invoke("Who did Z. B. Oakes receive a letter from?"))
#print(model.invoke("Tell me about Barnstable Public Schools"))
# 5. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
