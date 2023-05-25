#!/usr/bin/env python
# coding: utf-8

# In[67]:


from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.prompts import PromptTemplate


# In[56]:


import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[57]:


llm = OpenAI(openai_api_key="OPENAI_API_KEY")


# In[77]:


docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])


# In[81]:


loader = TextLoader("C:\\Users\\garre\\Documents\\wps.log")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[82]:


db = FAISS.from_documents(docs, embeddings)

query = "How much blue is there?"
docs = db.similarity_search(query)


# In[83]:


print(docs[0].page_content)


# In[84]:


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")


# In[85]:


query = "How much Green is there?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# In[86]:


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If the user asks a question related to previous questions or responses, use the {summaries} information section and cite "You" as the source.
If the user asks about information found the files, cite the file as the source instead.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
query = "How much green is there?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# In[76]:


query = "What was the last question I asked?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# In[ ]:




