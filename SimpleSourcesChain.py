#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI


# In[5]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[6]:


with open('C:\\Users\\garre\\Documents\\STU.txt') as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)

texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()


# In[7]:


docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))])


# In[9]:


chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())


# In[11]:


chain({"question": "What did the president say to the republicans?"}, return_only_outputs=True)


# In[ ]:




