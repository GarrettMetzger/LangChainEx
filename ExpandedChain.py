#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


# In[14]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[15]:


from langchain.document_loaders import TextLoader
loader = TextLoader('C:\\Users\\garre\\Documents\\STU.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)


# In[27]:


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# In[28]:


chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs,return_source_documents=True)


# In[29]:


query = "What did the president say about Republicans?"
result = qa({"query": query})


# In[19]:


result["result"]


# In[20]:


result["source_documents"]


# In[21]:


query = "What was the last question I asked?"
result = qa({"query": query})


# In[22]:


result["result"]


# In[ ]:




