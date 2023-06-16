#!/usr/bin/env python
# coding: utf-8

# In[16]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import getpass
import os
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor


# In[8]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[9]:


llm = OpenAI(temperature=0)


# In[12]:


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
state_of_union_store = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")


# In[14]:


vectorstore_info = VectorStoreInfo(
    name="state_of_union_address",
    description="the most recent state of the Union adress",
    vectorstore=state_of_union_store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)


# In[19]:


agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


# In[20]:


agent_executor.run("What did biden say to the republicans?")


# In[21]:


agent_executor.run("What was the last question I asked?")


# In[ ]:




