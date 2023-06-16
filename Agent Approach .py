#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from pathlib import Path
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain


# In[3]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[4]:


llm = OpenAI(temperature=0)


# In[5]:


with open('C:\\Users\\garre\\Documents\\STU.txt') as f:
    state_of_the_union = f.read()


# In[6]:


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)

docs = text_splitter.create_documents([state_of_the_union])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


# In[7]:


state_of_union = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())


# In[8]:


def source_gather(query):
    sources = db.similarity_search(query)
    state_of_union.run


# In[10]:


tools = [
    Tool(
        name = "State of Union QA System",
        func= lambda x:source_gather("input"),
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question."
    )
]


# In[20]:


prefix = """Have a conversation with a human, answering the following questions as best you can. Provide the file used to collect the answer at the end of the response. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")


# In[21]:


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


# In[22]:


agent_chain.run("What did the president say to the republicans?")


# In[23]:


agent_chain.run("What was the last question I asked?")


# In[ ]:




