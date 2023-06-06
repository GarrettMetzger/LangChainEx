#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import getpass
import pprint
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain


# In[30]:


def source_gather(query):
    source_gather.sources = db.similarity_search(query)
    return db.similarity_search(query)


# In[17]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[18]:


loader = TextLoader("C:\\Users\\garre\\Documents\\wps.log")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


# In[19]:


tools = [
    
    Tool.from_function(
        func= lambda x: source_gather("input"),
        name = "SearchDoc",
        description="useful for when you need to answer questions about the document or things in the document.Use this more than the normal search if the question is about the document such as 'What is the number associated with red in the document?'",
    ),
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


llm = OpenAI(temperature=0)


# In[22]:


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


# In[31]:


agent_chain.run("How much blue is there?")


# In[24]:


agent_chain.run("What were the sources of the last answer?")


# In[25]:


agent_chain.run("What color is a stop sign? How much of that color is in the document?")


# In[26]:


agent_chain.run("What is the number associated with the color blue in the document?")


# In[33]:


source_gather.sources


# In[ ]:




