#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


# In[29]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[30]:


loader = TextLoader("C:\\Users\\garre\\Documents\\wps.log")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


# In[31]:


tools = [
    
    Tool.from_function(
        func= lambda x:db.similarity_search("input"),
        name = "SearchDoc",
        description="useful for when you need to answer questions about the document or things in the document.Use this more than the normal search if the question is about the document such as 'What is the number associated with red in the document?'",
    ),
]


# In[32]:


prefix = """Have a conversation with a human, answering the following questions as best you can.Cite a source when possible. You have access to the following tools:"""
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


# In[33]:


llm = OpenAI(temperature=0)


# In[34]:


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


# In[35]:


agent_chain.run("How much blue is there?")


# In[36]:


agent_chain.run("What was the last question I asked?")


# In[ ]:




