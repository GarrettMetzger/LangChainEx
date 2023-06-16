#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader


# In[8]:


os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[9]:


llm = OpenAI(temperature=0)


# In[26]:


with open('C:\\Users\\garre\\Documents\\STU.txt') as f:
    state_of_the_union = f.read()


# In[32]:


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
)

docs = text_splitter.create_documents([state_of_the_union])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


# In[12]:


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# In[13]:


chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs,return_source_documents=True)


# In[14]:


def source_gather(q):
    source_gather.result = qa({"query": q})
    return source_gather.result["result"]


# In[15]:


tools = [
    Tool(
        name = "State of Union QA System",
        func= lambda x: source_gather(question),
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question."
    )
]


# In[16]:


prefix = """Have a conversation with a human, answering the following questions as best you can. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use the question asked as the input for the other documents. Offer follow up questions which the user may ask later.
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad","chat_history"]
)
memory = ConversationBufferMemory(memory_key="chat_history")


# In[17]:


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


# In[18]:


question= "What did the president say to the Republicans?"
agent_chain.run(question)


# In[32]:


source_gather.result['source_documents']


# In[19]:


question = "What did He say about the economy?"
agent_chain.run(question)


# In[ ]:




