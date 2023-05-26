#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# In[3]:


import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')


# In[4]:


llm = OpenAI(openai_api_key="OPENAI_API_KEY")


# In[5]:


loader = TextLoader("C:\\Users\\garre\\Documents\\wps.log")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# In[6]:


chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")


# In[7]:


query = "How much Green is there?"
chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# In[8]:


db = FAISS.from_documents(docs, embeddings)



# In[14]:


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If the user asks a question related to previous questions or responses, use the {summaries} information section and cite "You" as the source for that answer.
If the user asks about information found the files, cite the file as the source.
ALWAYS return a "SOURCES" part in your answer.

{summaries}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "summaries"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
query = "How much Blue is there?"
chain({"input_documents": db.similarity_search(query), "human_input": query}, return_only_outputs=True)


# In[16]:


query = "How much Green is there?"
chain({"input_documents": db.similarity_search(query), "human_input": query}, return_only_outputs=True)


# In[15]:


query = "What color did I ask about?"
chain({"input_documents": db.similarity_search(query), "human_input": query}, return_only_outputs=True)


# In[ ]:




