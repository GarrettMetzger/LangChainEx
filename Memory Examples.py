#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os


# In[8]:


os.environ["OPENAI_API_KEY"] = "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"


# In[9]:


from langchain.llms import OpenAI


# In[10]:


from langchain.chains.api.prompt import API_RESPONSE_PROMPT


# In[11]:


from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")


# In[12]:


from langchain.memory import ConversationBufferMemory


# In[13]:


history.messages


# In[14]:


from langchain.llms import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)


# In[15]:


conversation.predict(input="Hi there!")


# In[16]:


conversation.predict(input = "Can I have recommendations for dinner?")


# In[17]:


conversation.predict(input = "I am allergic to peanuts")


# In[18]:


conversation.predict(input = "No")


# In[19]:


conversation.predict(input = "What am I allergic to?")


# In[20]:


from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
OPENAI_API_KEY= "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"

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)


chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=2),
)

output = chatgpt_chain.predict(human_input="I want you to act as a Linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do not write explanations. Do not type commands unless I instruct you to do so. When I need to tell you something in English I will do so by putting text inside curly brackets {like this}. My first command is pwd.")
print(output)


# In[21]:


output = chatgpt_chain.predict(human_input="ls ~")
print(output)


# In[ ]:




