import os
from constants import openai_key
from langchain_openai.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title('Demo title')
inp = st.text_input('text_input')

llm = OpenAI( temperature = 0.8 )

prompt_1 = PromptTemplate(
    input_variables=['person'],
    template="What is the date of birth of {person}"
)
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
chain_1 = LLMChain(
    llm=llm,
    prompt=prompt_1,
    verbose=True,
    output_key='dob',
    memory=dob_memory
)


prompt_2 = PromptTemplate(
    input_variables=['dob'],
    template="Mention 3 most important things that happened around {dob}"
)
events_memory = ConversationBufferMemory(input_key='dob', memory_key='chat_history')
chain_2 = LLMChain(
    llm=llm,
    prompt=prompt_2,
    verbose=True,
    output_key='events',
    memory=events_memory
)


chain = SimpleSequentialChain(
    chains=[chain_1,chain_2],
    verbose=True
)

if inp:
    st.write(chain.invoke(inp))
    with st.expander(label='DOB'):
        st.info(body=dob_memory.buffer)
    with st.expander(label='events'):
        st.info(body=events_memory.buffer)