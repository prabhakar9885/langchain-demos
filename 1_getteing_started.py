import os;
from constants import openai_key;
from langchain_openai.llms import OpenAI;

import streamlit as st;

os.environ['OPENAI_API_KEY'] = openai_key;

st.title('Demo title');
inp = st.text_input('text_input');

llm = OpenAI( temperature = 0.8 )

if inp:
    st.write( llm.invoke(inp))