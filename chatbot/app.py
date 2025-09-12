from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# prompt Template 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please repond to the user queries"),
        ("user","Question:{question}")
    ]
)

# streamlit framework
st.title('Langchain Demo With LLama2')
input_text = st.text_input("Search the topic u want")

#ollama
llm = OllamaLLM(model="llama2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

