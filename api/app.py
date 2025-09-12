from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server with LangServe"
)

# LLM
llm = Ollama(model="llama2")

add_routes(
    app,
    llm,
    path="/llama2"
)

prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} with 100 words"
)

add_routes(
    app,
    prompt2 | llm,   #this is a pipeline first the prompt then it is fed to the llm
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
