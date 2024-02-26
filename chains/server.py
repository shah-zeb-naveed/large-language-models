#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
#from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
# import chat model from huggingfacehub
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceHub

# load dotenv
from dotenv import load_dotenv
load_dotenv()

# load model/modules
embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
    )

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
        'include_prompt_in_result' : False,
        "return_full_text": False
    },
)

chat_model = ChatHuggingFace(llm=llm)

# define app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai",
# )

# add_routes(
#     app,
#     ChatAnthropic(),
#     path="/anthropic",
# )

add_routes(
    app,
    chat_model,
    path="/hf",
)


model = chat_model
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
