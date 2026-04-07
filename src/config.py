from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# LM Studio runs an OpenAI compatible server on localhost:1234
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio" # can be anything

def get_llm(temperature=0.0):
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=LM_STUDIO_API_KEY,
        temperature=temperature,
        streaming=True,
    )

def get_embeddings():
    # Use local HuggingFace embeddings to avoid API calls
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
