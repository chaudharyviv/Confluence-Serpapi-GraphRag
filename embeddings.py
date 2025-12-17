from openai import OpenAI
from config import OPENAI_EMBED_MODEL

client = OpenAI()

def embed(text: str):
    return client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    ).data[0].embedding
