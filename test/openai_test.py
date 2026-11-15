import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

print("Embedding Model:", os.getenv("OPENAI_EMBED_MODEL"))
print("Chat Model:", os.getenv("OPENAI_CHAT_MODEL"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Test 1: OpenAI Embeddings
embedding_response = client.embeddings.create(
    model=os.getenv("OPENAI_EMBED_MODEL"),
    input="Hello OpenAI!"
)

embedding_data = embedding_response.data[0].embedding

print("Embedding dimension:", len(embedding_data))

# Test 2: OpenAI Chat Model
chat_response = client.chat.completions.create(
    model=os.getenv("OPENAI_CHAT_MODEL"),
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ],
    max_completion_tokens=300
)

print(
    "Response:",
    chat_response.choices[0].message.content
)