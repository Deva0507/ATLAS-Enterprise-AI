from app.config import settings

print("AWS Region:", settings().aws_region)
print("OpenAI API Key:", settings().openai_api_key)
print("OpenAI Embed Model:", settings().openai_embed_model)
print("OpenAI Chat Model:", settings().openai_chat_model)