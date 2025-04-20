import os

class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "xYaq8vnQzSbmKjZGvnefLa2ugS1jZnj2")
    MISTRAL_MODEL_ID = os.getenv("MISTRAL_MODEL_ID", "ft:open-mistral-7b:70f38c7a:20250416:2b82656c")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")