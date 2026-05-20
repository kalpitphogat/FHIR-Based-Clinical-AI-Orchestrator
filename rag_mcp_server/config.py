import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "discharge-rag").strip()
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default").strip()
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large").strip()
PINECONE_TOP_K_DEFAULT = int(os.getenv("PINECONE_TOP_K_DEFAULT", "5"))
RAG_MATCH_THRESHOLD = float(os.getenv("RAG_MATCH_THRESHOLD", "0.8"))

