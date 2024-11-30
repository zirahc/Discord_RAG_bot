import time
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
chat_history_index = os.getenv("CHAT_HISTORY_INDEX")

pc = Pinecone()
# Create Pinecone index if it doesn't exist
if not pc.has_index(chat_history_index):
    print("No chat history index")
    pc.create_index(
        name=chat_history_index,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    # Wait for index to be ready
    while not pc.describe_index(chat_history_index).status['ready']:
        time.sleep(1)

# Initialize embeddings model
embedding_model = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=chat_history_index, embedding=embedding_model)

async def save_message_to_vector_db(message):
    metadata = {
        "author": message["author"],
        "timestamp": str(message["timestamp"])
    }
    vectorstore.add_texts(
        texts=[f"[Author: {metadata['author']} at {metadata['timestamp']}] Message: {message['content']}"], 
        metadatas=[metadata]
    )

