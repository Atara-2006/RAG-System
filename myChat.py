import os
import httpx
from dotenv import load_dotenv

from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import CitationQueryEngine

load_dotenv()

unsafe_client = httpx.Client(verify=False)

Settings.llm = OpenAI(model="gpt-4o-mini", http_client=unsafe_client)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", http_client=unsafe_client)

DATA_DIR = "./data"   
PERSIST_DIR = "./storage_homework"

def main():
    
    if not os.path.exists(PERSIST_DIR):
        print("--- Loading 3 PDFs and creating index... ---")
        
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

  
    query_engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=3, 
        citation_chunk_size=512,
    )

    print("\n" + "="*50)
    print("Homework Model Ready - Answers with Sources")
    print("="*50)

   
    while True:
        question = input("\nEnter your question (or 'exit'): ")
        if question.lower() == 'exit':
            break
        
        try:
            response = query_engine.query(question)
            
            print(f"\nAnswer: {response}")
            
            print("\nSources used:")
            for node in response.source_nodes:
                source_file = node.node.metadata.get('file_name', 'Unknown Document')
                print(f"- {source_file}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()