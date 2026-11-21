import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

question = "What is the deadline for module registration changes?"  
result = collection.query(
    query_embeddings=model.encode([question]).tolist(),
    n_results=3,
)
for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
    print(f"[{meta['source']} - page {meta['page']}]")
    print(doc)
    print("-" * 40)