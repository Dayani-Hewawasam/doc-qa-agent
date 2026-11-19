import chromadb

collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

print("Total chunks:", collection.count())
print()

data = collection.get(limit=22, include=["documents", "metadatas"])

for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
    print(f"ID: {chunk_id}")
    print(f"Source: {meta['source']} | Page: {meta['page']} | Length: {len(doc)} characters")
    print(doc)
    print("=" * 60)