import os
import re
from collections import Counter

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
CHUNK_SIZE = 500
OVERLAP = 50


# STEP 1: Extract text from each page of a PDF
def extract_pages(path):
    doc = fitz.open(path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


# STEP 2a: Find lines that repeat on many pages (headers/footers)
def find_repeated_lines(pages):
    counter = Counter()
    for _, text in pages:
        unique_lines = set(line.strip() for line in text.split("\n"))
        for line in unique_lines:
            if line:
                counter[line] += 1
    limit = max(3, len(pages) * 0.5)
    return {line for line, count in counter.items() if count >= limit}


# STEP 2b: Clean one page of text
def clean_text(text, repeated_lines):
    text = text.replace("-\n", "")  # join split words
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if line in repeated_lines:
            continue  # remove header/footer
        if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", line, flags=re.I):
            continue  # remove page numbers
        kept.append(line)
    text = " ".join(kept)
    text = re.sub(r"\s+", " ", text).strip()  # collapse extra spaces
    return text


# STEP 3: Cut text into overlapping chunks
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def main():
    ids, documents, metadatas = [], [], []

    # Read every PDF in the data folder
    for filename in os.listdir(DATA_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        print(f"Reading {filename} ...")
        pages = extract_pages(os.path.join(DATA_FOLDER, filename))
        repeated = find_repeated_lines(pages)

        for page_num, raw_text in pages:
            text = clean_text(raw_text, repeated)
            for j, chunk in enumerate(chunk_text(text)):
                if len(chunk) < 30:  # skip tiny useless chunks
                    continue
                ids.append(f"{filename}-p{page_num}-c{j}")
                documents.append(chunk)
                metadatas.append({"source": filename, "page": page_num})

    print(f"Total chunks: {len(documents)}")
    if not documents:
        print("No text found. Check that your PDFs are in the data folder and are not scanned images.")
        return

    # STEP 4: Turn chunks into numbers (embeddings)
    print("Creating embeddings (first run downloads the model, please wait)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # STEP 5: Store in ChromaDB (start fresh each time)
    client = chromadb.PersistentClient(path=DB_FOLDER)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    collection = client.create_collection("docs")

    batch = 500
    for i in range(0, len(documents), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=documents[i:i + batch],
            embeddings=embeddings[i:i + batch],
            metadatas=metadatas[i:i + batch],
        )

    print(f"Done! {collection.count()} chunks stored in '{DB_FOLDER}'.")


if __name__ == "__main__":
    main()