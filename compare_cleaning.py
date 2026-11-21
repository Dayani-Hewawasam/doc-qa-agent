from ingest import extract_pages, find_repeated_lines, clean_text, chunk_text

pdf_path = "data/Computer Simulation of a Distance Protection for a Transmission Line-1-5.pdf"   # change to your real file name
page_to_check = 3                # change to any page number

pages = extract_pages(pdf_path)
repeated = find_repeated_lines(pages)

raw_text = dict(pages)[page_to_check]
clean = clean_text(raw_text, repeated)

print("=== REPEATED HEADERS/FOOTERS FOUND (removed) ===")
for line in repeated:
    print(" -", line)

print("\n=== RAW TEXT (before cleaning) ===")
print(raw_text)

print("\n=== CLEAN TEXT (after cleaning) ===")
print(clean)

print("\n=== CHUNKS ===")
for i, c in enumerate(chunk_text(clean)):
    print(f"--- chunk {i} ({len(c)} chars) ---")
    print(c)