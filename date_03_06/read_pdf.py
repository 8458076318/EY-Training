import fitz  # pip install pymupdf

# ── Open the PDF ───────────────────────────────────
doc = fitz.open("Day4_coding_Assignment.pdf")

print(f"Pages : {len(doc)}")
print(f"Title : {doc.metadata['title']}")
print(f"Author: {doc.metadata['author']}")

# ── Extract text from every page ───────────────────
for page_num, page in enumerate(doc, start=1):
    text = page.get_text()          # plain UTF-8 string
    print(f"\n── Page {page_num} ──")
    print(text.strip())

doc.close()