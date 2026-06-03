import logging
import os
import fitz
import pandas as pd

# --- C1: Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("Logging initialised — pipeline starting.")

# --- C2: Ingest ---
PDF_PATH = "./report.pdf"
logger.info("Opening PDF: %s", PDF_PATH)
doc = fitz.open(PDF_PATH)

pages: list[str] = []
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text("text")
    pages.append(text)
    logger.info("Page %d extracted — %d characters.", page_num + 1, len(text))

doc.close()
logger.info("Extraction complete. Pages collected: %d", len(pages))

# --- C3: Structure ---
records = []
for idx, text in enumerate(pages):
    records.append({
        "page_no":        idx + 1,
        "word_count":     len(text.split()),
        "first_50_chars": text.strip()[:50].replace("\n", " "),
    })

df = pd.DataFrame(records, columns=["page_no", "word_count", "first_50_chars"])
logger.info("DataFrame built:\n%s", df.to_string(index=False))

# --- C4: Store ---
os.makedirs("output", exist_ok=True)

CSV_PATH  = "output/pages.csv"
JSON_PATH = "output/pages.json"

df.to_csv(CSV_PATH, index=False)
logger.info("CSV written  → %s", CSV_PATH)

df.to_json(JSON_PATH, orient="records")
logger.info("JSON written → %s", JSON_PATH)

# --- C5: Summarise ---
from groq import Groq

GROQ_MODEL  = "llama-3.1-8b-instant"
CHAR_LIMIT  = 2000
combined_text = "\n\n".join(pages)[:CHAR_LIMIT]

client = Groq()
logger.info("Sending %d characters to Groq model '%s'.", len(combined_text), GROQ_MODEL)

response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "system", "content": "You are a concise technical summariser."},
        {"role": "user",   "content": "Summarise this in exactly 3 bullet points.\n\n" + combined_text},
    ],
    temperature=0.3,
    max_tokens=256,
)

summary = response.choices[0].message.content
logger.info("Groq response received.")
print("\n===== GROQ SUMMARY =====")
print(summary)
print("========================\n")

# --- C6: Visualise ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(df["page_no"], df["word_count"], color="#4C72B0", edgecolor="white", width=0.5)

plt.title("Word Count per Page — report.pdf")
plt.xlabel("Page Number")
plt.ylabel("Word Count")
plt.xticks(df["page_no"])
plt.tight_layout()

CHART_PATH = "output/word_counts.png"
plt.savefig(CHART_PATH, dpi=150)
logger.info("Bar chart saved → %s", CHART_PATH)
plt.show()