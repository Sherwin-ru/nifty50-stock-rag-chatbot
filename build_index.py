import pdfplumber
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_PATH = "dataset/NIFTY50_Full_RAG_Dataset_Filled.pdf"

def extract_text():
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_stocks(text):
    stocks = []
    entries = text.split("Stock:")

    for entry in entries[1:]:
        lines = entry.strip().split("\n")
        stock = {"name": lines[0].strip()}

        for line in lines:
            if "PE Ratio" in line:
                stock["pe"] = float(line.split(":")[1].strip())
            if "ROE" in line:
                stock["roe"] = float(line.split(":")[1].strip())
            if "Debt to Equity" in line:
                stock["debt"] = float(line.split(":")[1].strip())
            if "RSI" in line:
                stock["rsi"] = float(line.split(":")[1].strip())
            if "1-Year Return" in line:
                stock["return"] = float(line.split(":")[1].strip())

        stocks.append(stock)

    return stocks

def create_index(stocks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = [str(s) for s in stocks]
    embeddings = model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, "stock_index.faiss")

    with open("stocks.json", "w") as f:
        json.dump(stocks, f)

    print("✅ Index Created Successfully!")

if __name__ == "__main__":
    text = extract_text()
    stocks = parse_stocks(text)
    create_index(stocks)