import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests   # Make sure this is here

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("stock_index.faiss")

with open("stocks.json", "r") as f:
    stocks = json.load(f)

def semantic_search(query, top_k=3):
    embedding = model.encode([query])
    distances, indices = index.search(np.array(embedding), top_k)
    return [stocks[i] for i in indices[0]]

def generate_answer(context, question):
    import requests
    
    prompt = f"""
You are a professional Indian stock market analyst.

Context:
{context}

Question:
{question}

Give detailed financial analysis.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()
        print("DEBUG RESPONSE:", result)  # Important

        return result.get("response", "No response field found!")

    except Exception as e:
        return f"Error occurred: {e}"