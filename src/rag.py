import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGPipeline:
    def __init__(self, index_path, metadata_path):
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Load embedding model (same as Task 2)
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load generator LLM (lightweight but effective)
        self.generator = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            max_new_tokens=300,
            temperature=0.2
        )
    def retrieve(self, question, k=5):
        # Embed the question
        question_vector = self.embedder.encode([question]).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(question_vector, k)

        # Retrieve corresponding metadata
        results = self.metadata.iloc[indices[0]]

        return results
    def build_prompt(self, context, question):
        prompt = f"""
           You are a financial analyst assistant for CrediTrust.
           Answer the user's question using ONLY the complaint excerpts below.
           If the information is insufficient, clearly state that.

           Context:
           {context}

            Question:
            {question}

            Answer:
          """
        return prompt
    def answer(self, question, k=5):
        retrieved = self.retrieve(question, k)

        context = "\n\n".join(
            retrieved["text_chunk"].tolist()
        )

        prompt = self.build_prompt(context, question)

        response = self.generator(prompt)[0]["generated_text"]

        return response, retrieved


    
    

