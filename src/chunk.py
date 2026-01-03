
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer # converts text to embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter# splits text into chunks
import faiss # Facebook AI similarity search

# Optional for ChromaDB
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ComplaintEmbeddingProcessor:
    """
    Handles Task 2:
    - Stratified sampling
    - Text chunking
    - Embedding generation
    - Vector store creation (FAISS / ChromaDB)
    """

    def __init__(self, df: pd.DataFrame, product_col="Product", text_col="cleaned_narrative"):
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None")
        if product_col not in df.columns or text_col not in df.columns:
            raise ValueError(f"Columns {product_col} or {text_col} not in DataFrame")
        
        self.df = df.copy()
        self.product_col = product_col
        self.text_col = text_col
        self.chunked_df = None
        self.embeddings = None
        self.faiss_index = None
        self.model = None

        print(f"[INFO] Initialized processor with {self.df.shape[0]} records")

    # ---------- Stratified Sampling ----------
    def stratified_sample(self, n_samples: int = 12000, random_state: int = 42):# Ensure all 5 products are represented
        """
        Return a stratified sample across product categories
        """
        try:
            self.df = self.df.groupby(self.product_col, group_keys=False).apply(
                lambda x: x.sample(frac=min(1, n_samples / len(self.df)), random_state=random_state)
            )
            print(f"[INFO] Stratified sample created: {self.df.shape[0]} records")
        except Exception as e:
            print(f"[ERROR] Stratified sampling failed: {e}")
        return self.df

    # ---------- Text Chunking ----------
    def chunk_texts(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Split each narrative into chunks
        """
        if self.df.empty:
            raise ValueError("DataFrame is empty. Cannot chunk text.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_chunks = []
        for idx, row in self.df.iterrows():
            text = str(row[self.text_col])
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "complaint_id": row.get("complaint_id", ""),
                    "product": row.get(self.product_col, ""),
                    "issue": row.get("issue", ""),
                    "sub_issue": row.get("sub_issue", ""),
                    "chunk_index": i,
                    "chunk_text": chunk
                })
        
        self.chunked_df = pd.DataFrame(all_chunks)
        print(f"[INFO] Text chunking complete: {self.chunked_df.shape[0]} chunks")
        return self.chunked_df

    # ---------- Embedding Generation ----------
    def generate_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        if self.chunked_df is None or self.chunked_df.empty:
            raise ValueError("Chunked DataFrame is empty. Cannot generate embeddings.")
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"[INFO] Loaded embedding model: {model_name}")
            self.embeddings = self.model.encode(
                self.chunked_df["chunk_text"].tolist(),
                show_progress_bar=True,
                batch_size=batch_size
            )
            print(f"[INFO] Embeddings generated: {len(self.embeddings)} vectors")
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
        return self.embeddings
# ---------- FAISS Indexing ----------
    def build_faiss_index(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not found. Generate embeddings first.")
        
        try:
            embedding_matrix = np.array(self.embeddings).astype("float32")
            dimension = embedding_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embedding_matrix)
            print(f"[INFO] FAISS index created with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            print(f"[ERROR] FAISS index creation failed: {e}")
        return self.faiss_index

    def save_faiss_index(self, index_path: str = "../vectorstore/faiss_index.bin", metadata_path: str = "../vectorstore/faiss_metadata.csv"):
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.faiss_index, index_path)
            self.chunked_df.to_csv(metadata_path, index=False)
            print(f"[INFO] FAISS index saved to {index_path}")
            print(f"[INFO] Metadata saved to {metadata_path}")
        except Exception as e:
            print(f"[ERROR] Saving FAISS index failed: {e}")

    # ---------- ChromaDB Indexing (Optional) ----------
    def build_chroma_index(self, collection_name="complaints"):
        if not CHROMA_AVAILABLE:
            print("[WARNING] ChromaDB not installed, skipping Chroma index.")
            return None
        if self.embeddings is None:
            raise ValueError("Embeddings not found. Generate embeddings first.")
        
        try:
            client = chromadb.Client()
            collection = client.create_collection(collection_name)
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            collection.add(
                documents=self.chunked_df["chunk_text"].tolist(),
                metadatas=self.chunked_df[["complaint_id","product","issue","sub_issue","chunk_index"]].to_dict(orient="records"),
                ids=[str(i) for i in range(len(self.chunked_df))],
                embeddings=self.embeddings.tolist()
            )
            client.persist("vector_store/chroma_db")
            print("[INFO] ChromaDB index created and persisted")
            return collection
        except Exception as e:
            print(f"[ERROR] ChromaDB indexing failed: {e}")
            return None