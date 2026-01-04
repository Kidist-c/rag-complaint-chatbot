import gradio as gr
from src.rag import RAGPipeline

# ----------------------------
# Initialize RAG pipeline
# ----------------------------
try:
    rag = RAGPipeline(
        index_path="notebooks/vector_store/faiss_index.bin",
        metadata_path="notebooks/vector_store/faiss_metadata.csv"
    )
    print("[INFO] RAG pipeline loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load RAG pipeline: {e}")
    rag = None

# ----------------------------
# Gradio callback function
# ----------------------------
def answer_question(user_question):
    if rag is None:
        return "RAG pipeline not loaded.", ""
    if not user_question.strip():
        return "Please enter a question.", ""
    
    try:
        answer, retrieved_chunks = rag.answer(user_question, k=5)
        sources = "\n\n".join(
            [f"ID: {row['complaint_id']}, Product: {row['product']}\n{row['chunk_text']}"
             for _, row in retrieved_chunks.iterrows()]
        )
        return answer, sources
    except Exception as e:
        return f"Error: {e}", ""

# ----------------------------
# Build Gradio Interface
# ----------------------------
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about customer complaints..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources / Context")
    ],
    title="CrediTrust Complaint Analyzer",
    description="Ask questions about customer complaints and get context-aware answers."
)

# ----------------------------
# Launch the interface
# ----------------------------
if __name__ == "__main__":
    iface.launch(debug=True)