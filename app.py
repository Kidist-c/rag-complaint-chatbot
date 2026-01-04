# app.py
import gradio as gr
from src.rag import RAGPipeline  # your pipeline class

# ----------------------------
# Initialize RAG pipeline
# ----------------------------
try:
    rag = RAGPipeline(
        index_path="notebooks/vector_store/faiss_index.bin",    # adjust if needed
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
        # Get answer and retrieved chunks
        answer, retrieved_chunks = rag.answer(user_question, k=5)
        
        # Format retrieved chunks for display
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
    description="Ask questions about customer complaints and get context-aware answers.",
    allow_flagging="never"
)

# ----------------------------
# Launch the interface
# ----------------------------
if name == "main":
    iface.launch(debug=True)