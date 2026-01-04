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
# Gradio callback
# ----------------------------
def answer_question(user_question):
    if rag is None:
        return "RAG pipeline not loaded.", ""
    if not user_question.strip():
        return "Please enter a question.", ""
    
    answer, retrieved_chunks = rag.answer(user_question, k=5)
    sources = "\n\n".join(
        [
            f"ID: {row['complaint_id']}, Product: {row['product']}\n{row['chunk_text']}"
            for _, row in retrieved_chunks.iterrows()
        ]
    )
    return answer, sources

def clear_fields():
    return "", "", ""

# ----------------------------
# Build Gradio UI
# ----------------------------
with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:
    gr.Markdown("## 🏦 CrediTrust Complaint Analyzer")
    gr.Markdown("Ask questions about customer complaints and get context-aware answers.")

    question = gr.Textbox(
        lines=2,
        placeholder="Ask a question about customer complaints...",
        label="Your Question"
    )

    answer = gr.Textbox(label="Answer")
    sources = gr.Textbox(label="Sources / Context")

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    submit_btn.click(
        fn=answer_question,
        inputs=question,
        outputs=[answer, sources]
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[question, answer, sources]
    )

# ----------------------------
# Launch app
# ----------------------------
if __name__ == "__main__":
    demo.launch(debug=True)
