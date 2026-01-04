import gradio as gr
import sys
sys.path.append("../src")
from src.rag import RAGPipeline

# Initialize your RAG pipeline
rag = RAGPipeline(
    index_path="vector_store/faiss_index.bin",
    metadata_path="vector_store/faiss_metadata.csv"
)

def answer_question(user_question):
    """
    This function receives user input, queries the RAG pipeline,
    and returns the answer + source text chunks.
    """
    answer, sources = rag.query(user_question, top_k=5)
    sources_text = "\n\n".join([f"- {s}" for s in sources])
    return answer, sources_text

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    
    with gr.Row():
        user_input = gr.Textbox(label="Ask a question about customer complaints", lines=2)
        submit_btn = gr.Button("Ask")
    
    answer_box = gr.Textbox(label="Answer", lines=10)
    sources_box = gr.Textbox(label="Source Chunks", lines=10)
    
    clear_btn = gr.Button("Clear")
    
    submit_btn.click(answer_question, inputs=user_input, outputs=[answer_box, sources_box])
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[answer_box, sources_box, user_input])

demo.launch()
