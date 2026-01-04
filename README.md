# Intelligent Complaint Analysis for Financial Services

## Problem Statement

CrediTrust Financial receives thousands of customer complaints every month across multiple financial products, including credit cards, personal loans, savings accounts, and money transfers. These complaints contain valuable insights about customer dissatisfaction, product issues, and potential compliance risks.

However, the unstructured nature of complaint narratives makes it challenging for internal teams to extract actionable insights efficiently. Traditional manual analysis is time-consuming, prone to oversight, and often reactive rather than proactive.

This project aims to address this problem by building a Retrieval-Augmented Generation (RAG) system that allows non-technical users to query complaint data in natural language and receive concise, evidence-backed responses.

---

## Business Objectives

The intelligent complaint analysis system is designed to:

- Reduce the time required to identify emerging complaint trends from days to minutes.
- Empower non-technical teams (e.g., Customer Support, Compliance) to explore complaint data independently.
- Improve product decision-making by surfacing recurring issues and pain points across financial products.
- Enhance risk and compliance monitoring by identifying repeated violations or problematic patterns early.

---

## Features

1. EDA & Data Preprocessing

   - Analyze complaint distribution by product and issue.
   - Remove empty or irrelevant complaint narratives.
   - Clean and normalize complaint text.

2. Stratified Sampling & Text Chunking

   - Generate a representative sample of complaints across product categories.
   - Chunk long narratives into smaller, semantically coherent pieces to optimize retrieval.

3. Embeddings & Vector Store Indexing

   - Convert complaint chunks into vector embeddings using sentence-transformers/all-MiniLM-L6-v2.
   - Store embeddings in FAISS vector database along with metadata for traceability.

4. RAG Pipeline

   - Embed user queries and retrieve top relevant complaint chunks.
   - Generate context-aware, concise answers using an LLM (google/flan-t5-small).

5. Interactive Chat Interface
   - User-friendly web interface built with Gradio.
   - Input box for queries, display of AI-generated answers, and source chunks for transparency.

---
