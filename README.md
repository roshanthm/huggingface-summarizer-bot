# 🧠 AI Text Summarizer & Question Answering System  
A modular NLP application built using **Hugging Face Transformers** and **LangChain**, capable of generating high-quality summaries, refining them using a second model, and answering questions based on the final summary.  
Designed for CPU-friendly deployment and hackathon-ready performance.

---

## 🚀 Features

### ✨ Text Summarization
Takes long text and creates a **short**, **medium**, or **long** summary using a lightweight summarization model.

### 🔁 Summary Refinement  
Uses a second summarization model to polish and improve the generated summary.

### ❓ Question Answering  
Allows users to ask questions about the summary and returns accurate answers.

### ⚙ Modular Architecture  
All logic is cleanly separated:
- Summarizer  
- Refiner  
- Q/A Pipeline  
- Chain Logic  
- Prompt Template  

### 💻 CPU-Optimized  
All models run efficiently on CPU. No GPU required.

---

## 🛠️ Tech Stack

| Component     | Technology |
|---------------|------------|
| NLP Models    | Hugging Face Transformers |
| Pipeline Logic | LangChain + HuggingFacePipeline |
| Core Models   | distilBART, BART-Large, RoBERTa QA |
| Language       | Python |
| Runtime        | CPU-friendly |

---



