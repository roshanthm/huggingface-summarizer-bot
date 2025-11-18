from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

# ---------------------------------------------
# 1️⃣ Summarization Pipeline (CPU)
# ---------------------------------------------
# Replace with fast model for CPU:
# model="sshleifer/distilbart-cnn-12-6"   ← MUCH faster on CPU
summarization_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU mode
)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

# ---------------------------------------------
# 2️⃣ Refinement Pipeline (CPU)
# ---------------------------------------------
refinement_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large",
    device=-1  # CPU
)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

# ---------------------------------------------
# 3️⃣ Q/A Pipeline (CPU)
# ---------------------------------------------
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=-1  # CPU
)

# ---------------------------------------------
# 4️⃣ Prompt Template
# ---------------------------------------------
summary_template = PromptTemplate.from_template(
    "Summarize the following text in a {length} way:\n\n{text}"
)

# Chain → Summarizer → Refiner
summarization_chain = summary_template | summarizer | refiner

# ---------------------------------------------
# 5️⃣ Input from user
# ---------------------------------------------
text_to_summarize = input("\nEnter text to summarize:\n")
length = input("\nEnter summary length (short / medium / long): ")

# ---------------------------------------------
# 6️⃣ Generate Summary
# ---------------------------------------------
summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🔹 **Generated Summary:**")
print(summary)

# ---------------------------------------------
# 7️⃣ Q/A Loop
# ---------------------------------------------
while True:
    question = input("\nAsk a question about the summary (or type 'exit' to leave): ")

    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n🔹 **Answer:**")
    print(qa_result["answer"])
