from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DB_DIR = "career_pilot_chroma_db"


def get_llm(model: str = "openai/gpt-oss-20b",
            temperature: float = 0.2):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found")

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Stage 1: Load Documents
# -----------------------------
def load_text_file(file_path: str, source_name: str, doc_type: str) -> List[Document]:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": source_name, "doc_type": doc_type})]


def create_documents(resume_text: str, jd_text: str) -> List[Document]:
    return [
        Document(page_content=resume_text, metadata={"source": "uploaded_resume", "doc_type": "resume"}),
        Document(page_content=jd_text, metadata={"source": "uploaded_job_description", "doc_type": "job_description"}),
    ]


# -----------------------------
# Stage 2: Split Documents
# -----------------------------
def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


# -----------------------------
# Stage 3 + 4: Embeddings + Vector DB
# -----------------------------
def build_vectorstore(chunks: List[Document], persist_directory: str = DB_DIR):

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="career_pilot_rag",
    )

    return vectorstore

# -----------------------------
# Stage 5: Retrieve Context
# -----------------------------
def retrieve_context(vectorstore, query: str, k: int = 4):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    source_docs = retriever.invoke(query)

    print("\n--- RETRIEVED DOCUMENTS ---")

    for i, doc in enumerate(source_docs):
        print(f"\nDocument {i + 1}")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:300])

    context = "\n\n".join(
        doc.page_content for doc in source_docs
    )

    return context, source_docs

# -----------------------------
# Stage 6: Generate Answer
# -----------------------------

def run_career_pilot(vectorstore, resume_text: str, jd_text: str, question: str):
    llm = get_llm()

    # Retrieve relevant chunks from ChromaDB
    retrieval_query = f"""
    Find information from the resume and job description
    relevant to the following question:

    {question}
    """

    context, source_docs = retrieve_context(
        vectorstore,
        retrieval_query,
        k=6
    )

    # Include the original documents directly in the prompt.
    # This ensures the LLM receives both documents even if
    # ChromaDB does not retrieve every relevant chunk.
    prompt = ChatPromptTemplate.from_template("""
You are an expert AI Career Coach.

Your task is to analyze a candidate's resume against a job description.

IMPORTANT INSTRUCTIONS:
1. The resume and job description provided below are the primary sources.
2. Do not claim that the job description is missing when it is provided below.
3. Compare the candidate's actual skills with the job requirements.
4. Do not invent skills, qualifications, or experience.
5. Clearly distinguish between skills the candidate has and skills they need.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ADDITIONAL RETRIEVED CONTEXT FROM CHROMADB:
{context}

USER QUESTION:
{question}

Provide a detailed, practical answer using these sections when relevant:

1. Overall Resume and Job Match
2. ATS-Style Match Score (explain that it is an estimate)
3. Matching Skills
4. Missing Skills and Skill Gaps
5. Resume Improvement Suggestions
6. Recommended Projects
7. Interview Preparation Questions

If the job description is very short, explain that the analysis is limited
by the available job requirements instead of saying it is missing.

Make the response structured, specific, and actionable.
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "resume_text": resume_text,
        "jd_text": jd_text,
        "context": context,
        "question": question
    })

    return answer, source_docs


def generate_complete_report(vectorstore, resume_text: str, jd_text: str):
    question = """
    Analyze this resume against this job description. Provide ATS-style score, skill match, missing skills,
    resume improvement suggestions, project suggestions, and interview questions.
    """
    return run_career_pilot(vectorstore, resume_text, jd_text, question)
