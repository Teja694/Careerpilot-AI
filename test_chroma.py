
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("STEP 1: Initializing embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("STEP 2: Embedding model initialized!")

documents = [
    Document(
        page_content="I have experience in Python and machine learning.",
        metadata={"source": "resume"}
    ),
    Document(
        page_content="The candidate should know SQL and database management.",
        metadata={"source": "job_description"}
    ),
    Document(
        page_content="Experience with deep learning and neural networks is preferred.",
        metadata={"source": "job_description"}
    )
]

print("STEP 3: Documents created:", len(documents))

print("STEP 4: Creating ChromaDB vector store...")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="test_career_coach"
)

print("STEP 5: Documents stored successfully!")

query = "Which skills are related to machine learning?"

print("STEP 6: Searching:", query)

results = vectorstore.similarity_search(query, k=2)

print("STEP 7: Number of results:", len(results))

for result in results:
    print("\nRetrieved document:")
    print(result.page_content)
    print("Metadata:", result.metadata)

print("\nTEST COMPLETED!")
