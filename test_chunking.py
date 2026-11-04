
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(docs)


# Sample resume
resume = Document(
    page_content="I have experience in Python, machine learning, deep learning, SQL and TensorFlow. " * 30,
    metadata={"source": "resume"}
)

# Split the document
chunks = split_documents([resume])

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
