
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert text into numerical vectors
text1 = "I love machine learning"
text2 = "I enjoy learning artificial intelligence"

vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)

# Display results
print("Vector dimensions:", len(vector1))
print("First 10 values:", vector1[:10])

print("\nBoth texts converted into embeddings successfully!")
