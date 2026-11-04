
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Check whether the API key exists
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

# Initialize the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=api_key
)

# Send a prompt
response = llm.invoke("Explain what RAG is in two simple sentences.")

# Display the response
print(response.content)
