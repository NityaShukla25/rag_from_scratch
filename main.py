from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import numpy as np


# Load variables from the .env file
load_dotenv()


# Create OpenAI client using the API key from .env
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Path to our documents folder
documents_path = Path("documents")


# Function to split a document into chunks
def chunk_text(text):

    # Split text whenever there is a blank line
    chunks = text.split("\n\n")

    # Remove extra spaces and ignore empty chunks
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    return chunks


# This list will store all chunks from all documents
all_chunks = []


# Go through every file inside the documents folder
for file in documents_path.iterdir():

    # Read the complete file as text
    text = file.read_text()

    # Split the document into smaller chunks
    chunks = chunk_text(text)

    # Store every chunk along with its source file
    for chunk in chunks:

        all_chunks.append({
            "text": chunk,
            "source": file.name
        })


# Extract only the text from each chunk
# We need this list to generate embeddings
texts = [
    chunk["text"]
    for chunk in all_chunks
]


# Generate embeddings for all document chunks
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)


# Attach each generated embedding to its corresponding chunk
for i, item in enumerate(all_chunks):

    item["embedding"] = response.data[i].embedding


# User's question
query = "What CGPA is required for campus placement?"


# Convert the user's question into an embedding
# We use the SAME embedding model used for document chunks
query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)


# Extract the actual vector from the API response
query_embedding = query_response.data[0].embedding


# Function to calculate cosine similarity
# It tells us how semantically similar two vectors are
def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# Calculate similarity between the query and every chunk
for chunk in all_chunks:

    score = cosine_similarity(
        query_embedding,
        chunk["embedding"]
    )

    # Store the similarity score inside the chunk
    chunk["score"] = score


# Sort chunks from highest similarity to lowest similarity
all_chunks.sort(
    key=lambda x: x["score"],
    reverse=True
)


# Number of chunks we want to retrieve
top_k = 3


# Print the top 3 most relevant chunks
for chunk in all_chunks[:top_k]:

    print("Score:", chunk["score"])
    print("Source:", chunk["source"])
    print("Text:", chunk["text"])
    print()

# Create context from the retrieved chunks
context = "\n\n".join(
    chunk["text"]
    for chunk in all_chunks[:top_k]
)

print("CONTEXT:")
print(context)

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response.choices[0].message.content

print("ANSWER:")
print(answer)