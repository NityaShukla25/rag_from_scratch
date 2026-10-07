from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Load API key from .env
load_dotenv()


# --------------------------------
# 1. Read the document
# --------------------------------

with open("documents/placement.txt", "r") as file:
    text = file.read()


# --------------------------------
# 2. Convert text into a Document
# --------------------------------

document = Document(
    page_content=text,
    metadata={
        "source": "placement.txt"
    }
)


# --------------------------------
# 3. Split document into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    [document]
)


# Print chunks
print("\nChunks:")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)


# --------------------------------
# 4. Create embedding model
# --------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# --------------------------------
# 5. Create vector store
# --------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


# --------------------------------
# 6. Create retriever
# --------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# --------------------------------
# 7. User query
# --------------------------------

query = "What CGPA is required for campus placement?"


# --------------------------------
# 8. Retrieve relevant documents
# --------------------------------

results = retriever.invoke(
    query
)


# --------------------------------
# 9. Print retrieved documents
# --------------------------------

print("\nRetrieved Documents:")

for i, result in enumerate(results):

    print(f"\nResult {i + 1}:")
    print(result.page_content)
    print("Metadata:", result.metadata)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
# --------------------------------
# 10. Create context
# --------------------------------

context = "\n\n".join(
    result.page_content
    for result in results
)


# --------------------------------
# 11. Create the final prompt
# --------------------------------

messages = prompt.invoke(
    {
        "context": context,
        "question": query
    }
)


# --------------------------------
# 12. Generate answer using LLM
# --------------------------------

response = llm.invoke(
    messages
)


# --------------------------------
# 13. Print final answer
# --------------------------------

print("\nAnswer:")
print(response.content)