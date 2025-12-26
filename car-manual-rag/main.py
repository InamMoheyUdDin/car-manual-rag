from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ("OPENAI_API_KEY")
pinecone_api_key = os.environ("PINECONE_API_KEY")

if not openai_api_key:
    raise ValueError("OPEN_API_KEY not set")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set")

try:
    loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
    car_docs = loader.load()
except Exception as e:
    print(f"Error: {e}")

llm = ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

car_docs_split = text_splitter.split_documents(car_docs)

try:
    vectorstore = Chroma.from_documents(
        documents=car_docs_split,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={"k":3}
    )
except Exception as e:
    print(f"Error: {e}")

question = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

prompt = """
You are a helpful car assistant AI chatbot. Answer the questions asked based on the context provided below. If you are unsure, then just say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

prompt_template = ChatPromptTemplate([("human", prompt)])

try:
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
    )

    result = rag_chain.invoke(question)
    answer = str(result.content)
    print(answer)
except Exception as e:
    Print(f"Error: {e}")
