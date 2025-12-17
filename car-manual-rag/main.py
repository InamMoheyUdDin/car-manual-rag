from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

car_docs_split = text_splitter.split_documents(car_docs)

vectorstore = Chroma.from_documents(
    documents=car_docs_split,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

question = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

prompt = """
You are a helpful car assistant AI chatbot. Answer the questions asked based on the context provided below.
If you are unsure, then just say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""

prompt_template = ChatPromptTemplate([("human", prompt)])

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
)

result = rag_chain.invoke(question)
print(result.content)
