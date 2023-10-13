import os
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader

loader = TextLoader("./wonders.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Weaviate.from_documents(docs, embeddings, weaviate_url=os.getenv("WEAVIATE_URL"), by_text=False)