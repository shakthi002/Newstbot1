
import os
os.environ["COHERE_API_KEY"]='otNfOdqa4yewU0OwYzmLFydWZPnSJzuUk9vMxS0I'
os.environ['PINECONE_API_KEY']='ac2836be-5765-4e89-8d47-70ab0771e347'#'6dbebefb-e722-4241-8041-00f56ca935ca'
# os.environ['PINECONE_API_KEY']='97b14ea4-eb31-4004-bd0f-5b2b38ca5c92'#'6dbebefb-e722-4241-8041-00f56ca935ca'
os.environ['PINECONE_ENV']='gcp-starter'
os.environ['QDRANT_API_KEY']='B2p7WN_t2TIpugdRgeZ-S5ApOPZ-VigWZZxhxDE036aBbATU_mpx1g'
os.environ['GOOGLE_API_KEY']='AIzaSyAUggwhrE0LoTBDWrfeU6kxQuxA0FP6eCk'
os.environ['APIFY_API_TOKEN']='apify_api_K90vlEcLcKMx43KED0DpKQuxz2cTUr2CXPtv'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
os.environ['VOYAGE_API_KEY']='pa-yEmOi9CYAehyiFGbGJKRUwVxUfkdlNdXoqIulWYzNKs'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
from langchain_pinecone import PineconeVectorStore

import cohere
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import textwrap as tr
import random
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder

from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.cache import SQLiteCache
import voyageai
from langchain_community.embeddings import VoyageEmbeddings


# def create_hypothetical_chain():
    
#     prompt_template = """
#     Please write a passage to answer the question
#     Question: {question}

#     Answer:
#     """
#     prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
#     import google.generativeai as genai
#     genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     llm = ChatGoogleGenerativeAI(model="gemini-pro")
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     return llm_chain


def create_cache():
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

        
def vector_embedding(file_name=None):
    if file_name:
        with open(file_name, encoding='utf-8') as f:
            state_of_the_union = f.read()
            text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            separators=['\n \n'],
            chunk_size=3500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            )
            docs = text_splitter.create_documents([state_of_the_union])
        docs=docs
    # llm_chain=create_hypothetical_chain()
    model_name = "WhereIsAI/UAE-Large-V1"

    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    index_name = "newsbot"
    # This is a long document we can split up.
    if file_name:
        print('embeddings started')

        docsearch = PineconeVectorStore.from_documents(docs,embedding=embeddings, index_name="newsbot1")

        print('embeddings end')
    else:
          
        docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
    create_cache()
    return docsearch
