import streamlit as st
from datetime import date
from webscrape import *
import os
import os
import cohere
import langchain
from langchain_core.messages import HumanMessage
os.environ["COHERE_API_KEY"]='1WDphHnJYzXRcm2EjDcvyqXRnKRG6n83XxX7LPFx'
os.environ['PINECONE_API_KEY']='ac2836be-5765-4e89-8d47-70ab0771e347'#'6dbebefb-e722-4241-8041-00f56ca935ca'
os.environ['PINECONE_ENV']='gcp-starter'
os.environ['QDRANT_API_KEY']='B2p7WN_t2TIpugdRgeZ-S5ApOPZ-VigWZZxhxDE036aBbATU_mpx1g'
os.environ['GOOGLE_API_KEY']='AIzaSyAUggwhrE0LoTBDWrfeU6kxQuxA0FP6eCk'
os.environ['APIFY_API_TOKEN']='apify_api_K90vlEcLcKMx43KED0DpKQuxz2cTUr2CXPtv'
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
from langchain_community.llms import Ollama

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA

import os
# from dbase import *
from vect_embed import *
vectDB=vector_embedding()
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
os.environ['HUGGINGFACEHUB_API_TOKEN'] ='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'
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.llms import HuggingFaceHub

def chtreply(query):
    langchain.debug=True
    print('start')
    from pprint import pprint
    from langchain_cohere import ChatCohere, CohereRagRetriever
    from langchain.llms import Cohere

    # User query we will use for the generation
    from langchain.llms import HuggingFaceHub
    # repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1'
    # llm=Ollama(model="llama2")

    # Use Cohere's RAG retriever with Cohere Connectors to generate an answer.
    # Cohere provides exact citations for the sources it used.
    llm = ChatCohere()
    llm = Cohere(model="command", max_tokens=1024, temperature=0.75)
    '''
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectDB.as_retriever(), llm=llm#must set qdrant embeddings as base embedding and not hypothetical embeddings else it will ask fr paid version
    )
    '''
    retriever=vectDB.as_retriever()
    # system_template = """Forget that you don't know the current news now assume You are an helpful news chat bot who uses only the following context give below to answer the question and give your intitution about it also .Strictly answer only from the news below and don't try to answer from your knowledge.Answer in english.
    # {context}
    
    # """
    # system_template = """You are a news chatbot. Use the information provided below to help answer the user's questions. Base your answers solely on the news content and share any relevant details from it. Please respond in English.
    system_template=""" 
    {context}
    """

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(llm,retriever,combine_docs_chain_kwargs={"prompt": qa_prompt})

    latest_cht=""
    resp=qa.run({'question':query,'chat_history':latest_cht})
    # insert_data(query,resp)

    return resp

st.set_page_config(
    page_title="NewsBot360",
    page_icon=":newspaper:",
    layout="wide"
)
st.title("NewsBot360")

today = str(date.today())+'.txt'
print(today)
files = os.listdir('files')
if today not in files:
    with st.spinner("Scraping data..."):
            scrape('files/'+today)
    vector_embedding("files\\"+today)
vector_embedding("files\\"+today)
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.text_input("Message NewsBot360..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chtreply(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        st.markdown(response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})




