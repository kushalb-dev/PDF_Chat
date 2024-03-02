import streamlit as st
from langchain_astradb import AstraDBVectorStore
from PyPDF2 import PdfReader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from dotenv import load_dotenv
import os

def pdf_reader_splitter(pdf) -> list[str]:
    reader = PdfReader(pdf)
    
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        
        if content:
            raw_text += content

    splitter = CharacterTextSplitter(
        separator="\n",
        is_separator_regex=False,
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = splitter.split_text(raw_text)
    return texts

def main() -> None:    
    llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],temperature=0.2)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="test",
    token=os.environ['ASTRADB_TOKEN'],
    api_endpoint=os.environ['ASTRADB_API_ENDPOINT']
    )
    
    st.header("PDF Chatter")
    
    col1, col2 = st.columns(2)
    with col1:
        pdf = st.file_uploader(accept_multiple_files=False, type='pdf', label="Upload PDF here...")

        if pdf:
            chunks = pdf_reader_splitter(pdf)
            vstore.add_texts(chunks)
    
        vstore_wrapped = VectorStoreIndexWrapper(vectorstore=vstore)
    
    with col2:        
        input = st.text_input(label="Write your query here...", key="pdf-question")
            
        if input:
            answer = vstore_wrapped.query(input, llm=llm)
            st.text_area(label="Generating Answer", value=answer.strip())

if __name__ == "__main__":
    load_dotenv()
    main()