import os
import streamlit as st
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from ApiKey import openapi_key  # Make sure this file exists with your API key

os.environ['OPENAI_API_KEY'] = openapi_key

st.set_page_config(page_title="Article Researcher", page_icon=":microscope:", layout="wide")  # Wide layout for better use of space

st.title("Article Researcher")

# Sidebar for URL input
st.sidebar.title("Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", placeholder="Enter URL")  # Placeholder text
    urls.append(url)

analyse_btn = st.sidebar.button("Analyze Articles")  # More descriptive button text

# Main content area
main_placeholder = st.empty()  # Placeholder for query input

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

embeddings = get_embeddings()

if analyse_btn:
    if not any(urls):
        st.sidebar.warning("Please provide at least one URL.")  # Warning in sidebar
    else:
        try:
            with st.spinner("Loading and processing articles..."):
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=200,
                    chunk_overlap=40
                )

                docs = text_splitter.split_documents(data)

                vector_data = FAISS.from_documents(docs, embeddings)
                vector_data.save_local("faiss_store.index")
                st.success("Articles analyzed successfully!")

        except Exception as e:
            st.sidebar.error(f"An error occurred during analysis: {e}")  # Error in sidebar


llm = OpenAI(temperature=0.9, max_tokens=500)

# Query input in the main area
query = main_placeholder.text_input("Enter your question:", placeholder="e.g., What are the latest advancements in lung cancer treatment?")  # Placeholder text

if query:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching for answer..."):
                if os.path.exists("faiss_store.index"):
                    vectorstore = FAISS.load_local("faiss_store.index", embeddings, allow_dangerous_deserialization=True)

                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=llm, retriever=vectorstore.as_retriever(), verbose=False  # verbose=False for cleaner output
                    )

                    result = chain({"question": query}, return_only_outputs=True)

                    st.header("Answer")
                    st.write(result["answer"])

                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        for source in sources.split("\n"):
                            st.write(source)

                else:
                    st.error("Index file not found. Please analyze the articles first.")

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")