import streamlit as st
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import QueryBundle, get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from typing import List
import os   
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords
from llama_index.core import SimpleDirectoryReader

nltk.download('stopwords')
load_dotenv()

st.title("Indian Constitution Chatbot")
st.write("Ask me questions about the Indian Constitution!")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to load documents and set up the query engine
def initialize_query_engine():
    if 'documents' not in st.session_state:
        documents = SimpleDirectoryReader('data').load_data()
        for doc in documents:
            doc.text = preprocess_text(doc.text)
        
        st.session_state['documents'] = documents
    else:
        documents = st.session_state['documents']
    
    if 'nodes' not in st.session_state:
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        st.session_state['nodes'] = nodes
    else:
        nodes = st.session_state['nodes']

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Initialize embeddings and LLM with the provided API key only once
    if 'embed_model' not in st.session_state:
        st.session_state['embed_model'] = GeminiEmbedding(
            model_name="models/embedding-001", api_key=GEMINI_API_KEY
        )
        Settings.embed_model = st.session_state['embed_model']
        
    if 'llm' not in st.session_state:
        st.session_state['llm'] = Gemini(api_key=GEMINI_API_KEY)
        Settings.llm = st.session_state['llm']

    # Set up the storage context and add documents
    if 'storage_context' not in st.session_state:
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        st.session_state['storage_context'] = storage_context
    else:
        storage_context = st.session_state['storage_context']

    # Create indexes
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    # Set up retrievers
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, top_k=5)

    # Define a custom retriever to combine vector and keyword retrievers
    class CustomRetriever(BaseRetriever):
        def __init__(self, vector_retriever: VectorIndexRetriever, keyword_retriever: KeywordTableSimpleRetriever, mode: str = "AND") -> None:
            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            if mode not in ("AND", "OR"):
                raise ValueError("Invalid mode.")
            self._mode = mode
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

            # Merge nodes based on selected mode
            vector_ids = {n.node.node_id for n in vector_nodes}
            keyword_ids = {n.node.node_id for n in keyword_nodes}

            combined_dict = {n.node.node_id: n for n in vector_nodes}
            combined_dict.update({n.node.node_id: n for n in keyword_nodes})

            retrieve_ids = (
                vector_ids.intersection(keyword_ids) if self._mode == "AND" 
                else vector_ids.union(keyword_ids)
            )
            
            retrieve_nodes = [combined_dict[r_id] for r_id in retrieve_ids]
            return retrieve_nodes
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)


    response_synthesizer = get_response_synthesizer()
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )
    return custom_query_engine

if 'custom_query_engine' not in st.session_state:
    with st.spinner("Hi! Let me read the constitution📖 first, please wait for a few minutes⌛..."):
        # Initialize the query engine and store it in session state
        st.session_state['custom_query_engine'] = initialize_query_engine()
    st.success("Document processing completed! Feel free to ask questions now.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Hmm! Good question, looking for your answer👀🔍..."):
        custom_query_engine = st.session_state['custom_query_engine']
        result = custom_query_engine.query(query)
        st.write("Response:")
        st.write(result.response) 
