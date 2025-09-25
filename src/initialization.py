"""
System initialization functions for loading models and vector stores
"""

import os
import logging
import streamlit as st
from typing import Optional

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from .llm_wrappers import create_gemini_llm
from .memory_system import IntelligentMemorySystem
from .config import AppConfig, get_api_credentials
from .utils import create_welcome_message

logger = logging.getLogger(__name__)


@st.cache_resource
def load_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Load optimized embeddings model"""
    try:
        return HuggingFaceEmbeddings(
            model_name=AppConfig.EMBEDDINGS_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        st.error("Failed to load embeddings model. Please refresh the page.")
        return None


@st.cache_resource
def load_vector_store() -> Optional[PineconeVectorStore]:
    """Initialize Pinecone vector store"""
    try:
        # Get API credentials
        _, pinecone_api_key, pinecone_index_name, _ = get_api_credentials()
        
        if not pinecone_api_key:
            st.error("🔑 Pinecone API key required")
            return None
            
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        
        embeddings = load_embeddings()
        if embeddings is None:
            return None
            
        return PineconeVectorStore(index=index, embedding=embeddings)
        
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")
        st.error(f"Knowledge base connection failed: {str(e)}")
        return None


@st.cache_resource
def load_llm():
    """Load LLM - Primary: Gemini, Alternative: Hugging Face (commented)"""
    
    # Primary Option: Google Gemini
    google_api_key, _, _, _ = get_api_credentials()
    
    if not google_api_key:
        st.error("🔑 Google API key required")
        return None

    llm = create_gemini_llm(google_api_key)
    if llm is None:
        st.error("AI model connection failed")
        return None
        
    return llm
    
    # Alternative Option: Hugging Face (Commented)
    # _, _, _, hf_api_key = get_api_credentials()
    # 
    # if not hf_api_key:
    #     st.error("🔑 Hugging Face API key required")
    #     return None
    # 
    # llm = create_huggingface_llm(hf_api_key, AppConfig.HF_MODEL_ID)
    # if llm is None:
    #     st.error("AI model connection failed")
    #     return None
    #     
    # return llm


def initialize_memory_system() -> Optional[IntelligentMemorySystem]:
    """Initialize intelligent memory system"""
    if "intelligent_memory" not in st.session_state:
        embeddings = load_embeddings()
        if embeddings:
            st.session_state.intelligent_memory = IntelligentMemorySystem(embeddings)
        else:
            st.session_state.intelligent_memory = None
    return st.session_state.intelligent_memory


def initialize_chat_session():
    """Initialize chat session with welcome message"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = create_welcome_message()
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_message,
            "timestamp": st.session_state.get('session_start_time', None)
        })


def check_system_status(vector_store, llm) -> bool:
    """Check if all critical components are loaded successfully"""
    if not vector_store:
        st.error("⚠️ Knowledge base failed to initialize")
        return False
        
    if not llm:
        st.error("⚠️ AI model failed to initialize")
        return False
        
    return True


def display_system_status(vector_store, llm):
    """Display system status indicators"""
    col1, col2 = st.columns(2)
    
    with col1:
        if vector_store:
            st.markdown('<div class="api-status status-success">✅ Knowledge Base Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status status-error">❌ Knowledge Base Failed</div>', unsafe_allow_html=True)

    with col2:
        if llm:
            # Primary: Gemini
            st.markdown('<div class="api-status status-success">✅ AI Model Ready (Gemini)</div>', unsafe_allow_html=True)
            # Alternative: Hugging Face (Commented)
            # st.markdown('<div class="api-status status-success">✅ AI Model Ready (Hugging Face)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status status-error">❌ AI Model Failed</div>', unsafe_allow_html=True)
