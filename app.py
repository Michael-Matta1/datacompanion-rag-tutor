"""
Data Companion AI Tutor - Main Streamlit Application
"""

import streamlit as st
import json
import logging
import warnings
from datetime import datetime
from dotenv import load_dotenv

from src.config import AppConfig
from src.initialization import (
    load_vector_store, load_llm, initialize_memory_system, 
    initialize_chat_session, check_system_status, display_system_status
)
from src.utils import load_course_metadata, get_system_css, format_source_documents
from src.rag_pipeline import process_user_query, update_memory_with_conversation


# Configure warnings and logging
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title=AppConfig.PAGE_TITLE,
    page_icon=AppConfig.PAGE_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state=AppConfig.SIDEBAR_STATE
)

# Apply CSS styling
st.markdown(get_system_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎓 Data Companion AI Tutor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent data science learning with advanced memory systems</p>', unsafe_allow_html=True)

# Initialize system components
with st.spinner("🚀 Initializing AI tutor systems..."):
    vector_store = load_vector_store()
    llm = load_llm()
    course_metadata = load_course_metadata(AppConfig.METADATA_PATH)
    memory_system = initialize_memory_system()

# Display system status
display_system_status(vector_store, llm)

# Stop execution if critical components failed
if not check_system_status(vector_store, llm):
    st.error("⚠️ System initialization failed. Please check your API keys and refresh.")
    st.stop()

# Initialize chat session
initialize_chat_session()

# Chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about data science..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now()
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            # Process user query
            response, source_docs, is_successful = process_user_query(
                vector_store, llm, prompt, memory_system, course_metadata
            )
            
            st.markdown(response)
            
            # Update memory system if successful
            if is_successful:
                update_memory_with_conversation(memory_system, prompt, response, llm)
            
            # Display source documents
            if source_docs:
                formatted_docs = format_source_documents(source_docs)
                
                with st.expander(f"📚 **Course References** ({len(source_docs)} sources)", expanded=False):
                    for doc_info in formatted_docs:
                        st.markdown(f"**📖 Source {doc_info['index']}: {doc_info['source']}**")
                        st.markdown(f"*{doc_info['content']}*")
                        
                        if doc_info['index'] < len(formatted_docs):
                            st.divider()
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": len(source_docs),
                "timestamp": datetime.now()
            })

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if memory_system:
            memory_system.clear_memory()
        st.rerun()
    
    # Download chat button
    if st.session_state.get("messages"):
        # Clean messages before saving
        safe_messages = []
        for msg in st.session_state.messages:
            safe_messages.append({
                "role": str(msg.get("role", "")),
                "content": str(msg.get("content", ""))
            })

        st.download_button(
            "📥 Download Chat",
            json.dumps(safe_messages, indent=2),
            "chat.json",
            mime="application/json",
            use_container_width=True
        )

    # Show memory stats
    if memory_system:
        st.markdown("### 🧠 Memory Status")
        stats = memory_system.get_memory_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Facts", stats['facts'])
        with col2:
            st.metric("Turns", stats['conversation_turns'])

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7F8C8D; font-size: 0.85rem; padding: 1rem;'>
        <strong>Data Companion AI Tutor</strong> • Powered by Google Gemini & Advanced Memory Systems<br>
        Professional data science learning companion
    </div>
    """, 
    unsafe_allow_html=True
)

# Alternative Footer for Hugging Face (Commented)
# st.markdown(
#     """
#     <div style='text-align: center; color: #7F8C8D; font-size: 0.85rem; padding: 1rem;'>
#         <strong>Data Companion AI Tutor</strong> • Powered by Hugging Face & Advanced Memory Systems<br>
#         Professional data science learning companion
#     </div>
#     """, 
#     unsafe_allow_html=True
# )
