"""
Configuration settings and environment variable management
"""

import os
import streamlit as st
from typing import Optional, Tuple


def get_api_credentials() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Get API credentials from environment variables (Streamlit secrets as fallback)"""
    
    # Check environment variables first 
    google_api_key = os.environ.get("GOOGLE_API_KEY") # Google API Key (Primary)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "datacamp-courses-index")
    
    # Fallback to Streamlit secrets only if env vars are missing and secrets exist
    if not google_api_key and hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    
    if not pinecone_api_key and hasattr(st, 'secrets') and 'PINECONE_API_KEY' in st.secrets:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        if not pinecone_index_name:
            pinecone_index_name = st.secrets.get("PINECONE_INDEX_NAME", "datacamp-courses-index")


    # Hugging Face API Key (Alternative - commented out)
    hf_api_key = None
    # try:
    #     hf_api_key = st.secrets["HUGGINGFACE_API_KEY"]
    # except:
    #     hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
    
    return google_api_key, pinecone_api_key, pinecone_index_name, hf_api_key


class AppConfig:
    """Application configuration constants"""
    
    # Model settings
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_MODEL_NAME = "gemini-1.5-flash"
    HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # Alternative
    
    # Memory settings
    CONVERSATION_WINDOW_SIZE = 12
    MAX_USER_FACTS = 25
    FACT_SIMILARITY_THRESHOLD = 0.82
    FACT_RELEVANCE_THRESHOLD = 0.25
    
    # Vector store settings
    SIMILARITY_SEARCH_K = 5
    
    # LLM generation settings
    LLM_TEMPERATURE = 0.2
    LLM_TOP_P = 0.9
    LLM_MAX_OUTPUT_TOKENS = 600
    
    # Course recommendation settings
    MAX_COURSE_RECOMMENDATIONS = 3
    COURSE_DESCRIPTION_MAX_LENGTH = 200
    
    # UI settings
    PAGE_TITLE = "Data Companion AI Tutor"
    PAGE_ICON = "🎓"
    LAYOUT = "wide"
    SIDEBAR_STATE = "collapsed"
    
    # File paths
    METADATA_PATH = "metadata/courses_info.csv"
    
    # Response validation
    MIN_RESPONSE_LENGTH = 20
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Course matching keywords for intelligent course discovery
    TOPIC_KEYWORDS = {
        'machine learning': [
            'machine learning', 'ml', 'supervised', 'unsupervised', 
            'classification', 'regression', 'clustering', 'algorithms'
        ],
        'python': [
            'python', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'scikit-learn', 'jupyter', 'notebook'
        ],
        'data analysis': [
            'data analysis', 'statistics', 'statistical', 'hypothesis testing', 
            'eda', 'exploratory data analysis', 'data exploration'
        ],
        'data visualization': [
            'visualization', 'plotting', 'charts', 'graphs', 'visual', 
            'dashboard', 'plots'
        ],
        'sql': [
            'sql', 'database', 'queries', 'joins', 'tables', 'postgresql', 
            'mysql', 'sqlite'
        ],
        'r programming': [
            'r programming', 'ggplot', 'dplyr', 'tidyverse', 'r studio', 
            'cran', 'r language'
        ],
        'deep learning': [
            'deep learning', 'neural networks', 'tensorflow', 'keras', 
            'pytorch', 'cnn', 'rnn', 'lstm'
        ],
        'data science': [
            'data science', 'data scientist', 'data mining', 'big data', 
            'analytics'
        ],
        'statistics': [
            'statistics', 'probability', 'statistical inference', 
            'hypothesis', 'correlation', 'regression analysis'
        ],
        'data engineering': [
            'data engineering', 'etl', 'data pipeline', 'apache spark', 
            'hadoop', 'data warehouse'
        ],
        'business analytics': [
            'business analytics', 'business intelligence', 'kpi', 
            'metrics', 'reporting'
        ]
    }
    
    # Course request detection keywords
    COURSE_REQUEST_KEYWORDS = [
        'link', 'course', 'recommend', 'recommendation', 'suggestion', 
        'what should i learn', 'where can i learn', 'course about', 
        'link to', 'course on', 'best course', 'learn about', 'study', 
        'tutorial', 'training', 'class', 'lesson', 'teach me', 
        'show me courses', 'find courses', 'course for', 'curriculum'
    ]
    
    # Memory-related query detection keywords
    MEMORY_KEYWORDS = [
        'remember', 'recall', 'mentioned', 'said before', 'previous', 
        'earlier', 'last time', 'told you', 'background', 'profile',
        'what do you know about me', 'my experience', 'my level',
        'we discussed', 'conversation', 'chat history'
    ]
    
    # Personal query detection keywords
    PERSONAL_KEYWORDS = [
        'my', 'i am', 'i work', 'i study', 'i want', 'i need', 
        'my goal', 'my background', 'my experience', 'my level',
        'about me', 'tell you about', 'personally', 'my situation'
    ]
    
    # Difficulty level mappings for course recommendations
    DIFFICULTY_MAPPINGS = {
        'beginner': ['1', 'beginner', 'basic', 'introduction', 'intro', 'fundamentals'],
        'intermediate': ['2', 'intermediate', 'medium', 'advanced beginner'],
        'advanced': ['3', 'advanced', 'expert', 'professional', 'master']
    }
    
    # Language emoji mappings for course display
    LANGUAGE_EMOJIS = {
        'python': '🐍',
        'r': '📊', 
        'sql': '🗃️',
        'scala': '⚡',
        'shell': '💻',
        'git': '🔀',
        'spreadsheet': '📈',
        'theory': '📚',
        'unknown': '💻'
    }
    
    # Difficulty emoji mappings
    DIFFICULTY_EMOJIS = {
        'beginner': '🟢',
        'intermediate': '🟡', 
        'advanced': '🔴',
        'unknown': '⚪'
    }
    
    # System prompts and messages
    WELCOME_MESSAGE_TEMPLATE = """👋 Welcome to your Data Companion AI Tutor!

I'm your intelligent learning companion with advanced memory capabilities. I can help you master data science through personalized guidance that adapts to your background and learning progress.

**What I can help you with:**
• Python & R programming from basics to advanced
• Data analysis with pandas, NumPy, dplyr, and more  
• Data visualization using matplotlib, seaborn, plotly, ggplot2
• Machine learning algorithms, model building, and evaluation
• Statistics, probability, and hypothesis testing
• Data engineering concepts and best practices

**Smart Features:**
• **Adaptive Memory**: I remember your background, preferences, and progress
• **Contextual Understanding**: I connect concepts across our conversations
• **Personalized Responses**: My answers adapt to your experience level
• **Course Recommendations**: I can suggest specific DataCamp courses with direct links

**Try asking me:**
• "Give me a link about machine learning"
• "What courses should I take for Python data analysis?"
• "I want to learn data visualization"

Feel free to tell me about your background or goals - I'll remember and tailor my responses accordingly. What would you like to explore today?"""
    
    FALLBACK_RESPONSE_TEMPLATE = """I don't have specific information about this topic in the DataCamp course materials. 

I can help you with:
• Python and R programming concepts
• Data analysis and manipulation techniques
• Machine learning algorithms and implementation
• Statistical methods and hypothesis testing
• Data visualization best practices
• Data engineering fundamentals

Could you rephrase your question or ask about a specific data science topic?"""
    
    ERROR_RESPONSE_TEMPLATE = """I'm having trouble generating a complete response. 

Could you try:
- Rephrasing your question more specifically
- Asking about a particular data science concept or technique
- Providing more context about what you're trying to learn

I'm here to help you succeed in your data science journey!"""
    
    # CSS styling template
    CSS_TEMPLATE = """
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2E86C1, #E74C3C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        text-align: center;
        color: #5D6D7E;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .status-indicator {
        position: fixed;
        top: 70px;
        right: 20px;
        background: linear-gradient(135deg, #27AE60, #2ECC71);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        font-weight: 500;
    }
    .api-status {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 500;
        border-left: 4px solid;
    }
    .status-success { 
        background: #D5F4E6; 
        border-left-color: #27AE60; 
        color: #1B5E20; 
    }
    .status-error { 
        background: #FADBD8; 
        border-left-color: #E74C3C; 
        color: #B71C1C; 
    }
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
</style>
"""