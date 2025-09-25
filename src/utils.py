"""
Utility functions for course recommendations, metadata loading, and formatting
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_course_metadata(metadata_path: str = "metadata/courses_info.csv") -> pd.DataFrame:
    """Load course metadata with enhanced processing"""
    try:
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            
            # Clean and preprocess the data
            if not df.empty:
                # Create search-friendly text combining multiple fields
                df['search_text'] = (
                    df['title'].fillna('') + ' ' + 
                    df['description'].fillna('') + ' ' + 
                    df['short_description'].fillna('') + ' ' +
                    df['content_area'].fillna('') + ' ' +
                    df['programming_language'].fillna('')
                ).str.lower()
                
                # Clean difficulty level
                df['difficulty_level'] = df['difficulty_level'].fillna('Unknown')
                
                # Ensure required columns exist
                required_cols = ['title', 'link', 'programming_language', 'difficulty_level', 'content_area']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 'Unknown'
                
                logger.info(f"Loaded {len(df)} courses from metadata")
            
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Metadata loading failed: {e}")
        return pd.DataFrame()


def find_relevant_courses(query: str, course_metadata: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """Find relevant courses based on query using intelligent matching"""
    if course_metadata.empty:
        return []
    
    try:
        # Extract key terms and topics from query
        query_lower = query.lower()
        
        # Define topic keywords for better matching
        topic_keywords = {
            'machine learning': ['machine learning', 'ml', 'supervised', 'unsupervised', 'classification', 'regression', 'clustering'],
            'python': ['python', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn'],
            'data analysis': ['data analysis', 'statistics', 'statistical', 'hypothesis testing', 'eda'],
            'data visualization': ['visualization', 'plotting', 'charts', 'graphs', 'visual'],
            'sql': ['sql', 'database', 'queries', 'joins'],
            'r programming': ['r programming', 'ggplot', 'dplyr', 'tidyverse'],
            'deep learning': ['deep learning', 'neural networks', 'tensorflow', 'keras', 'pytorch'],
            'data science': ['data science', 'data scientist'],
            'statistics': ['statistics', 'probability', 'statistical inference']
        }
        
        # Score courses based on relevance
        relevant_courses = []
        
        for idx, course in course_metadata.iterrows():
            score = 0
            course_search_text = course.get('search_text', '').lower()
            
            # Direct keyword matching
            for topic, keywords in topic_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    if any(keyword in course_search_text for keyword in keywords):
                        score += 3
                        
            # Additional scoring factors
            if any(word in course_search_text for word in query_lower.split() if len(word) > 3):
                score += 1
                
            # Boost score for exact matches in title
            if any(word in course.get('title', '').lower() for word in query_lower.split() if len(word) > 3):
                score += 2
                
            # Ensure difficulty is properly formatted
            difficulty_raw = course.get('difficulty_level', 'Unknown')
            difficulty_clean = str(difficulty_raw).strip()
            
            # Map numeric difficulty to descriptive text
            if difficulty_clean in ['1', '1.0']:
                difficulty_display = 'Beginner'
            elif difficulty_clean in ['2', '2.0']:
                difficulty_display = 'Intermediate' 
            elif difficulty_clean in ['3', '3.0']:
                difficulty_display = 'Advanced'
            else:
                difficulty_display = difficulty_clean
            
            if score > 0:  # Only if course is relevant
                relevant_courses.append({
                    'course': course,
                    'score': score,
                    'title': course.get('title', 'Unknown Course'),
                    'link': course.get('link', '#'),
                    'language': course.get('programming_language', 'Unknown'),
                    'difficulty': difficulty_clean,  
                    'difficulty_display': difficulty_display,  # For display
                    'description': course.get('short_description', '')[:200] + '...' if len(course.get('short_description', '')) > 200 else course.get('short_description', ''),
                    'content_area': course.get('content_area', 'Unknown')
                })
        
        # Sort by score and return top results
        relevant_courses.sort(key=lambda x: x['score'], reverse=True)
        return relevant_courses[:top_k]
        
    except Exception as e:
        logger.error(f"Error finding relevant courses: {e}")
        return []



def format_course_recommendations(courses: List[Dict]) -> str:
    """Format course recommendations into a readable response with fixed difficulty emoji mapping"""
    if not courses:
        return ""
        
    recommendations = "\n\n**📚 Recommended DataCamp Courses:**\n\n"
    
    for i, course in enumerate(courses, 1):
        # Fixed difficulty emoji mapping
        difficulty_value = str(course['difficulty']).lower().strip()
        
        # Check for numeric values first, then text values
        if difficulty_value in ['1', '1.0'] or 'beginner' in difficulty_value or 'basic' in difficulty_value:
            difficulty_emoji = "🟢"
            difficulty_display = "Beginner"
        elif difficulty_value in ['2', '2.0'] or 'intermediate' in difficulty_value:
            difficulty_emoji = "🟡" 
            difficulty_display = "Intermediate"
        elif difficulty_value in ['3', '3.0'] or 'advanced' in difficulty_value:
            difficulty_emoji = "🔴"
            difficulty_display = "Advanced"
        else:
            difficulty_emoji = "⚪"
            difficulty_display = "Unknown"
        
        # Language emoji mapping
        language_lower = str(course['language']).lower()
        if 'python' in language_lower:
            language_emoji = "🐍"
        elif language_lower in ['r', 'r programming']:
            language_emoji = "📊"
        elif 'sql' in language_lower:
            language_emoji = "🗃️"
        elif 'scala' in language_lower:
            language_emoji = "⚡"
        elif 'shell' in language_lower or 'bash' in language_lower:
            language_emoji = "💻"
        elif 'git' in language_lower:
            language_emoji = "🔀"
        elif 'spreadsheet' in language_lower:
            language_emoji = "📈"
        else:
            language_emoji = "💻"
        
        recommendations += f"**{i}. {course['title']}**\n"
        recommendations += f"{language_emoji} {course['language']} • {difficulty_emoji} {difficulty_display} • {course['content_area']}\n"
        recommendations += f"*{course['description']}*\n"
        recommendations += f"🔗 **[Start Course]({course['link']})**\n\n"
    
    return recommendations


def check_course_request(question: str) -> bool:
    """Check if user is asking for course links or recommendations"""
    course_request_keywords = [
        'link', 'course', 'recommend', 'suggestion', 'what should i learn', 
        'where can i learn', 'course about', 'link to', 'course on',
        'best course', 'learn about', 'study', 'tutorial'
    ]
    
    return any(keyword in question.lower() for keyword in course_request_keywords)


def create_welcome_message() -> str:
    """Create the welcome message for new users"""
    return """👋 Welcome to your Data Companion AI Tutor!

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


def create_fallback_response() -> str:
    """Create a fallback response when no specific information is available"""
    return """I don't have specific information about this topic in the DataCamp course materials. 

I can help you with:
• Python and R programming concepts
• Data analysis and manipulation techniques
• Machine learning algorithms and implementation
• Statistical methods and hypothesis testing
• Data visualization best practices
• Data engineering fundamentals

Could you rephrase your question or ask about a specific data science topic?"""


def format_source_documents(docs: List[Any]) -> List[Dict[str, str]]:
    """Format source documents for display"""
    formatted_docs = []
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'DataCamp Course')
        content_preview = doc.page_content[:350].replace('\n', ' ')
        
        formatted_docs.append({
            'index': i,
            'source': source,
            'content': f"{content_preview}..."
        })
    
    return formatted_docs


def validate_response(response: str, min_length: int = 20) -> bool:
    """Validate if the response meets quality standards"""
    return response and len(response.strip()) >= min_length


def create_error_response() -> str:
    """Create a generic error response"""
    return """I'm having trouble generating a complete response. 

Could you try:
- Rephrasing your question more specifically
- Asking about a particular data science concept or technique
- Providing more context about what you're trying to learn

I'm here to help you succeed in your data science journey!"""


def get_system_css() -> str:
    """Return the CSS styling for the Streamlit app"""
    return """
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
