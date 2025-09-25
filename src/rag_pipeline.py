"""
Enhanced RAG (Retrieval-Augmented Generation) pipeline with intelligent memory integration
"""

import logging
from typing import List, Tuple, Optional, Any
import pandas as pd

from .utils import (
    check_course_request, find_relevant_courses, format_course_recommendations,
    create_fallback_response, validate_response, create_error_response
)
from .memory_system import IntelligentMemorySystem

logger = logging.getLogger(__name__)


def enhanced_rag_pipeline(
    vector_store, 
    llm, 
    question: str, 
    memory_system: Optional[IntelligentMemorySystem], 
    course_metadata: pd.DataFrame
) -> Tuple[str, List[Any]]:
    """Advanced RAG pipeline with intelligent memory integration and course recommendations"""
    try:
        # Get memory context
        memory_context = memory_system.format_context_for_prompt(question, llm) if memory_system else ""
        
        # Get intent classification
        query_intent = memory_system.get_relevant_memory_context(question, llm).get('query_intent', {}) if memory_system else {}
        
        # Check if user is asking for course links or recommendations
        is_course_request = check_course_request(question)
        
        # Handle memory-focused queries intelligently
        if query_intent.get('is_personal', False) or query_intent.get('is_memory_related', False):
            if memory_context.strip():
                memory_prompt = f"""You are an experienced data science tutor with perfect memory of our conversations.

{memory_context}

Current question: {question}

Provide a helpful response based on our conversation history and what you know about the user. If you don't have the specific information requested, acknowledge what you do remember and ask for clarification in a friendly manner."""
                
                try:
                    response = llm.invoke(memory_prompt)
                    return response, []
                except Exception as e:
                    logger.error(f"Memory response failed: {e}")
                    return "I remember our conversations, but I'm having trouble accessing that specific information right now. Could you provide more context?", []
        
        # Find relevant courses if this is a course-related query
        relevant_courses = []
        course_recommendations = ""
        
        if is_course_request and not course_metadata.empty:
            relevant_courses = find_relevant_courses(question, course_metadata, top_k=3)
            if relevant_courses:
                course_recommendations = format_course_recommendations(relevant_courses)
        
        # Standard RAG for knowledge queries
        try:
            docs = vector_store.similarity_search(question, k=5)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            docs = []
        
        if not docs and not course_recommendations:
            fallback_response = create_fallback_response()
            return fallback_response, []
        
        # Enhanced prompt with memory integration and course awareness
        course_content = "\n\n".join([
            f"Course: {doc.metadata.get('source', 'DataCamp')}\n{doc.page_content[:600]}" 
            for doc in docs
        ])
        
        enhanced_prompt = f"""You are a senior data scientist and expert tutor specializing in DataCamp courses. Provide clear, educational responses tailored to the user's background and learning needs.

{memory_context}

DataCamp Course Materials:
{course_content}

User Question: {question}

Guidelines:
- Provide clear, step-by-step explanations
- Use examples from the course materials when relevant
- Adapt your response to the user's experience level if known
- Be encouraging and supportive
- If course materials don't cover the topic, acknowledge limitations clearly
- Focus on practical application and understanding
- If the user is asking for course recommendations or links, be sure to mention that you have specific course recommendations available

Response:"""
        
        try:
            response = llm.invoke(enhanced_prompt)
            
            # Append course recommendations if found
            if course_recommendations:
                response += "\n\n" + course_recommendations
            
            return response, docs
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # If LLM fails but we have course recommendations, return them
            if course_recommendations:
                return f"I found some relevant courses for you:\n\n{course_recommendations}", []
            
            return "I'm experiencing technical difficulties. Please try rephrasing your question.", []
            
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        return "I encountered an error processing your question. Please try again.", []


def process_user_query(
    vector_store,
    llm,
    query: str,
    memory_system: Optional[IntelligentMemorySystem],
    course_metadata: pd.DataFrame
) -> Tuple[str, List[Any], bool]:
    """
    Process user query and return response, source documents, and success status
    
    Returns:
        Tuple of (response_text, source_documents, is_successful)
    """
    try:
        # Generate enhanced response using RAG pipeline
        response, source_docs = enhanced_rag_pipeline(
            vector_store, llm, query, memory_system, course_metadata
        )
        
        # Validate response quality
        if not validate_response(response):
            response = create_error_response()
            return response, [], False
        
        return response, source_docs, True
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        error_response = "I encountered a technical issue. Please try your question again."
        return error_response, [], False


def update_memory_with_conversation(
    memory_system: Optional[IntelligentMemorySystem],
    user_input: str,
    assistant_response: str,
    llm
):
    """Update memory system with the conversation turn"""
    if memory_system:
        try:
            memory_system.add_conversation_turn(user_input, assistant_response, llm)
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
