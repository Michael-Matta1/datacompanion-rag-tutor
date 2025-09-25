"""
Intelligent memory system combining LangChain memory with semantic extraction
"""

import json
import re
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from langchain.memory import ConversationBufferWindowMemory

logger = logging.getLogger(__name__)


class IntelligentMemorySystem:
    """Advanced memory system combining LangChain memory with intelligent semantic extraction"""
    
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        
        # LangChain conversation memory
        self.conversation_memory = ConversationBufferWindowMemory(
            k=12,  # Keep last 12 conversation turns
            memory_key="chat_history",
            return_messages=True,
            input_key="human_input",
            output_key="output"
        )
        
        # Semantic knowledge extraction
        self.user_facts = []
        self.topic_knowledge = {}
        self.context_embeddings = []
        
    def classify_query_intent(self, llm, query: str) -> Dict[str, Any]:
        """Intelligently classify query intent without hardcoded patterns"""
        classification_prompt = f"""Analyze this user query and classify its intent. Return a JSON object with these fields:
- "is_personal": true/false (asks about user's personal info, preferences, or previous statements)
- "is_memory_related": true/false (references previous conversations or asks for recall)
- "is_learning_focused": true/false (asks about data science concepts or learning)
- "confidence_score": 0.0-1.0 (confidence in classification)
- "main_intent": one of ["personal_recall", "concept_learning", "clarification", "general"]

Query: "{query}"

Return only valid JSON:"""
        
        try:
            response = llm.invoke(classification_prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
        
        # Fallback classification
        return {
            "is_personal": False,
            "is_memory_related": False,
            "is_learning_focused": True,
            "confidence_score": 0.3,
            "main_intent": "concept_learning"
        }

    def extract_semantic_information(self, llm, user_input: str, assistant_response: str):
        """Extract semantic information using advanced prompting"""
        extraction_prompt = f"""Analyze this conversation turn and extract structured information. Return JSON with:
- "user_facts": array of factual statements about the user (background, goals, preferences)
- "learning_context": object with user's experience level and interests
- "key_topics": array of main data science topics discussed
- "user_questions_type": categorize the user's question style

Conversation:
User: {user_input}
Assistant: {assistant_response}

Extract meaningful information and return only valid JSON:"""
        
        try:
            response = llm.invoke(extraction_prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                return extracted
        except Exception as e:
            logger.error(f"Semantic extraction failed: {e}")
        
        return {"user_facts": [], "learning_context": {}, "key_topics": [], "user_questions_type": "general"}

    def add_conversation_turn(self, user_input: str, assistant_response: str, llm=None):
        """Add conversation with intelligent processing"""
        # Update LangChain memory
        self.conversation_memory.save_context(
            {"human_input": user_input},
            {"output": assistant_response}
        )
        
        # Extract semantic information
        if llm:
            extracted_info = self.extract_semantic_information(llm, user_input, assistant_response)
            
            # Process user facts
            for fact in extracted_info.get("user_facts", []):
                if fact and len(fact) > 5:
                    self._add_user_fact(fact)
            
            # Update learning context
            learning_context = extracted_info.get("learning_context", {})
            if learning_context:
                self._update_learning_context(learning_context)
            
            # Track topics
            for topic in extracted_info.get("key_topics", []):
                if topic:
                    self._track_topic_interest(topic)

    def _add_user_fact(self, fact_text: str):
        """Add user fact with intelligent deduplication"""
        try:
            fact_embedding = self._get_embedding(fact_text)
            
            # Check semantic similarity with existing facts
            for existing_fact in self.user_facts:
                similarity = cosine_similarity(
                    [fact_embedding], 
                    [existing_fact['embedding']]
                )[0][0]
                
                if similarity > 0.82:  # High semantic similarity threshold
                    return  # Skip duplicate
            
            # Add new fact
            fact_entry = {
                'content': fact_text,
                'timestamp': datetime.now(),
                'embedding': fact_embedding,
                'relevance_score': 1.0
            }
            
            self.user_facts.append(fact_entry)
            
            # Maintain reasonable size
            if len(self.user_facts) > 25:
                # Keep most recent and most relevant facts
                sorted_facts = sorted(self.user_facts, 
                                    key=lambda x: (x['relevance_score'], x['timestamp']), 
                                    reverse=True)
                self.user_facts = sorted_facts[:20]
                
        except Exception as e:
            logger.error(f"Error adding user fact: {e}")

    def _update_learning_context(self, context: Dict):
        """Update user's learning context"""
        if 'experience_level' in context:
            self.topic_knowledge['experience_level'] = context['experience_level']
        if 'interests' in context:
            current_interests = self.topic_knowledge.get('interests', [])
            new_interests = context['interests'] if isinstance(context['interests'], list) else [context['interests']]
            self.topic_knowledge['interests'] = list(set(current_interests + new_interests))

    def _track_topic_interest(self, topic: str):
        """Track user's topic interests"""
        if 'topic_frequency' not in self.topic_knowledge:
            self.topic_knowledge['topic_frequency'] = {}
        
        self.topic_knowledge['topic_frequency'][topic] = self.topic_knowledge['topic_frequency'].get(topic, 0) + 1

    def _get_embedding(self, text: str):
        """Get embedding with error handling"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.zeros(384)

    def get_relevant_memory_context(self, query: str, llm=None):
        """Get contextually relevant memory for the query"""
        context = {
            'conversation_history': [],
            'relevant_facts': [],
            'learning_context': {},
            'query_intent': {}
        }
        
        # Get query intent classification
        if llm:
            context['query_intent'] = self.classify_query_intent(llm, query)
        
        # Get conversation history from LangChain
        try:
            memory_vars = self.conversation_memory.load_memory_variables({})
            chat_history = memory_vars.get('chat_history', [])
            
            # Format recent conversation turns
            recent_turns = []
            for i in range(0, min(len(chat_history), 8), 2):  # Last 4 turns
                if i + 1 < len(chat_history):
                    user_msg = chat_history[i].content if hasattr(chat_history[i], 'content') else str(chat_history[i])
                    assistant_msg = chat_history[i + 1].content if hasattr(chat_history[i + 1], 'content') else str(chat_history[i + 1])
                    
                    recent_turns.append({
                        'user': user_msg[:200],
                        'assistant': assistant_msg[:200]
                    })
            
            context['conversation_history'] = recent_turns
            
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
        
        # Get semantically relevant user facts
        if self.user_facts:
            context['relevant_facts'] = self._find_relevant_facts(query, top_k=3)
        
        # Add learning context
        context['learning_context'] = self.topic_knowledge
        
        return context

    def _find_relevant_facts(self, query: str, top_k: int = 3) -> List[str]:
        """Find semantically relevant user facts"""
        if not self.user_facts:
            return []
        
        try:
            query_embedding = self._get_embedding(query)
            
            similarities = []
            for fact in self.user_facts:
                sim = cosine_similarity([query_embedding], [fact['embedding']])[0][0]
                similarities.append((fact, sim))
            
            # Sort by similarity and return top facts
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            relevant_facts = []
            for fact, sim in similarities[:top_k]:
                if sim > 0.25:  # Lower threshold for more inclusive recall
                    relevant_facts.append(fact['content'])
            
            return relevant_facts
            
        except Exception as e:
            logger.error(f"Error finding relevant facts: {e}")
            return []

    def format_context_for_prompt(self, query: str, llm=None) -> str:
        """Format memory context for LLM prompt"""
        context = self.get_relevant_memory_context(query, llm)
        
        formatted_context = ""
        
        # Add query intent awareness
        intent = context.get('query_intent', {})
        if intent.get('is_personal') or intent.get('is_memory_related'):
            formatted_context += "[User is asking about personal/previous information]\n"
        
        # Add conversation history
        if context['conversation_history']:
            formatted_context += "Recent conversation context:\n"
            for turn in context['conversation_history'][-2:]:  # Last 2 turns
                formatted_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
        
        # Add relevant user facts
        if context['relevant_facts']:
            facts_text = " | ".join(context['relevant_facts'][:2])
            formatted_context += f"User background: {facts_text}\n\n"
        
        # Add learning context
        learning_ctx = context['learning_context']
        if learning_ctx.get('experience_level'):
            formatted_context += f"User experience level: {learning_ctx['experience_level']}\n"
        
        if learning_ctx.get('interests'):
            interests = ", ".join(learning_ctx['interests'][:3])
            formatted_context += f"User interests: {interests}\n"
        
        return formatted_context

    def clear_memory(self):
        """Clear all memory systems"""
        self.conversation_memory.clear()
        self.user_facts = []
        self.topic_knowledge = {}
        self.context_embeddings = []

    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics for debugging/UI purposes"""
        try:
            conv_vars = self.conversation_memory.load_memory_variables({})
            turns = len(conv_vars.get('chat_history', [])) // 2
        except:
            turns = 0
            
        return {
            'facts': len(self.user_facts),
            'conversation_turns': turns,
            'topics_tracked': len(self.topic_knowledge.get('topic_frequency', {}))
        }
