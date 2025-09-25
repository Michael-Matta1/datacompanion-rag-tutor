"""
LLM wrapper classes for different AI model providers
"""

import logging
from typing import Any, Optional
from langchain.llms.base import LLM
from pydantic import Field

# Gemini API
import google.generativeai as genai

# Hugging Face API (Alternative - commented out)
# from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class GeminiLLM(LLM):
    """Google Gemini LLM wrapper for LangChain"""
    model: Any = Field(exclude=True)
    model_name: str = "gemini-1.5-flash"

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=600,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "I'm experiencing technical difficulties. Please try your question again."


# Alternative LLM Class: Hugging Face (Commented)
# class HuggingFaceLLM(LLM):
#     """Enhanced Hugging Face LLM wrapper with better error handling"""
#     client: InferenceClient = Field(exclude=True)
#     model_id: str
# 
#     @property
#     def _llm_type(self) -> str:
#         return "huggingface_inference"
# 
#     def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
#         try:
#             response = self.client.chat_completion(
#                 model=self.model_id,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=600,
#                 temperature=0.2,
#                 top_p=0.9
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             logger.error(f"LLM call failed: {e}")
#             return "I'm experiencing technical difficulties. Please try your question again."


def create_gemini_llm(api_key: str) -> Optional[GeminiLLM]:
    """Factory function to create Gemini LLM instance"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return GeminiLLM(model=model)
    except Exception as e:
        logger.error(f"Failed to create Gemini LLM: {e}")
        return None


# Alternative factory function for Hugging Face (Commented)
# def create_huggingface_llm(api_key: str, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> Optional[HuggingFaceLLM]:
#     """Factory function to create Hugging Face LLM instance"""
#     try:
#         client = InferenceClient(token=api_key)
#         return HuggingFaceLLM(client=client, model_id=model_id)
#     except Exception as e:
#         logger.error(f"Failed to create Hugging Face LLM: {e}")
#         return None
