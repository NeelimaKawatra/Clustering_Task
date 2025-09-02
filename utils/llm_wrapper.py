# utils/llm_wrapper.py - LLM Abstraction Layer

import os
import json
from typing import Dict, Any, Optional, List
import streamlit as st

class LLMWrapper:
    """Wrapper for accessing different LLM APIs without direct API calls"""
    
    def __init__(self):
        self.provider = None
        self.client = None
        self.config = {}
        self.initialized = False
    
    def initLLM(self, provider: str = "openai", config: Dict[str, Any] = None) -> bool:
        """Initialize LLM connection
        
        Args:
            provider: "openai", "anthropic", "local", "mock"
            config: Provider-specific configuration
        
        Returns:
            bool: Success status
        """
        try:
            self.provider = provider.lower()
            self.config = config or {}
            
            if self.provider == "openai":
                return self._init_openai()
            elif self.provider == "anthropic":
                return self._init_anthropic()
            elif self.provider == "local":
                return self._init_local()
            elif self.provider == "mock":
                return self._init_mock()
            else:
                st.error(f"Unsupported LLM provider: {provider}")
                return False
                
        except Exception as e:
            st.error(f"LLM initialization failed: {e}")
            return False
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client"""
        try:
            import openai
            
            # Get API key from config or environment
            api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            self.initialized = True
            return True
            
        except ImportError:
            st.error("OpenAI package not installed. Run: pip install openai")
            return False
    
    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client"""
        try:
            import anthropic
            
            api_key = self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                st.error("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
                return False
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.initialized = True
            return True
            
        except ImportError:
            st.error("Anthropic package not installed. Run: pip install anthropic")
            return False
    
    def _init_local(self) -> bool:
        """Initialize local LLM (placeholder for local models)"""
        # Placeholder for local models like Ollama, Llama.cpp, etc.
        st.info("Local LLM support coming soon")
        self.initialized = True
        return True
    
    def _init_mock(self) -> bool:
        """Initialize mock LLM for testing"""
        self.initialized = True
        return True
    
    def callLLM(self, prompt: str, context: Dict[str, Any] = None, 
                temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """Generic LLM call function
        
        Args:
            prompt: The input prompt
            context: Additional context (clusters, settings, etc.)
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            
        Returns:
            str: LLM response or None if failed
        """
        if not self.initialized:
            st.error("LLM not initialized. Call initLLM() first.")
            return None
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, context, temperature, max_tokens)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt, context, temperature, max_tokens)
            elif self.provider == "local":
                return self._call_local(prompt, context, temperature, max_tokens)
            elif self.provider == "mock":
                return self._call_mock(prompt, context, temperature, max_tokens)
            else:
                return None
                
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            return None
    
    def _call_openai(self, prompt: str, context: Dict[str, Any], 
                     temperature: float, max_tokens: int) -> str:
        """Call OpenAI API"""
        
        # Build system message with context
        system_message = self._build_system_message(context)
        
        response = self.client.chat.completions.create(
            model=self.config.get('model', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, context: Dict[str, Any], 
                        temperature: float, max_tokens: int) -> str:
        """Call Anthropic API"""
        
        system_message = self._build_system_message(context)
        full_prompt = f"{system_message}\n\nUser: {prompt}"
        
        response = self.client.completions.create(
            model=self.config.get('model', 'claude-3-sonnet-20240229'),
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.completion
    
    def _call_local(self, prompt: str, context: Dict[str, Any], 
                    temperature: float, max_tokens: int) -> str:
        """Call local LLM"""
        # Placeholder - would integrate with Ollama, etc.
        return "Local LLM response (not implemented)"
    
    def _call_mock(self, prompt: str, context: Dict[str, Any], 
                   temperature: float, max_tokens: int) -> str:
        """Mock LLM for testing"""
        
        # Simple pattern-based responses for testing
        p = prompt.lower()
        
        if "suggest" in p and "name" in p:
            return "Based on the cluster content, I suggest the name 'Topic Analysis' for this cluster."
        elif "move" in p and "cluster" in p:
            return "I recommend moving the shorter texts to the outliers cluster as they may not contain enough information for proper categorization."
        elif "improve" in p:
            return "To improve your clusters, consider: 1) Merging small clusters with similar themes, 2) Moving obvious outliers to a separate group, 3) Renaming clusters with descriptive labels."
        else:
            return f"I understand you want help with: '{prompt}'. Here's a general suggestion for improving your clustering results."
    
    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build system message with clustering context"""
        
        base_message = """You are an expert in text clustering and data analysis. 
        You help users improve their clustering results by suggesting cluster names, 
        moving items between clusters, and providing analysis insights."""
        
        if context:
            clusters_info = context.get('clusters', [])
            if clusters_info:
                base_message += f"\n\nCurrent clusters: {len(clusters_info)} clusters with the following structure:\n"
                for i, cluster in enumerate(clusters_info[:5]):  # Limit to first 5 for context
                    name = cluster.get('name', f'Cluster {i+1}')
                    item_count = len(cluster.get('items', []))
                    base_message += f"- {name}: {item_count} items\n"
        
        base_message += "\nProvide specific, actionable suggestions. Be concise and helpful."
        return base_message


# Global LLM instance
_llm_instance = None

def get_llm_wrapper() -> LLMWrapper:
    """Get global LLM wrapper instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance

# Convenience functions for direct use
def initLLM(provider: str = "mock", config: Dict[str, Any] = None) -> bool:
    """Initialize LLM - convenience function"""
    wrapper = get_llm_wrapper()
    return wrapper.initLLM(provider, config)

def callLLM(prompt: str, context: Dict[str, Any] = None, 
            temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
    """Call LLM - convenience function"""
    wrapper = get_llm_wrapper()
    return wrapper.callLLM(prompt, context, temperature, max_tokens)