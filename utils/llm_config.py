# utils/llm_config.py - LLM Configuration UI

import streamlit as st
import os
from utils.llm_wrapper import get_llm_wrapper

def show_llm_config_sidebar():
    """Show LLM configuration in sidebar"""
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "AI Provider",
            ["mock", "openai", "anthropic", "local"],
            help="Choose your AI provider",
            key="llm_provider"
        )
        
        # API Key input for real providers
        if provider in ["openai", "anthropic"]:
            api_key_env = f"{provider.upper()}_API_KEY"
            existing_key = os.getenv(api_key_env)
            
            if existing_key:
                st.success(f"✅ {provider.title()} API key found in environment")
                show_key = st.checkbox("Show/change API key")
                
                if show_key:
                    new_key = st.text_input(
                        f"{provider.title()} API Key",
                        value=existing_key[:8] + "..." if existing_key else "",
                        type="password",
                        key=f"{provider}_api_key_input"
                    )
            else:
                st.warning(f"⚠️ No {provider.title()} API key found")
                new_key = st.text_input(
                    f"Enter {provider.title()} API Key",
                    type="password",
                    help=f"Set {api_key_env} environment variable or enter here",
                    key=f"{provider}_api_key_input"
                )
                
                if new_key:
                    # Store in session for this session only
                    st.session_state[f"{provider}_api_key"] = new_key
        
        # Model selection
        if provider == "openai":
            model = st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="Choose OpenAI model"
            )
        elif provider == "anthropic":
            model = st.selectbox(
                "Model", 
                ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                help="Choose Anthropic model"
            )
        else:
            model = "default"
        
        # Temperature setting
        temperature = st.slider(
            "Creativity",
            0.0, 1.0, 0.7,
            help="Higher = more creative, Lower = more focused"
        )
        
        # Initialize/Test button
        if st.button("🔧 Initialize AI", use_container_width=True):
            config = {
                "model": model,
                "temperature": temperature
            }
            
            # Add API key from session if available
            if provider in ["openai", "anthropic"]:
                session_key = st.session_state.get(f"{provider}_api_key")
                if session_key:
                    config["api_key"] = session_key
            
            wrapper = get_llm_wrapper()
            success = wrapper.initLLM(provider, config)
            
            if success:
                st.success(f"✅ {provider.title()} initialized")
                st.session_state.llm_ready = True
            else:
                st.error("❌ Initialization failed")
                st.session_state.llm_ready = False
        
        # Show current status
        wrapper = get_llm_wrapper()
        if wrapper.initialized:
            st.info(f"🟢 AI Ready: {wrapper.provider}")
        else:
            st.info("🔴 AI Not Initialized")

def get_llm_status():
    """Get current LLM initialization status"""
    wrapper = get_llm_wrapper()
    return {
        "initialized": wrapper.initialized,
        "provider": wrapper.provider,
        "ready": st.session_state.get("llm_ready", False)
    }

def quick_llm_test():
    """Quick test of LLM functionality"""
    from utils.llm_wrapper import callLLM
    
    test_prompt = "Say 'Hello, I'm working!' in exactly those words."
    response = callLLM(test_prompt, max_tokens=50)
    
    return response is not None