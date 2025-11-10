# frontend/frontend_llm_settings.py
"""
LLM Configuration Page - Session-wide settings for all AI operations
"""

import streamlit as st
import os
from typing import Optional

def tab_llm_settings(backend_available: bool):
    """
    LLM Configuration Page
    
    Allows users to configure:
    - Provider (mock/openai)
    - Model selection
    - Temperature (creativity level)
    - Max tokens (response length)
    
    Settings persist across all tabs in the session.
    """
    
    # Track tab visit
    if backend_available and hasattr(st.session_state, 'backend') and st.session_state.backend:
        try:
            st.session_state.backend.track_activity(
                st.session_state.session_id, 
                "tab_visit", 
                {"tab_name": "llm_settings"}
            )
        except Exception:
            pass
    
    # Initialize session defaults if not present
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = {
            'provider': 'mock',
            'model': 'gpt-4o-mini',
            'temperature': 0.7,
            'max_tokens': 500,
            'initialized': False
        }
    
    # Header
    st.subheader("🔧 LLM Provider Configuration")
    st.markdown("""
    Configure your AI assistant settings once for the entire session.
    These settings will apply to all AI operations (cluster naming, suggestions, etc.).
    """)
    
    # Provider Selection Section
    with st.container():
        st.markdown("### 1️⃣ Choose Provider")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            provider = st.selectbox(
                "LLM Provider",
                options=["mock", "openai"],
                index=0 if st.session_state.llm_config['provider'] == 'mock' else 1,
                help="Mock provider for testing (no API calls). OpenAI requires API key.",
                key="provider_select"
            )
        
        with col2:
            if provider == "mock":
                st.info("🤖 **Test Mode**")
            else:
                st.success("🚀 **Production Mode**")
    
    # Provider-specific configuration
    st.markdown("---")
    
    if provider == "openai":
        _configure_openai_provider()
    else:
        _configure_mock_provider()
    
    st.markdown("---")
    
    # Generation Parameters Section
    st.markdown("### 2️⃣ Generation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "🌡️ Temperature (Creativity Level)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_config['temperature'],
            step=0.05,
            help="Lower = more focused and consistent. Higher = more creative and varied.",
            key="temperature_slider"
        )
        
        # Temperature guidance
        if temperature <= 0.3:
            st.caption("❄️ **Low**: Deterministic, factual (best for naming)")
        elif temperature <= 0.7:
            st.caption("🌤️ **Medium**: Balanced creativity (recommended)")
        else:
            st.caption("🔥 **High**: Very creative (experimental)")
    
    with col2:
        max_tokens = st.number_input(
            "📝 Max Tokens",
            min_value=100,
            max_value=2000,
            value=st.session_state.llm_config.get('max_tokens', 500),
            step=50,
            help="Maximum length of AI responses (1 token ≈ 0.75 words)",
            key="max_tokens_input"
        )
        
        # Token guidance
        word_estimate = int(max_tokens * 0.75)
        st.caption(f"📊 Approximately **{word_estimate}** words")
    
    st.markdown("---")
    
    # Save Configuration Section
    st.markdown("### 3️⃣ Save Configuration")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        save_button = st.button(
            "💾 Save & Apply Configuration", 
            type="primary", 
            use_container_width=True,
            key="save_config_btn"
        )
    
    with col2:
        if st.session_state.llm_config.get('initialized'):
            if st.button("🔄 Reset to Defaults", use_container_width=True, key="reset_config_btn"):
                _reset_to_defaults()
                st.rerun()
    
    with col3:
        if st.button("🧪 Test LLM", use_container_width=True, key="test_llm_btn"):
            _test_llm_connection(provider, temperature, max_tokens)
    
    # Handle save button
    if save_button:
        _save_configuration(provider, temperature, max_tokens)
    
    st.markdown("---")
    
    # Current Configuration Display
    if st.session_state.llm_config.get('initialized'):
        _display_current_config()
    else:
        st.info("ℹ️ No configuration saved yet. Configure and save your settings above.")
    
    # Additional Information Sections
    _display_cost_estimates(provider)
    _display_usage_tips()
    _display_temperature_guide()


def _configure_openai_provider():
    """Configure OpenAI-specific settings"""
    
    st.markdown("#### OpenAI Configuration")
    
    # API Key Status
    api_key = os.getenv("OPENAI_API_KEY")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if api_key:
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            st.success(f"✅ API Key Configured: `{masked_key}`")
        else:
            st.error("❌ No API key found!")
            st.markdown("""
            **To set up your API key:**
            1. Run `python setup_api_keys.py` in your terminal
            2. Or manually create a `.env` file with: `OPENAI_API_KEY=your_key_here`
            3. Restart the application
            """)
    
    with col2:
        if not api_key:
            st.markdown("[Get API Key →](https://platform.openai.com/api-keys)")
    
    # Model Selection
    if api_key:
        current_model = st.session_state.llm_config['model']
        
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        # Ensure current model is in list
        if current_model not in models:
            current_model = "gpt-4o-mini"
        
        model = st.selectbox(
            "🤖 OpenAI Model",
            options=models,
            index=models.index(current_model),
            help="gpt-4o-mini recommended for best cost/performance ratio",
            key="model_select"
        )
        
        # Store model in temp variable for save button to access
        st.session_state._temp_model = model
        
        # Model recommendations
        if model == "gpt-4o-mini":
            st.info("💡 **Recommended**: Best balance of speed, cost, and quality")
        elif model == "gpt-4o":
            st.warning("💰 **Premium**: 10x more expensive but more capable")
        elif model == "gpt-3.5-turbo":
            st.info("💵 **Budget**: Older model, cheapest option")


def _configure_mock_provider():
    """Configure Mock provider settings"""
    
    st.markdown("#### Mock Provider Configuration")
    
    st.info("""
    🤖 **Mock LLM Mode**
    
    The Mock provider simulates LLM responses without making API calls:
    - ✅ Free (no API costs)
    - ✅ Fast (instant responses)
    - ✅ Useful for testing and development
    - ⚠️ Returns example/template responses
    
    **When to use Mock:**
    - Testing the application features
    - Demonstrating functionality without costs
    - Developing without API access
    
    **When to use OpenAI:**
    - Real clustering analysis
    - Production use
    - Generating actual suggestions
    """)
    
    # Store mock model
    st.session_state._temp_model = "mock"


def _save_configuration(provider: str, temperature: float, max_tokens: int):
    """Save LLM configuration to session state and initialize wrapper"""
    
    model = st.session_state.get('_temp_model', 'gpt-4o-mini')
    
    # Update session state
    st.session_state.llm_config = {
        'provider': provider,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'initialized': True
    }
    
    # Initialize LLM wrapper with new settings
    try:
        from frontend.frontend_finetuning import get_llm_wrapper
        from backend.finetuning_backend import get_finetuning_backend
        
        backend = get_finetuning_backend()
        llm_wrapper = get_llm_wrapper()
        llm_wrapper.backend = backend
        
        if provider != "mock":
            config = {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY")
            }
            success = llm_wrapper.initLLM(provider=provider, config=config)
            
            if success:
                st.success(f"✅ LLM configured successfully!")
                st.balloons()
                st.info(f"**Active Configuration:** {provider.upper()} ({model}) at temperature {temperature:.2f}")
                
                # Track configuration
                if hasattr(st.session_state, 'backend') and st.session_state.backend:
                    try:
                        st.session_state.backend.track_activity(
                            st.session_state.session_id,
                            "llm_configured",
                            {
                                "provider": provider,
                                "model": model,
                                "temperature": temperature,
                                "max_tokens": max_tokens
                            }
                        )
                    except Exception:
                        pass
                
                # Brief pause then rerun
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to initialize LLM. Check your API key and model selection.")
        else:
            llm_wrapper.initLLM(provider="mock", config={"model": "mock"})
            st.success("✅ Mock LLM configured for testing!")
            st.info("The Mock provider will return example responses without making API calls.")
            
            import time
            time.sleep(1)
            st.rerun()
    
    except ImportError as e:
        st.error(f"❌ Failed to import LLM modules: {e}")
        st.info("Make sure all dependencies are installed: `pip install openai python-dotenv`")
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")


def _reset_to_defaults():
    """Reset configuration to default values"""
    st.session_state.llm_config = {
        'provider': 'mock',
        'model': 'gpt-4o-mini',
        'temperature': 0.7,
        'max_tokens': 500,
        'initialized': False
    }
    st.success("🔄 Reset to default configuration")


def _test_llm_connection(provider: str, temperature: float, max_tokens: int):
    """Test LLM connection with a simple query"""
    
    model = st.session_state.get('_temp_model', 'gpt-4o-mini')
    
    with st.spinner("🧪 Testing LLM connection..."):
        try:
            from frontend.frontend_finetuning import callLLM
            
            # Temporarily set config for test
            old_config = st.session_state.llm_config.copy()
            st.session_state.llm_config = {
                'provider': provider,
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'initialized': True
            }
            
            # Test query
            test_prompt = "Say 'LLM connection successful!' and nothing else."
            response = callLLM(test_prompt, context={}, temperature=temperature, max_tokens=50)
            
            # Restore old config
            st.session_state.llm_config = old_config
            
            if response:
                st.success(f"✅ **Connection Successful!**")
                st.code(response, language=None)
            else:
                st.error("❌ Test failed: No response received")
        
        except Exception as e:
            st.error(f"❌ Test failed: {e}")


def _display_current_config():
    """Display currently active configuration"""
    
    st.markdown("### 📋 Active Configuration")
    
    config = st.session_state.llm_config
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Provider", 
            config['provider'].upper(),
            help="Active LLM provider"
        )
    
    with col2:
        st.metric(
            "Model", 
            config['model'],
            help="Active model name"
        )
    
    with col3:
        st.metric(
            "Temperature", 
            f"{config['temperature']:.2f}",
            help="Creativity level (0.0-1.0)"
        )
    
    with col4:
        st.metric(
            "Max Tokens", 
            config['max_tokens'],
            help="Maximum response length"
        )
    
    st.success("✅ **Status**: Configuration active for all AI operations in this session")


def _display_cost_estimates(provider: str):
    """Display cost estimates for OpenAI"""
    
    if provider == "openai":
        with st.expander("💰 OpenAI Cost Estimates", expanded=False):
            st.markdown("""
            **Approximate API Costs (as of 2024):**
            
            | Model | Input (per 1M tokens) | Output (per 1M tokens) | Typical Operation |
            |-------|----------------------|------------------------|-------------------|
            | **gpt-4o-mini** | ~$0.15 | ~$0.60 | $0.001 - $0.01 |
            | **gpt-4o** | ~$2.50 | ~$10.00 | $0.01 - $0.10 |
            | **gpt-4-turbo** | ~$10.00 | ~$30.00 | $0.05 - $0.20 |
            | **gpt-3.5-turbo** | ~$0.50 | ~$1.50 | $0.002 - $0.02 |
            
            **Example Costs for Clustery Operations:**
            - Cluster naming (50 clusters): $0.01 - $0.05
            - Entry move suggestions (100 entries): $0.02 - $0.10
            - Merge/split analysis: $0.01 - $0.05
            
            **💡 Tip**: Start with `gpt-4o-mini` for the best cost/performance ratio.
            
            *Note: Prices subject to change. Check [OpenAI Pricing](https://openai.com/pricing) for current rates.*
            """)


def _display_usage_tips():
    """Display usage tips and best practices"""
    
    with st.expander("💡 Usage Tips & Best Practices", expanded=False):
        st.markdown("""
        ### Temperature Guidelines
        
        | Temperature Range | Best For | Example Use Case |
        |------------------|----------|------------------|
        | **0.0 - 0.3** | Factual, consistent tasks | Cluster naming with technical terms |
        | **0.4 - 0.6** | Balanced tasks | General suggestions and analysis |
        | **0.7 - 0.9** | Creative exploration | Brainstorming alternative approaches |
        | **0.9 - 1.0** | Highly creative | Experimental (may be less accurate) |
        
        ### Model Selection Guide
        
        **gpt-4o-mini** (Recommended) ⭐
        - Fast and affordable
        - Great for most clustering tasks
        - Best cost/performance ratio
        
        **gpt-4o**
        - More capable reasoning
        - Better for complex analysis
        - 10x more expensive
        
        **gpt-3.5-turbo**
        - Budget option
        - Older technology
        - May be less accurate
        
        ### Best Practices
        
        1. 🧪 **Start with Mock provider** to test features without costs
        2. 💰 **Use gpt-4o-mini** for production work
        3. ❄️ **Lower temperature** (0.2-0.4) for factual operations
        4. 🔥 **Higher temperature** (0.6-0.8) for creative suggestions
        5. 📊 **Monitor usage** at [OpenAI Dashboard](https://platform.openai.com/usage)
        6. 💾 **Save configuration** before starting analysis
        7. 🔒 **Protect API keys** (never share or commit to git)
        
        ### Common Issues
        
        **"No API key found"**
        - Run `python setup_api_keys.py`
        - Or create `.env` file with your key
        
        **"Rate limit exceeded"**
        - You've hit OpenAI's usage limits
        - Wait and try again, or upgrade your plan
        
        **"Insufficient quota"**
        - Add credits at [OpenAI Billing](https://platform.openai.com/account/billing)
        
        **"Invalid API key"**
        - Check your key at [OpenAI Keys](https://platform.openai.com/api-keys)
        - Generate a new key if needed
        """)


def _display_temperature_guide():
    """Display detailed temperature guide"""
    
    with st.expander("🌡️ Temperature Deep Dive", expanded=False):
        st.markdown("""
        ### What is Temperature?
        
        Temperature controls the **randomness** of LLM responses:
        
        - **Low temperature**: Model picks the most likely next word (deterministic)
        - **High temperature**: Model considers less likely alternatives (creative)
        
        ### Visual Example
        
        **Prompt**: "Name this cluster: 'late delivery', 'slow shipping', 'package delayed'"
        
        **Temperature = 0.0** ❄️
```
        Response 1: "Delivery Delays"
        Response 2: "Delivery Delays"
        Response 3: "Delivery Delays"
        → Always identical (boring but consistent)
```
        
        **Temperature = 0.5** 🌤️
```
        Response 1: "Delivery Delays"
        Response 2: "Shipping Issues"
        Response 3: "Late Deliveries"
        → Reasonable variations
```
        
        **Temperature = 1.0** 🔥
```
        Response 1: "Snail Mail Chronicles"
        Response 2: "Temporal Transport Troubles"
        Response 3: "Patience-Testing Parcels"
        → Very creative (maybe too much!)
```
        
        ### Technical Details
        
        Temperature affects the probability distribution:
```
        Low Temperature (0.0):
           *         ← Only picks peak
          ***
         *****
        *******
        
        High Temperature (1.0):
        *******     ← Picks anywhere
        *******     (flatter distribution)
        *******
```
        
        ### Recommendations for Clustery
        
        | Task | Recommended Temperature |
        |------|------------------------|
        | Cluster naming | 0.2 - 0.3 |
        | Entry classification | 0.3 - 0.4 |
        | Merge suggestions | 0.4 - 0.5 |
        | Free-form analysis | 0.5 - 0.7 |
        | Brainstorming | 0.7 - 0.9 |
        
        **Default (0.7)** is a good all-purpose setting.
        """)


# Health check function
def check_llm_configuration() -> dict:
    """
    Check if LLM is properly configured
    Returns dict with status information
    """
    
    if 'llm_config' not in st.session_state:
        return {
            'configured': False,
            'message': 'LLM not configured. Go to LLM Settings page.',
            'provider': None,
            'model': None
        }
    
    config = st.session_state.llm_config
    
    if not config.get('initialized'):
        return {
            'configured': False,
            'message': 'LLM configuration incomplete. Go to LLM Settings page.',
            'provider': config.get('provider'),
            'model': config.get('model')
        }
    
    # Check API key for OpenAI
    if config.get('provider') == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                'configured': False,
                'message': 'OpenAI API key not found. Run setup_api_keys.py',
                'provider': 'openai',
                'model': config.get('model')
            }
    
    return {
        'configured': True,
        'message': f"Using {config.get('provider').upper()} ({config.get('model')})",
        'provider': config.get('provider'),
        'model': config.get('model'),
        'temperature': config.get('temperature'),
        'max_tokens': config.get('max_tokens')
    }