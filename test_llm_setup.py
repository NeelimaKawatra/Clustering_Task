#!/usr/bin/env python3
"""
Test script to verify LLM API setup for Clustery application.
Run this script to check if your API keys and models are working correctly.
"""

import os
import sys
from typing import Dict, Any

def test_environment_variables():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    # Try to load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("📁 Loaded .env file")
    except ImportError:
        print("📁 .env file support not available (install python-dotenv for .env support)")
    except Exception as e:
        print(f"📁 .env file not found or error: {e}")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print(f"✅ OPENAI_API_KEY found: {openai_key[:8]}...")
    else:
        print("❌ OPENAI_API_KEY not found")
    
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        print(f"✅ ANTHROPIC_API_KEY found: {anthropic_key[:8]}...")
    else:
        print("❌ ANTHROPIC_API_KEY not found")
    
    return bool(openai_key and openai_key != "your_openai_api_key_here") or bool(anthropic_key and anthropic_key != "your_anthropic_api_key_here")

def test_imports():
    """Test if required packages are installed."""
    print("\n📦 Checking package imports...")
    
    try:
        import openai
        print(f"✅ OpenAI package installed: {openai.__version__}")
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install openai")
        return False
    
    try:
        import anthropic
        print(f"✅ Anthropic package installed: {anthropic.__version__}")
    except ImportError:
        print("❌ Anthropic package not installed. Run: pip install anthropic")
        return False
    
    return True

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI connection...")
    
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello from OpenAI!'"}],
            max_tokens=10
        )
        
        print(f"✅ OpenAI test successful: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def test_anthropic_connection():
    """Test Anthropic API connection."""
    print("\n🤖 Testing Anthropic connection...")
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        response = client.completions.create(
            model="claude-3-haiku-20240307",
            prompt="Human: Say 'Hello from Anthropic!'\n\nAssistant:",
            max_tokens=10
        )
        
        print(f"✅ Anthropic test successful: {response.completion}")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        return False

def test_clustery_integration():
    """Test Clustery's LLM wrapper."""
    print("\n🔧 Testing Clustery LLM integration...")
    
    try:
        # Add the project root to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import without Streamlit dependencies for testing
        from frontend.frontend_finetuning import LLMWrapper
        
        # Test mock LLM
        wrapper = LLMWrapper()
        success = wrapper.initLLM("mock", {"model": "mock"})
        if success:
            print("✅ Mock LLM wrapper working")
        else:
            print("❌ Mock LLM wrapper failed")
            return False
        
        # Test basic LLM call without Streamlit
        response = wrapper.callLLM("Say 'Hello, I am working!' in exactly those words.", max_tokens=20)
        if response and "Hello" in response:
            print("✅ Quick LLM test passed")
        else:
            print("❌ Quick LLM test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Clustery integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    # Check if running with Streamlit (which causes issues)
    if 'streamlit' in sys.modules:
        print("❌ This script should be run with 'python test_llm_setup.py', not 'streamlit run'")
        print("Please run: python test_llm_setup.py")
        return
    
    print("🚀 Clustery LLM Setup Test")
    print("=" * 40)
    
    # Test environment
    env_ok = test_environment_variables()
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Please install required packages first:")
        print("pip install openai anthropic")
        return
    
    # Test API connections
    openai_ok = False
    anthropic_ok = False
    
    if os.getenv("OPENAI_API_KEY"):
        openai_ok = test_openai_connection()
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_ok = test_anthropic_connection()
    
    # Test Clustery integration
    clustery_ok = test_clustery_integration()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    print(f"Environment variables: {'✅' if env_ok else '❌'}")
    print(f"Package imports: {'✅' if imports_ok else '❌'}")
    print(f"OpenAI API: {'✅' if openai_ok else '❌'}")
    print(f"Anthropic API: {'✅' if anthropic_ok else '❌'}")
    print(f"Clustery integration: {'✅' if clustery_ok else '❌'}")
    
    if openai_ok or anthropic_ok:
        print("\n🎉 You're ready to use real LLM APIs in Clustery!")
        print("Go to the Fine-tuning tab and select your preferred provider.")
    else:
        print("\n⚠️  No working LLM APIs found.")
        print("You can still use Clustery with the mock LLM for testing.")

if __name__ == "__main__":
    main()
