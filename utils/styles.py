import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the application"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px 20px;
        margin: -1rem -1rem 0.5rem -1rem;
        border-radius: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-content {
        display: flex;
        justify-content: center;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    .logo {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0px 24px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #667eea !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    div.block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)