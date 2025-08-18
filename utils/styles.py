import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles optimized for sidebar navigation with performance indicators"""
    st.markdown("""
    <style>
    /* Hide the default header */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    /* Sidebar navigation styling */
    .stRadio > div {
        background-color: transparent;
    }
    
    .stRadio > div > label {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        display: block;
        width: 100%;
    }
    
    .stRadio > div > label:hover {
        background-color: #f8f9fa;
        border-color: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Fast loading indicator */
    .fast-startup {
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 8px 0;
        animation: fadeInScale 0.5s ease-out;
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Performance metrics */
    .perf-metric {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    
    /* Loading states */
    .loading-state {
        background: linear-gradient(-45deg, #667eea, #764ba2, #667eea, #764ba2);
        background-size: 400% 400%;
        animation: gradientShift 2s ease infinite;
        color: white;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main content area styling */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: none;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Primary button special styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        font-size: 1.1rem;
        padding: 0.8rem 1.5rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4);
    }
    
    /* Fast processing indicator */
    .fast-processing {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%);
        color: #744210;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #f6d55c;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 215, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #5a67d8;
        background-color: #f0f4ff;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        padding: 0.5rem;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e9ecef;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background-color: #ffffff;
    }
    
    /* Selectbox and other input styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Radio button in main content (not sidebar) */
    .main .stRadio > div > label {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    
    .main .stRadio > div > label:hover {
        background-color: #e9ecef;
        border-color: #667eea;
    }
    
    .main .stRadio > div > label[data-checked="true"] {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    /* Success/error/info message styling */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        animation: slideInFromLeft 0.3s ease-out;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        animation: slideInFromLeft 0.3s ease-out;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        animation: slideInFromLeft 0.3s ease-out;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        animation: slideInFromLeft 0.3s ease-out;
    }
    
    @keyframes slideInFromLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Dataframe styling */
    .stDataFrame > div {
        border-radius: 8px;
        border: 1px solid #e9ecef;
        overflow: hidden;
    }
    
    /* Page title styling */
    .main h1 {
        color: #2d3748;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar section titles */
    .sidebar h3 {
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Sidebar metrics */
    .sidebar .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Custom spacing */
    .stMarkdown {
        margin-bottom: 1rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    
    /* Loading spinner override */
    .stSpinner {
        text-align: center;
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Custom alert boxes */
    .custom-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .custom-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .custom-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)