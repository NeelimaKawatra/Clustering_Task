# Figma link: 
https://www.figma.com/board/D0Hpszl1SgnXcgoMGE39j1/Flowchart?node-id=0-1&p=f&t=wXJQsMWbusnalnO3-0

# 🔍 Clustery - Intelligent Text Clustering Tool

A professional, user-friendly web application for clustering survey responses and text data using advanced machine learning techniques.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/NeelimaKawatra/Clustering_Task
```

### 2. Install Dependencies
```bash
# Basic installation (works with mock clustering)
pip install streamlit pandas

# Full installation (for real BERTopic clustering)
pip install streamlit pandas bertopic sentence-transformers umap-learn hdbscan nltk
```

### 3. Run the Application
```bash
streamlit run main.py
```

### 4. Open in Browser
Navigate to `http://localhost:8501` and start clustering!

## 📁 Project Structure

```
Clustering_Task/
├── main.py
├── backend.py
├── tabs/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── results.py
└── utils/
    ├── __init__.py
    ├── session_state.py
    ├── styles.py
    └── helpers.py
```
####Frontend is split into tabs and utils for ease.
## 🛠️ Usage Guide

### Step 1: Data Loading 📁
- Upload CSV or Excel files (up to 300 rows)
- Select text column for clustering
- Choose respondent ID column (optional)
- Automatic data validation and preview

### Step 2: Preprocessing 🔧
- Choose from 4 preprocessing levels:
  - No preprocessing
  - Basic cleaning (URLs, emails, whitespace)
  - Advanced cleaning (+ stopwords, punctuation)
  - Custom preprocessing (full control)
- Before/after text comparison
- Smart preprocessing recommendations

### Step 3: Clustering ⚙️
- Automatic parameter optimization
- Custom parameter adjustment available
- Real-time clustering with progress tracking
- Performance metrics and timing

### Step 4: Results 📊
- Detailed cluster analysis with confidence scores
- Interactive cluster exploration
- Professional CSV exports
- Comprehensive summary reports
- Session analytics

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configuration:
```bash
# Logging
LOG_LEVEL=INFO
LOG_FILE=clustery_activity.log

# Model Settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_FILE_SIZE=300

# UI Settings
APP_TITLE=Clustery - Text Clustering Tool
```

### Clustering Parameters
The application automatically optimizes parameters based on dataset size:

| Dataset Size | Min Cluster Size | UMAP Neighbors | UMAP Components |
|--------------|------------------|-----------------|-----------------|
| < 50 texts   | 3-5             | 5               | 5               |
| 50-200 texts | 5-8             | 10              | 8               |
| > 200 texts  | 8-12            | 15              | 10              |

## 📊 Analytics & Tracking

Clustery tracks comprehensive user analytics:

- **Session Metrics**: Duration, completion rate, activity counts
- **Usage Patterns**: Tab visits, feature usage, export behavior
- **Performance Data**: Processing times, clustering success rates
- **Error Tracking**: Failed operations with context

Access analytics through:
- Session summary on completion
- Activity logs in `clustery_activity.log`
- Optional real-time sidebar (uncomment in code)

