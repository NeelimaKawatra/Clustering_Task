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
#Create environment from environment.yml
conda env create -f environment.yml
#Activate the environment
conda activate clustery
```

### 3. Run the Application
```bash
streamlit run main.py
```

### 4. Open in Browser
Navigate to `http://localhost:8501` and start clustering!

### 5. Optional: Set Up AI Assistant
For AI-powered clustering assistance, set up your LLM API keys:

```bash
# Set environment variables (Windows PowerShell)
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Or test your setup
python test_llm_setup.py
```

See [LLM_SETUP_GUIDE.md](LLM_SETUP_GUIDE.md) for detailed instructions.

## 📁 Project Structure

```
Clustering_Task/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ env.yml
├─ clustery_activity.log                 # run-time logs
├─ main.py                               # app entrypoint (UI/orchestrates tabs)
├─ backend.py                            # core pipeline: load → preprocess → cluster → evaluate
├─ backend_finetuning.py                 # hyperparam search / tuning for clustering
├─ logs/                                 # (extra logs)
├─ tabs/
│  ├─ __init__.py
│  ├─ data_loading.py                    # file upload, dtype casting, column selection
│  ├─ preprocessing.py                   # impute, scale, encode, PCA/UMAP
│  ├─ clustering.py                      # KMeans/DBSCAN/Agglomerative/GMM wrappers
│  ├─ results.py                         # metrics, plots, cluster summaries
│  └─ finetuning.py                      # search strategies & scoring
├─ utils/
│  ├─ __init__.py
│  ├─ helpers.py                         # small utilities
│  ├─ llm_config.py                      # config for LLM features (if used)
│  ├─ llm_wrapper.py                     # LLM helper
│  ├─ session_state.py                   # app/session state handling
│  ├─ styles.py                          # UI styling/theme helpers

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

