# 🔍 Clustery - Text Clustering Tool

Streamlit app for clustering survey responses and text data using machine learning.

## Quick Start

```bash
# Clone repository
git clone https://github.com/NeelimaKawatra/Clustering_Task

# Create environment
conda env create -f environment.yml
conda activate clustery

# Run app
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

## Features

1. **Data Loading**: Upload CSV/Excel files (up to 300 rows), select text column
2. **Preprocessing**: Choose from 4 cleaning levels or customize
3. **Clustering**: Auto-optimized parameters with real-time progress
4. **Results**: Cluster analysis with confidence scores, exports, and analytics
5. **AI Assist** (optional): LLM-powered suggestions for cluster names, merges, and splits

## AI Setup (Optional)

Enable AI features with one command:

```bash
python setup_api_keys.py
```

Enter your API key when prompted. See [LLM_SETUP_GUIDE.md](LLM_SETUP_GUIDE.md) for details.

## Project Structure

```text
Project_Clustery/
├─ main.py                    # App entrypoint
├─ frontend/                  # UI components (data, preprocessing, clustering, results, finetuning)
├─ backend/                   # Core pipeline (loading, preprocessing, clustering)
├─ utils/                     # Helpers, session state, styles
└─ config.py                  # Configuration
```

## Auto-Parameter Optimization

| Dataset Size | Min Cluster Size | UMAP Neighbors | UMAP Components |
|--------------|------------------|----------------|-----------------|
| < 50 texts   | 3-5             | 5              | 5               |
| 50-200 texts | 5-8             | 10             | 8               |
| > 200 texts  | 8-12            | 15             | 10              |

## Analytics

Session metrics, usage patterns, and performance data tracked in `clustery_activity.log`.
