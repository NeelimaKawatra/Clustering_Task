import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import Counter

# Force packages to be available
PACKAGES_AVAILABLE = True

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
except ImportError as e:
    PACKAGES_AVAILABLE = True

def has_text_content(series):
    """Check if a column contains meaningful text content"""
    if series.dtype == 'object':
        # Remove null values
        text_data = series.dropna().astype(str)
        
        if len(text_data) == 0:
            return False
        
        # Check if most values have more than just numbers/single characters
        meaningful_text = text_data[text_data.str.len() > 2].count()
        has_words = text_data.str.contains(' ', na=False).sum()
        
        # At least 30% should be meaningful text with spaces
        return (meaningful_text > len(text_data) * 0.3) and (has_words > 0)
    
    return False

def get_optimal_parameters(n_texts):
    """Get optimal BERTopic parameters based on dataset size"""
    if n_texts < 50:
        return {
            'min_cluster_size': max(3, n_texts // 15),
            'min_samples': 2,
            'n_neighbors': 5,
            'n_components': 5,
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    elif n_texts < 200:
        return {
            'min_cluster_size': max(5, n_texts // 25),
            'min_samples': 3,
            'n_neighbors': 10,
            'n_components': 8,
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    else:
        return {
            'min_cluster_size': max(8, n_texts // 40),
            'min_samples': 4,
            'n_neighbors': 15,
            'n_components': 10,
            'embedding_model': 'all-mpnet-base-v2'
        }

def classify_confidence(probabilities, high_threshold=0.7, low_threshold=0.3):
    """Classify confidence levels based on HDBSCAN probabilities"""
    high_conf = probabilities >= high_threshold
    medium_conf = (probabilities >= low_threshold) & (probabilities < high_threshold)
    low_conf = probabilities < low_threshold
    
    return high_conf, medium_conf, low_conf

@st.cache_data
def run_bertopic_clustering(texts, params):
    """Run BERTopic clustering with caching"""
    
    # Set up UMAP
    umap_model = umap.UMAP(
        n_neighbors=params['n_neighbors'],
        n_components=params['n_components'],
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Set up HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    # Set up embedding model
    embedding_model = SentenceTransformer(params['embedding_model'])
    
    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
        calculate_probabilities=True
    )
    
    # Fit the model
    topics, probabilities = topic_model.fit_transform(texts)
    
    return topic_model, topics, probabilities

def page_upload():
    """Page 1: File Upload and Column Selection"""
    st.title("🔍 Welcome to Clustery: Short Text Clustering")
    st.markdown("---")
    
    # File upload section
    st.subheader("📁 Please upload your file")
    
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing your data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully!")
            
            # Show top 5 rows
            st.subheader("📋 Top 5 rows of your data:")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("---")
            
            # Column selection
            st.subheader("🎯 Which column would you like to cluster?")
            
            selected_column = st.selectbox(
                "Select a column:",
                df.columns,
                help="Choose the column containing the text you want to cluster"
            )
            
            # Check if selected column has text content
            if selected_column:
                if has_text_content(df[selected_column]):
                    # Show some sample data from selected column
                    st.subheader(f"📖 Sample data from '{selected_column}':")
                    
                    sample_data = df[selected_column].dropna().head(5)
                    for i, text in enumerate(sample_data, 1):
                        st.write(f"**{i}.** {text}")
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_responses = df[selected_column].count()
                        st.metric("Total Responses", total_responses)
                    
                    with col2:
                        avg_length = df[selected_column].dropna().astype(str).str.len().mean()
                        st.metric("Average Length", f"{avg_length:.1f} chars")
                    
                    with col3:
                        unique_count = df[selected_column].nunique()
                        st.metric("Unique Responses", unique_count)
                    
                    st.markdown("---")
                    
                    # Proceed button
                    if st.button("🚀 Let's do clustering!", type="primary", use_container_width=True):
                        # Store data in session state
                        st.session_state['survey_data'] = df
                        st.session_state['text_column'] = selected_column
                        st.session_state['current_page'] = 'clustering'
                        
                        st.balloons()
                        st.success("🎉 Great! Moving to clustering setup...")
                        st.rerun()
                
                else:
                    # Warning for non-text column
                    st.error("⚠️ **This column doesn't have text to work on!**")
                    st.write("Please select a column that contains text responses suitable for clustering.")
                    
                    # Show some sample data to help user understand
                    st.write(f"Sample data from '{selected_column}':")
                    sample_data = df[selected_column].dropna().head(3)
                    for i, value in enumerate(sample_data, 1):
                        st.write(f"**{i}.** {value}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.write("Please make sure your file is a valid CSV or Excel file.")

def page_clustering():
    """Page 2: BERTopic Clustering"""
    st.title("🔍 Clustery: BERTopic Clustering")
    
    # Back button
    if st.button("← Back to Upload", help="Go back to file upload"):
        st.session_state['current_page'] = 'upload'
        st.rerun()
    
    # Check if packages are available
    if not PACKAGES_AVAILABLE:
        st.error("❌ **Missing required packages!**")
        st.write("Please install the following packages:")
        st.code("pip install bertopic sentence-transformers umap-learn hdbscan")
        st.info("After installation, restart this page.")
        return
    
    # Get data from session state
    if 'survey_data' not in st.session_state or 'text_column' not in st.session_state:
        st.error("❌ **No data found!** Please upload data first.")
        if st.button("🔄 Go to Upload"):
            st.session_state['current_page'] = 'upload'
            st.rerun()
        return
    
    df = st.session_state['survey_data']
    text_column = st.session_state['text_column']
    
    # Prepare text data
    texts = df[text_column].dropna().astype(str).tolist()
    texts = [text.strip() for text in texts if len(text.strip()) > 2]
    
    st.success(f"✅ **Data loaded!** Using {len(texts)} responses from column '{text_column}'")
    
    # Show data summary
    with st.expander("📊 Data Summary"):
        st.write(f"**Total responses:** {len(texts)}")
        st.write(f"**Column:** {text_column}")
        st.write(f"**Average length:** {np.mean([len(text) for text in texts]):.1f} characters")
        st.write("**Sample responses:**")
        for i, text in enumerate(texts[:3]):
            st.write(f"{i+1}. {text}")
    
    st.markdown("---")
    
    # Get optimal parameters
    optimal_params = get_optimal_parameters(len(texts))
    
    st.subheader("🔧 Clustering Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Recommended parameters for your dataset:**")
        for key, value in optimal_params.items():
            st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
    
    with col2:
        use_optimal = st.radio(
            "Parameter choice:",
            ["Use recommended", "Customize"],
            help="Recommended settings work best for most datasets"
        )
    
    # Parameter selection
    if use_optimal == "Customize":
        st.subheader("⚙️ Custom Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            min_cluster_size = st.slider("Min Cluster Size", 2, 20, optimal_params['min_cluster_size'])
            n_neighbors = st.slider("UMAP Neighbors", 2, 30, optimal_params['n_neighbors'])
            embedding_model = st.selectbox("Embedding Model", 
                                         ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                                         index=0)
        
        with col2:
            min_samples = st.slider("Min Samples", 1, 10, optimal_params['min_samples'])
            n_components = st.slider("UMAP Components", 2, 20, optimal_params['n_components'])
        
        params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'embedding_model': embedding_model
        }
    else:
        params = optimal_params
    
    st.markdown("---")
    
    # Clustering button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start BERTopic Clustering", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Initializing BERTopic...")
                progress_bar.progress(10)
                
                status_text.text("🔄 Running clustering algorithm...")
                progress_bar.progress(30)
                
                # Run clustering
                topic_model, topics, probabilities = run_bertopic_clustering(texts, params)
                progress_bar.progress(80)
                
                # Store results
                st.session_state['topic_model'] = topic_model
                st.session_state['topics'] = topics
                st.session_state['probabilities'] = probabilities
                st.session_state['texts'] = texts
                st.session_state['clustering_complete'] = True
                
                progress_bar.progress(100)
                status_text.text("✅ Clustering completed!")
                
                st.balloons()
                st.success("🎉 **Clustering successful!**")
                
            except Exception as e:
                st.error(f"❌ **Error during clustering:** {str(e)}")
                st.write("This might be due to:")
                st.write("- Dataset too small")
                st.write("- Text responses too similar")
                st.write("- Parameter settings need adjustment")
                return
    
    # Show results if clustering is complete
    if st.session_state.get('clustering_complete', False):
        
        st.markdown("---")
        st.header("📊 Clustering Results")
        
        topic_model = st.session_state['topic_model']
        topics = st.session_state['topics']
        probabilities = st.session_state['probabilities']
        texts = st.session_state['texts']
        
        # Basic statistics
        unique_topics = len(set(topics))
        outliers = sum(1 for t in topics if t == -1)
        clustered = len(texts) - outliers
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🗂️ Total Clusters", unique_topics - (1 if outliers > 0 else 0))
        with col2:
            st.metric("✅ Clustered", clustered)
        with col3:
            st.metric("❓ Outliers", outliers)
        with col4:
            cluster_rate = (clustered / len(texts)) * 100
            st.metric("📈 Success Rate", f"{cluster_rate:.1f}%")
        
        # Confidence analysis
        st.subheader("🎯 Confidence Analysis")
        
        high_conf, medium_conf, low_conf = classify_confidence(probabilities)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_count = sum(high_conf)
            high_pct = (high_count/len(probabilities)*100)
            st.metric("🟢 High Confidence", f"{high_count}", f"{high_pct:.1f}%")
            st.caption("Probability ≥ 0.7")
        
        with col2:
            med_count = sum(medium_conf)
            med_pct = (med_count/len(probabilities)*100)
            st.metric("🟡 Medium Confidence", f"{med_count}", f"{med_pct:.1f}%")
            st.caption("Probability 0.3-0.7")
        
        with col3:
            low_count = sum(low_conf)
            low_pct = (low_count/len(probabilities)*100)
            st.metric("🔴 Low Confidence", f"{low_count}", f"{low_pct:.1f}%")
            st.caption("Probability < 0.3")
        
        # Topic details
        st.subheader("📝 Cluster Details")
        
        topic_info = topic_model.get_topic_info()
        if len(topic_info) > 0:
            # Filter out outliers for main display
            main_topics = topic_info[topic_info['Topic'] != -1] if -1 in topic_info['Topic'].values else topic_info
            
            for idx, row in main_topics.iterrows():
                topic_num = row['Topic']
                topic_size = row['Count']
                
                # Get topic words
                topic_words = topic_model.get_topic(topic_num)
                top_words = [word for word, score in topic_words[:5]]
                
                # Get sample texts for this topic
                topic_indices = [i for i, t in enumerate(topics) if t == topic_num]
                topic_texts = [texts[i] for i in topic_indices]
                topic_probs = [probabilities[i] for i in topic_indices]
                
                with st.expander(f"📋 **Cluster {topic_num}** ({topic_size} responses) - {', '.join(top_words[:3])}"):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**🔤 Top Keywords:**")
                        st.write(", ".join(top_words))
                        
                        st.write("**📄 Sample Responses:**")
                        # Show top 5 responses with highest confidence
                        sorted_samples = sorted(zip(topic_texts, topic_probs), key=lambda x: x[1], reverse=True)
                        for i, (text, prob) in enumerate(sorted_samples[:5]):
                            confidence_emoji = "🟢" if prob >= 0.7 else "🟡" if prob >= 0.3 else "🔴"
                            st.write(f"{confidence_emoji} {text} *(conf: {prob:.2f})*")
                    
                    with col2:
                        avg_confidence = np.mean(topic_probs)
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        high_conf_in_topic = sum(1 for p in topic_probs if p >= 0.7)
                        st.metric("High Confidence Items", high_conf_in_topic)
        
        # Next steps
        st.markdown("---")
        st.subheader("🎯 What's Next?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ **Export results**", type="primary", use_container_width=True):
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'text': texts,
                    'cluster': topics,
                    'confidence': probabilities
                })
                
                # Add cluster labels
                cluster_labels = {}
                for topic_num in set(topics):
                    if topic_num != -1:
                        words = topic_model.get_topic(topic_num)[:3]
                        cluster_labels[topic_num] = "_".join([word for word, score in words])
                    else:
                        cluster_labels[topic_num] = "outlier"
                
                results_df['cluster_label'] = results_df['cluster'].map(cluster_labels)
                
                st.session_state['final_results'] = results_df
                st.success("🎉 Results ready for export!")
                st.download_button(
                    "📥 Download Results CSV",
                    results_df.to_csv(index=False),
                    "clustering_results.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("🔧 **Manual review**", use_container_width=True):
                st.session_state['current_page'] = 'review'
                st.rerun()
        
        with col3:
            if st.button("🔄 **Start over**", use_container_width=True):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state['current_page'] = 'upload'
                st.rerun()

def page_review():
    """Page 3: Manual Review (Coming Soon)"""
    st.title("🔧 Manual Review")
    
    if st.button("← Back to Results"):
        st.session_state['current_page'] = 'clustering'
        st.rerun()
    
    st.info("🚧 **Manual review interface coming soon!**")
    st.write("This page will allow you to:")
    st.write("- 📝 Move items between clusters")
    st.write("- 🔗 Merge clusters")
    st.write("- ✂️ Split clusters")
    st.write("- 🎯 Focus on low-confidence items")
    st.write("- 🏷️ Rename clusters")

def main():
    st.set_page_config(page_title="Clustery", layout="wide")
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'upload'
    
    # Navigation
    current_page = st.session_state.get('current_page', 'upload')
    
    if current_page == 'upload':
        page_upload()
    elif current_page == 'clustering':
        page_clustering()
    elif current_page == 'review':
        page_review()

if __name__ == "__main__":
    main()
