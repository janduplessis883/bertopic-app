import streamlit as st
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import io

# Title and Description
st.title("BERTopic The App")
st.write("This app demonstrates the use of BERTopic for topic modeling on free text data.")

# ===== SIDEBAR CONFIGURATION =====
st.sidebar.header("⚙️ Configuration")

# UMAP Parameters
st.sidebar.subheader("UMAP Parameters")
n_neighbors_max = st.sidebar.slider(
    "Max n_neighbors (UMAP)",
    min_value=2,
    max_value=15,
    value=5,
    help="Number of neighbors for UMAP dimensionality reduction. Larger values preserve more global structure."
)

n_components = st.sidebar.selectbox(
    "n_components (UMAP)",
    [2, 3],
    index=0,
    help="Number of dimensions for UMAP output (2D or 3D visualization)"
)

min_dist = st.sidebar.slider(
    "min_dist (UMAP)",
    min_value=0.0,
    max_value=0.99,
    value=0.0,
    step=0.01,
    help="Minimum distance between points in UMAP embedding"
)

metric = st.sidebar.selectbox(
    "Metric (UMAP)",
    ["cosine", "euclidean", "manhattan"],
    index=0,
    help="Distance metric for UMAP"
)

# BERTopic Parameters
st.sidebar.subheader("BERTopic Parameters")
min_topic_size = st.sidebar.slider(
    "Minimum Topic Size",
    min_value=2,
    max_value=10,
    value=2,
    help="Minimum number of documents to form a topic"
)

language = st.sidebar.selectbox(
    "Language",
    ["english", "multilingual"],
    index=0,
    help="Language for stopword removal"
)

# Visualization Parameters
st.sidebar.subheader("Visualization Parameters")
top_n_topics = st.sidebar.slider(
    "Top N Topics to Display",
    min_value=5,
    max_value=20,
    value=10,
    help="Number of top topics to show in visualizations"
)

show_all_viz = st.sidebar.checkbox(
    "Show All Visualizations",
    value=True,
    help="Enable/disable all visualizations"
)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

documents = None
source = None

# TAB 1: Manual Input
with tab1:
    st.header("Input Text")
    st.write("Enter your text data below. Each line will be treated as a separate document.")
    user_input = st.text_area("Input your text here (one document per line):", height=200)

    if user_input:
        documents = user_input.split("\n")
        documents = [doc.strip() for doc in documents if doc.strip()]
        source = "manual"

# TAB 2: CSV Upload
with tab2:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read and display the CSV
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        # Select the column containing free text
        column_name = st.selectbox(
            "Select the column containing free text:",
            df.columns,
            help="Choose the column that contains the text you want to analyze"
        )

        # Extract documents from selected column
        if column_name:
            documents = df[column_name].dropna().astype(str).tolist()
            documents = [doc.strip() for doc in documents if doc.strip()]
            source = "csv"
            st.success(f"Loaded {len(documents)} documents from '{column_name}'")

# Analysis Section - appears after documents are loaded
if documents:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        analyze_button = st.button("Start Analysis", type="primary", key="analyze_btn")

    if analyze_button:
        # Initialize BERTopic
        st.header("Step 1: Initialize BERTopic")
        st.write("Initializing the BERTopic model...")

        # Configure UMAP with parameters from sidebar
        n_neighbors = max(2, min(n_neighbors_max, len(documents) - 2))
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            init='random'  # Use random initialization instead of spectral
        )
        topic_model = BERTopic(
            umap_model=umap_model,
            verbose=False,
            min_topic_size=min_topic_size,
            language=language
        )

        # Fit the model
        st.header("Step 2: Fit the Model")
        st.write("Fitting the BERTopic model to your data...")
        try:
            topics, probs = topic_model.fit_transform(documents)
        except Exception as e:
            st.error(f"Error fitting model: {e}")
            st.stop()

        # Display Topics
        st.header("Step 3: Extracted Topics")
        st.write("The following topics were extracted:")
        topic_info = topic_model.get_topic_info()
        st.dataframe(topic_info)

        # Visualizations
        st.header("Step 4: Visualizations")

        if show_all_viz:
            try:
                # Topic Frequency Bar Chart
                st.subheader("Topic Frequency")
                fig = topic_model.visualize_barchart(top_n_topics=min(top_n_topics, len(topic_info) - 1))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate topic frequency chart: {e}")

            try:
                # Topic Hierarchy
                st.subheader("Topic Hierarchy")
                fig_hierarchy = topic_model.visualize_hierarchy()
                st.plotly_chart(fig_hierarchy, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate hierarchy chart: {e}")

            try:
                # Topic Heatmap
                st.subheader("Topic Similarity Heatmap")
                fig_heatmap = topic_model.visualize_heatmap()
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

            try:
                # Topic Words
                st.subheader("Top Words per Topic")
                fig_words = topic_model.visualize_topics()
                st.plotly_chart(fig_words, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate topic words chart: {e}")
        else:
            try:
                # Topic Frequency Bar Chart
                st.subheader("Topic Frequency")
                fig = topic_model.visualize_barchart(top_n_topics=min(top_n_topics, len(topic_info) - 1))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate topic frequency chart: {e}")

        st.success("BERTopic analysis complete!")
else:
    if source is None:
        st.info("👆 Enter text manually or upload a CSV file to get started (minimum 3-5 documents recommended).")
    else:
        st.warning("Please enter at least one valid document (minimum 3-5 recommended for better results).")
