import certifi
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from io import BytesIO
import requests

# Constants
TRAINED_DB_PATH = "db"  # Ensure this folder exists with .jpg images
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Load Model ---
@st.cache_resource
def load_model() -> tf.keras.Model:
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

# --- Feature Extraction ---
def extract_features(image_path: Union[str, BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        tf.keras.backend.clear_session()
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        tf.keras.backend.clear_session()
        return None

# --- Load DB Features ---
@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(db_path: str, model: tf.keras.Model) -> tuple[np.ndarray, list[str]]:
    feature_list = []
    image_paths = []
    try:
        if not os.path.exists(db_path):
            st.warning(f"Database path '{db_path}' not found.")
            return np.empty((0, 2048)), []
        for img_path in os.listdir(db_path):
            if img_path.lower().endswith(".jpg"):
                path = os.path.join(db_path, img_path)
                features = extract_features(path, model)
                if features is not None:
                    feature_list.append(features)
                    image_paths.append(path)
        feature_vectors = np.vstack(feature_list) if feature_list else np.empty((0, 2048))
        return feature_vectors, image_paths
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return np.empty((0, 2048)), []

# --- Find Similar Images ---
def find_similar_images(
    image_path: Union[str, BytesIO], 
    feature_vectors: np.ndarray, 
    image_paths: list[str],
    model: tf.keras.Model, 
    threshold: float = 0.5, 
    top_n: int = 5
) -> list[str]:

    query_features = extract_features(image_path, model)
    if query_features is None or feature_vectors.size == 0:
        st.warning("No features available for comparison.")
        return []

    query_features = query_features.reshape(1, -1)
    if feature_vectors.ndim == 1:
        feature_vectors = feature_vectors.reshape(1, -1)

    similarities = cosine_similarity(query_features, feature_vectors)
    indices = [i for i, sim in enumerate(similarities[0]) if sim > threshold]
    indices = sorted(indices, key=lambda i: similarities[0][i], reverse=True)
    return [image_paths[i] for i in indices[:top_n]]

# --- Initialize Session State ---
def init_session_state():
    if "feature_vectors" not in st.session_state:
        st.session_state.feature_vectors = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = None

# --- Load Image from URL ---
def load_image_from_url(url: str) -> BytesIO:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Visual Image Search", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        body { background-color:#0E1117; color:#FFFFFF; }
        .stButton>button { background-color:#FF6B6B; color:#FFFFFF; border-radius:10px; padding:10px 20px; }
        .stSlider>div>div>div>div>div { color:#FF6B6B; }
        .stFileUploader>div>div>label { color:#FF6B6B; }
        .stTextInput>div>input { background-color:#1E1E1E; color:#FFFFFF; border-radius:8px; border:1px solid #FF6B6B; padding:8px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🖼️ Visual Image Search Engine")
    st.write("Upload an image or provide a URL to find visually similar images from the database.")

    init_session_state()
    model = load_model()

    if st.session_state.feature_vectors is None:
        with st.spinner("Loading database features..."):
            st.session_state.feature_vectors, st.session_state.image_paths = get_feature_vectors_from_db(TRAINED_DB_PATH, model)
            st.success("✅ Database loaded successfully!")

    st.sidebar.header("🔍 Search Options")
    uploaded_img_file = st.sidebar.file_uploader("Upload an image (.jpg)", type="jpg")
    image_url = st.sidebar.text_input("Or enter an image URL:")
    threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
    top_n = st.sidebar.slider("Number of Similar Images", 1, 10, 5)

    input_image = uploaded_img_file or (load_image_from_url(image_url) if image_url else None)

    if input_image:
        uploaded_img = Image.open(input_image)
        st.image(uploaded_img, caption="Input Image", use_container_width=True)  # Updated

        if st.button("🔎 Find Similar Images"):
            with st.spinner("Searching for similar images..."):
                similar_images = find_similar_images(
                    input_image,
                    st.session_state.feature_vectors,
                    st.session_state.image_paths,
                    model,
                    threshold,
                    top_n
                )
                if similar_images:
                    st.success("✅ Similar images found!")
                    cols = st.columns(3)
                    for idx, img_path in enumerate(similar_images):
                        img = Image.open(img_path)
                        with cols[idx % 3]:
                            st.image(img, caption=f"Similar Image {idx + 1}", use_container_width=True)  # Updated
                else:
                    st.warning("⚠️ No similar images found!")

    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
