import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Check and create the uploads directory
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load feature list and filenames from pickle files
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# Streamlit app title
st.title('Image Search And Product Suggestion')

# Sidebar for information and navigation
st.sidebar.header('About this App')
st.sidebar.write("""
This application allows users to upload an image of clothing, 
and it suggests similar items based on the features extracted from the image.
""")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

# Function for feature extraction
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to recommend similar items
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        if features is not None:
            # Recommendation
            indices = recommend(features, feature_list)

            # Show recommendations in a card layout
            st.header('Recommended Products')
            cols = st.columns(6)
            for i, col in enumerate(cols):
                if i < len(indices[0]):
                    with col:
                        st.image(filenames[indices[0][i]], caption=f'Product {i + 1}', use_column_width=True)
                        git init
                         # Show distance for more information
                else:
                    st.empty()  # Leave empty space for alignment
    else:
        st.header("Some error occurred in file upload")
