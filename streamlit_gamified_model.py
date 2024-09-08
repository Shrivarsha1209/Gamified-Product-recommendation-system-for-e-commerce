import tensorflow as tf
import pandas as pd
import streamlit as st
from surprise import dump
import numpy as np
import random
from PIL import Image
import io

# Load the recommendation model
_, loaded_algo = dump.load('recommendations_model')

# Load the trained CNN model for FashionMNIST classification
cnn_model = tf.keras.models.load_model("fashion_mnist_cnn.h5")

# Mock categories (FashionMNIST labels)
category_map = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# Load Walmart dataset
data = pd.read_csv("C:/Users/pshri/Downloads/walmart.csv/walmart.csv")

# Initialize user points and badge system
user_points = {}
user_badges = {}

# Define a list of badges and their point requirements
BADGES = {
    "New Explorer": 100,
    "Engaged User": 300,
    "Top Shopper": 500,
    "Master Shopper": 1000
}

# Function to classify a product image using the CNN model
def classify_uploaded_image(image):
    image = image.convert('L').resize((28, 28))  # Convert image to grayscale and resize to 28x28
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape
    prediction = cnn_model.predict(image_array)
    return np.argmax(prediction)

# Function to recommend products using the SVD model
def recommend_products(user_id, n=5):
    if user_id not in data['User_ID'].unique():
        st.write(f"User ID {user_id} is a new user. Returning popular items as a recommendation.")
        popular_products = data.groupby('Product_ID')['Purchase'].mean().sort_values(ascending=False).head(n)
        return [(product_id, purchase) for product_id, purchase in popular_products.items()]

    all_product_ids = data['Product_ID'].unique()
    user_data = data[data['User_ID'] == user_id]
    user_product_ids = set(user_data['Product_ID'].unique())

    predictions = []
    for product_id in all_product_ids:
        if product_id not in user_product_ids:
            pred = loaded_algo.predict(user_id, product_id)
            predictions.append((product_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Function to add points to the user
def add_points(user_id, points):
    if user_id not in user_points:
        user_points[user_id] = 0
    user_points[user_id] += points
    st.write(f"User {user_id} has earned {points} points! Total points: {user_points[user_id]}")
    check_badges(user_id)

# Function to check for badges
def check_badges(user_id):
    for badge, point_threshold in BADGES.items():
        if user_points.get(user_id, 0) >= point_threshold and badge not in user_badges.get(user_id, []):
            if user_id not in user_badges:
                user_badges[user_id] = []
            user_badges[user_id].append(badge)
            st.write(f"Congratulations! User {user_id} has earned the {badge} badge!")

# Streamlit UI
st.title("Gamified Product Recommendation System with Image Classification")

user_id_input = st.text_input("Enter User ID:", "")

# Image upload and classification
uploaded_file = st.file_uploader("Upload a product image for classification", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Classify the uploaded image
    category = classify_uploaded_image(image)
    st.write(f"Classified as: {category_map[category]}")
    
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        add_points(user_id, 20)  # Add points for image upload

# Simulate points for actions (e.g., purchase, feedback)
if st.button("Simulate Purchase"):
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        add_points(user_id, points=random.randint(10, 50))  # Simulate points for making a purchase

# Recommend products
if st.button("Recommend Products"):
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        top_products = recommend_products(user_id, n=5)

        st.write(f"Top 5 recommended products for User ID {user_id}:")
        for product in top_products:
            product_id, estimated_purchase = product
            st.write(f"Product ID: {product_id}, Estimated Purchase: {estimated_purchase:.2f}")

            # User actions: Like or Save recommendation
            if st.button(f"Save Product {product_id} to Wishlist"):
                add_points(user_id, 20)  # Add points for saving a product

            if st.button(f"Like Product {product_id}"):
                add_points(user_id, 10)  # Add points for liking a product

# Show user's badges
if st.button("Show Badges"):
    if user_id_input.isdigit():
        user_id = int(user_id_input)
        if user_id in user_badges:
            st.write(f"User {user_id} Badges: {', '.join(user_badges[user_id])}")
        else:
            st.write(f"User {user_id} has no badges yet.")
