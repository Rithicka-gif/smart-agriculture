
# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import base64
import os

# ---------------------------
# Page Config (MUST BE FIRST)
# ---------------------------
st.set_page_config(page_title="Crop Disease Detection", page_icon="🌱")

# ---------------------------
# Background Function
# ---------------------------
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255,255,255,0.3);  /* lighter overlay */
        z-index: -1;
    }}
    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

# Call background AFTER page config
set_bg("backgroud_image.jpg")

# ---------------------------
# Title
# ---------------------------
st.markdown(
    "<h1 style='text-align: center; color: white;'>Crop Disease Detection</h1>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <h4 style='color: white;'>Upload a crop leaf image to detect the disease and get remedy.</h4>
    <p style='color: white; font-weight: 600;'>Choose an image:</p>
    """,
    unsafe_allow_html=True
)
# ---------------------------
# Load Model & Class Names
# ---------------------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

# ---------------------------
# Disease Information Database
# ---------------------------
disease_info = {

# ---------------- APPLE ----------------
"Apple___Apple_scab": {
    "description": "Fungal disease causing olive-green or brown spots on leaves and fruits.",
    "remedy": "Spray Mancozeb or Captan fungicide every 7–10 days.",
    "prevention": "Ensure good air circulation and remove fallen leaves."
},

"Apple___Black_rot": {
    "description": "Dark rot lesions on fruits and circular leaf spots.",
    "remedy": "Prune infected branches and apply Thiophanate-methyl.",
    "prevention": "Maintain orchard sanitation."
},

"Apple___Cedar_apple_rust": {
    "description": "Yellow-orange spots on leaves caused by fungal infection.",
    "remedy": "Apply Myclobutanil fungicide.",
    "prevention": "Remove nearby cedar trees if possible."
},

"Apple___healthy": {
    "description": "The apple plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Maintain proper irrigation and fertilization."
},

# ---------------- BLUEBERRY ----------------
"Blueberry___healthy": {
    "description": "The blueberry plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Ensure proper soil acidity and watering."
},

# ---------------- CHERRY ----------------
"Cherry_(including_sour)___Powdery_mildew": {
    "description": "White powdery fungal growth on leaves.",
    "remedy": "Apply Sulfur or Potassium bicarbonate spray.",
    "prevention": "Improve air circulation."
},

"Cherry_(including_sour)___healthy": {
    "description": "The cherry plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Regular pruning and monitoring."
},

# ---------------- CORN ----------------
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
    "description": "Rectangular gray lesions on leaves.",
    "remedy": "Apply Azoxystrobin fungicide.",
    "prevention": "Rotate crops and use resistant hybrids."
},

"Corn_(maize)___Common_rust_": {
    "description": "Reddish-brown pustules on leaves.",
    "remedy": "Apply Propiconazole fungicide.",
    "prevention": "Plant resistant varieties."
},

"Corn_(maize)___Northern_Leaf_Blight": {
    "description": "Long gray-green lesions on leaves.",
    "remedy": "Use Mancozeb spray.",
    "prevention": "Crop rotation and debris removal."
},

"Corn_(maize)___healthy": {
    "description": "The corn plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Balanced fertilization."
},

# ---------------- GRAPE ----------------
"Grape___Black_rot": {
    "description": "Black spots on leaves and shriveled fruits.",
    "remedy": "Apply Mancozeb or Myclobutanil.",
    "prevention": "Remove infected berries."
},

"Grape___Esca_(Black_Measles)": {
    "description": "Dark streaks in wood and spotted leaves.",
    "remedy": "Remove infected vines.",
    "prevention": "Avoid pruning wounds during wet weather."
},

"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
    "description": "Brown irregular leaf spots.",
    "remedy": "Apply Copper-based fungicide.",
    "prevention": "Ensure good drainage."
},

"Grape___healthy": {
    "description": "The grape plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Regular vineyard maintenance."
},

# ---------------- ORANGE ----------------
"Orange___Haunglongbing_(Citrus_greening)": {
    "description": "Yellow shoots and bitter fruits caused by bacterial infection.",
    "remedy": "Remove infected trees immediately.",
    "prevention": "Control psyllid insects."
},

# ---------------- PEACH ----------------
"Peach___Bacterial_spot": {
    "description": "Dark lesions on leaves and fruits.",
    "remedy": "Apply Copper spray.",
    "prevention": "Avoid overhead irrigation."
},

"Peach___healthy": {
    "description": "The peach plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Maintain orchard hygiene."
},

# ---------------- PEPPER ----------------
"Pepper,_bell___Bacterial_spot": {
    "description": "Water-soaked spots on leaves and fruits.",
    "remedy": "Apply Copper-based bactericide.",
    "prevention": "Use disease-free seeds."
},

"Pepper,_bell___healthy": {
    "description": "The pepper plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Proper spacing and watering."
},

# ---------------- POTATO ----------------
"Potato___Early_blight": {
    "description": "Brown concentric rings on leaves.",
    "remedy": "Apply Chlorothalonil.",
    "prevention": "Crop rotation."
},

"Potato___Late_blight": {
    "description": "Dark water-soaked lesions spreading quickly.",
    "remedy": "Apply Metalaxyl immediately.",
    "prevention": "Avoid excess irrigation."
},

"Potato___healthy": {
    "description": "The potato plant appears healthy.",
    "remedy": "No treatment required.",
    "prevention": "Proper soil drainage."
}

}

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[class_index]

    st.success(f"🌾 Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # ---------------------------
    # Show Disease Details
    # ---------------------------
    st.subheader("🧾 Disease Details")

    if "healthy" in predicted_class.lower():
        st.success("🌿 The plant appears healthy!")
        st.info("Maintain proper watering, sunlight, and balanced fertilization.")
    else:
        if predicted_class in disease_info:
            info = disease_info[predicted_class]

            disease_name = predicted_class.split("___")[1].replace("_", " ")

            st.write(f"### 🦠 Disease: {disease_name}")
            st.write("### 📌 Description")
            st.write(info["description"])

            st.write("### 💊 Remedy")
            st.success(info["remedy"])

            st.write("### 🌾 Prevention / Avoidance Technique")
            st.info(info["prevention"])
        else:
            st.warning("Detailed remedy not available for this disease yet.")
