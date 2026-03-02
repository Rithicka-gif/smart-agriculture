# ===========================================================
# 🌾 Soil-to-Crop Recommendation Model Training
# ===========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# 1️⃣ Load Soil Dataset
# -----------------------------
data = pd.read_csv("data_core.csv")  # Replace with your soil dataset file

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------

# pH Category
def ph_category(ph):
    if ph < 6.5:
        return 'Acidic'
    elif 6.5 <= ph <= 7.5:
        return 'Neutral'
    else:
        return 'Alkaline'
    
data['pH_Category'] = data['Phosphorous'].apply(ph_category)

# Moisture Level
def moisture_level(moisture):
    if moisture < 35:
        return 'Low'
    elif moisture <= 60:
        return 'Medium'
    else:
        return 'High'

data['Moisture_Level'] = data['Moisture'].apply(moisture_level)

# Soil Suitability Score
def soil_score(row):
    score = 0
    score += 40 if row['pH_Category']=="Neutral" else 20
    score += 30 if row['Moisture_Level']=="Medium" else 10
    score += min((row['Nitrogen'] + row['Phosphorous'] + row['Potassium'])/3, 30)
    return score

data['Soil_Suitability_Score'] = data.apply(soil_score, axis=1)

# -----------------------------
# 3️⃣ Prepare Features & Target
# -----------------------------
numeric_features = ['Temparature','Humidity','Moisture','Nitrogen','Phosphorous','Potassium','Soil_Suitability_Score']
categorical_features = ['Soil Type','pH_Category','Moisture_Level']

# One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(data[categorical_features + numeric_features])

# Save preprocessor
joblib.dump(preprocessor, "soil_preprocessor.pkl")

# Target Encoding
y = data['Crop Type']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save Label Encoder
joblib.dump(le, "crop_label_encoder.pkl")

# -----------------------------
# 4️⃣ Build Neural Network
# -----------------------------
num_classes = len(le.classes_)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model.fit(X, y_encoded, epochs=20, batch_size=8, validation_split=0.1)

# -----------------------------
# 6️⃣ Save Model
# -----------------------------
model.save("smart_soil_crop_model.h5")

print("✅ Training Complete! All files saved:")
print("- smart_soil_crop_model.h5")
print("- soil_preprocessor.pkl")
print("- crop_label_encoder.pkl")
