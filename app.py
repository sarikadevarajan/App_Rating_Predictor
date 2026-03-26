import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="App Rating Predictor", page_icon="📱", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stNumberInput, .stSelectbox {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Title section
st.markdown("<h1 style='text-align: center;'>📱 App Rating Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict app ratings based on features like installs, reviews, and category</p>", unsafe_allow_html=True)

st.write("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 App Metrics")
    reviews = st.number_input("Reviews", min_value=0)
    installs = st.number_input("Installs", min_value=0)
    size = st.number_input("Size (bytes)", min_value=0)
    price = st.number_input("Price ($)", min_value=0.0)

with col2:
    st.subheader("📂 App Details")
    category = st.selectbox("Category", [
        "FAMILY", "GAME", "TOOLS", "BUSINESS", "LIFESTYLE", "FINANCE"
    ])

    type_app = st.selectbox("Type", ["Free", "Paid"])

    content_rating = st.selectbox("Content Rating", [
        "Everyone", "Teen", "Mature 17+"
    ])

    genres = st.selectbox("Genres", [
        "Action", "Casual", "Simulation", "Tools", "Education"
    ])

st.write("---")

# Prediction button
if st.button("🚀 Predict Rating"):

    # Log transform
    reviews_log = np.log1p(reviews)
    installs_log = np.log1p(installs)
    size_log = np.log1p(size)
    price_log = np.log1p(price)

    # Create input df
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Fill numeric
    input_df['Reviews'] = reviews_log
    input_df['Size'] = size_log
    input_df['Installs'] = installs_log
    input_df['Price'] = price_log

    # Encoding
    cat_col = f'Category_{category}'
    if cat_col in columns:
        input_df[cat_col] = 1

    if 'Type_Paid' in columns:
        input_df['Type_Paid'] = 1 if type_app == 'Paid' else 0

    cr_col = f'Content Rating_{content_rating}'
    if cr_col in columns:
        input_df[cr_col] = 1

    gen_col = f'Genres_{genres}'
    if gen_col in columns:
        input_df[gen_col] = 1

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Output section
    st.markdown("### 🎯 Prediction Result")

    st.markdown(f"""
        <div style="
            background-color:#ffffff;
            padding:20px;
            border-radius:15px;
            text-align:center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        ">
            <h2 style="color:#4CAF50;">⭐ {round(prediction[0], 2)}</h2>
            <p>Estimated App Rating</p>
        </div>
    """, unsafe_allow_html=True)