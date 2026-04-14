import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pytesseract
from PIL import Image
import speech_recognition as sr
import re
import plotly.graph_objects as go
import cv2
import base64

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="AI Engine RUL", layout="wide")

# Background Function
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                        url("data:image/webp;base64,{data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# APPLY BACKGROUND
set_bg("abc.webp")

# LOAD
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
active_sensors = joblib.load("active_sensors.pkl")

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>

/* GLASS EFFECT */
.block-container {
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
}

/* MAKE ALL TEXT WHITE */
h1, h2, h3, h4, h5, h6, label, p, span, div {
    color: white !important;
}

/* FILE UPLOADER TEXT */
[data-testid="stFileUploader"] * {
    color: black !important;
}

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.9);
    padding: 10px;
    border-radius: 10px;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    font-size: 20px;
    padding: 14px;
    border-radius: 12px;
    background-color: black;
    color: white;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #4CAF50;
    transform: scale(1.05);
}

/* RESULT CARD */
.result-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.green {background-color: rgba(0,255,0,0.2); color:white;}
.yellow {background-color: rgba(255,255,0,0.2); color:white;}
.red {background-color: rgba(255,0,0,0.2); color:white;}

</style>
""", unsafe_allow_html=True)

# ===============================
# FUNCTIONS
# ===============================

def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh
    
def parse_sensors(text):
    sensors = {}

    text = text.lower()
    text = text.replace('$', 's')
    text = text.replace('§', 's')
    text = text.replace('@', '0')
    text = text.replace('o', '0')

    # Remove garbage characters
    text = re.sub(r'[^a-z0-9.= \n]', ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Extract patterns
    matches = re.findall(r's\s*(\d+)\s*(?:=|is)?\s*([0-9.]+)', text)

    for num, val in matches:
        key = f"s{num}"
        if key in active_sensors:
            try:
                sensors[key] = float(val)
            except:
                continue

    return sensors
    
# ===============================
# SESSION STATE
# ===============================
if "sensor_inputs" not in st.session_state:
    st.session_state.sensor_inputs = {s: 0.0 for s in active_sensors}
    
# ===============================
# UI
# ===============================
st.markdown("<h1 style='text-align:center;'>AI ENGINE RUL PREDICTION</h1>", unsafe_allow_html=True)

# ===============================
# AI INPUT
# ===============================
st.subheader("AI Input")

left, right = st.columns([2,1])

# IMAGE
with left:
    file = st.file_uploader("Upload Sensor Image")
    if file:
        img = Image.open(file)
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, config='--psm 6')
        sensors = parse_sensors(text)
        st.session_state.sensor_inputs.update(sensors)
        st.success(f"{len(sensors)} sensors detected from image")

# ===============================
# MANUAL INPUT
# ===============================
st.markdown("---")
st.subheader("🧾 Sensor Inputs")

cols = st.columns(3)

for i, key in enumerate(active_sensors):
    with cols[i % 3]:
        st.session_state.sensor_inputs[key] = st.number_input(
            key,
            value=float(st.session_state.sensor_inputs[key])
        )

# ===============================
# VALIDATION
# ===============================
filled = sum(v != 0 for v in st.session_state.sensor_inputs.values())
st.info(f"{filled}/{len(active_sensors)} sensors filled")

# ===============================
# PREDICT
# ===============================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🚀 Predict RUL")

if predict:

    if filled < 5:
        st.error("Enter at least 5 sensors")
        st.stop()

    df = pd.DataFrame([st.session_state.sensor_inputs])

    for col in df.columns:
        df[col+"_mean"] = df[col]
        df[col+"_std"] = 0
        df[col+"_diff"] = 0

    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)

    pred = int(np.clip(xgb_model.predict(df_scaled)[0], 0, 125))

    # STATUS
    if pred > 80:
        status = "SAFE ENGINE"
        cls = "green"
    elif pred > 30:
        status = "MAINTENANCE SOON"
        cls = "yellow"
    else:
        status = "CRITICAL FAILURE"
        cls = "red"

    st.markdown(f"<div class='result-card {cls}'>{status}<br>RUL: {pred}</div>", unsafe_allow_html=True)

    # GAUGE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={'text': "Engine Health"},
        gauge={
            'axis': {'range': [0,125]},
            'bar': {
                'color': "white",
                'thickness': 0.25
            },
    
            'steps': [
                {'range': [0,30], 'color': "red"},
                {'range': [30,80], 'color': "yellow"},
                {'range': [80,125], 'color': "green"}
            ],
    
            # Optional: cleaner look
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "black"
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)