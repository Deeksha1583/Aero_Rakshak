import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import re
import plotly.graph_objects as go
import cv2
import base64
import easyocr

st.set_page_config(page_title="AI Engine RUL", layout="wide")

# Load model
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
active_sensors = joblib.load("active_sensors.pkl")

# Load EasyOCR
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

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

set_bg("1.jpg")

# Styling
st.markdown("""
<style>
h1, h2, h3, h4, h5, h6, label, p, span, div {
    color: white !important;
}

[data-testid="stFileUploader"] * {
    color: black !important;
}

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.9);
    padding: 10px;
    border-radius: 10px;
}

div.stButton > button {
    width: 100%;
    font-size: 22px;
    padding: 15px;
    border-radius: 12px;
    background-color: black;
    color: white;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #4CAF50;
    transform: scale(1.05);
}

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

# Functions
def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

def extract_text_easyocr(image):
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result])
    return text
    
def parse_sensors(text):
    sensors = {}
    
    text = text.lower()
    text = text.replace('$', 's')
    text = text.replace('§', 's')
    text = text.replace('@', '0')

    text = re.sub(r'(\d)(sensor)', r'\1 \2', text)
    text = re.sub(r'(sensor)(\d)', r'\1 \2', text)

    tokens = text.split()

    i = 0
    while i < len(tokens):
        if tokens[i] == "sensor":
            # Sensor number
            if i + 1 < len(tokens):
                num = tokens[i+1]
                
                # Case: "58"
                if num.startswith('5') and len(num) >= 2:
                    num = num[1:]

                # Case: "5 8"
                elif num == '5' and i + 2 < len(tokens):
                    num = tokens[i+2]
                    i += 1

                key = f"s{num}"

                # Value
                if i + 2 < len(tokens):
                    val = tokens[i+2]

                    # Handle "=" case
                    if val == "=" and i + 3 < len(tokens):
                        val = tokens[i+3]

                    # Handle split decimal
                    if i + 3 < len(tokens):
                        next_val = tokens[i+3]
                        if val.isdigit() and next_val.isdigit():
                            val = val + "." + next_val

                    try:
                        value = float(val)
                        if key in active_sensors:
                            sensors[key] = value
                    except:
                        pass
                i += 3
            else:
                i += 1
        else:
            i += 1

    return sensors
    
# Session state
if "sensor_inputs" not in st.session_state:
    st.session_state.sensor_inputs = {s: 0.0 for s in active_sensors}
    
# UI
st.markdown("<h1 style='text-align:center;'>AI ENGINE RUL PREDICTION</h1>", unsafe_allow_html=True)

# AI Input
st.subheader("AI Input")

left, right = st.columns([2,1])

# IMAGE
with left:
    file = st.file_uploader("Upload Sensor Image")
    if file:
        img = Image.open(file)
        processed = preprocess_image(img)
        with st.spinner("AI is extracting sensor values..."):
            try:
                text = extract_text_easyocr(processed)
                st.text_area("OCR Raw Output", text, height=120)

                sensors = parse_sensors(text)
                
                st.session_state.sensor_inputs.update(sensors)
                st.success(f"{len(sensors)} sensors detected")
            except:
                st.error("OCR failed. Try clearer image.")

# Manual Input
st.markdown("---")
st.subheader("Sensor Inputs")

cols = st.columns(3)

for i, key in enumerate(active_sensors):
    with cols[i % 3]:
        st.session_state.sensor_inputs[key] = st.number_input(
            key,
            value=float(st.session_state.sensor_inputs[key])
        )

# Validation
filled = sum(v != 0 for v in st.session_state.sensor_inputs.values())
st.info(f"{filled}/{len(active_sensors)} sensors filled")

# Prediction Button
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3,2,3])

with col2:
    predict = st.button("🚀 Predict RUL", use_container_width=True)

# Result
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

    # Status
    if pred > 80:
        status = "SAFE ENGINE"
        cls = "#28a745"
    elif pred > 30:
        status = "MAINTENANCE SOON"
        cls = "#ffc107"
    else:
        status = "CRITICAL FAILURE"
        cls = "#dc3545"

    # Popup
    st.markdown("""
    <style>
    .popup-box {
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translate(-50%, -20%);
        background: rgba(0,0,0,0.9);
        padding: 30px;
        border-radius: 15px;
        z-index: 9999;
        width: 50%;
        text-align: center;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="popup-box">
        <h2 style="color:white;">🚀 Engine Health Result</h2>
        <div style="
            background-color:{cls};
            padding:15px;
            border-radius:10px;
            font-size:22px;
            font-weight:bold;
            color:white;
        ">
            {status}<br><br>
            Predicted RUL: {pred} cycles
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Gauge
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