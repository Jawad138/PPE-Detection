"""Streamlit app: upload an image and check PPE compliance.

Run with:  python -m streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from ppe_utils import REQUIRED_PPE, analyze, draw

WEIGHTS = Path(__file__).resolve().parent / "best.pt"

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")
st.title("🦺 PPE Detection")
st.caption("Upload a construction-site photo to check each person for a helmet and safety vest.")


@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS))


if not WEIGHTS.exists():
    st.error("best.pt not found. Train the model first with `python train.py`.")
    st.stop()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown("**Required PPE:** " + ", ".join(sorted(REQUIRED_PPE)))

uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded:
    image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    result = load_model().predict(image, conf=conf, verbose=False)[0]
    persons, dets = analyze(result)
    annotated = draw(image, persons, dets)

    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    ok = sum(p.compliant for p in persons)
    m1, m2, m3 = st.columns(3)
    m1.metric("People detected", len(persons))
    m2.metric("Compliant", ok)
    m3.metric("Non-compliant", len(persons) - ok)

    st.subheader("Detections")
    st.dataframe(
        [{"class": n, "confidence": round(c, 3), "box (x1,y1,x2,y2)": tuple(round(v) for v in b)}
         for b, n, c in dets],
        use_container_width=True,
    )
