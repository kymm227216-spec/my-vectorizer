import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.title("Vectorizer (Cloud版)")

uploaded = st.file_uploader("画像アップロード", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("L")
    np_img = np.array(img)
    _, bin_img = cv2.threshold(np_img, 200, 255, cv2.THRESH_BINARY)
    st.image(bin_img, caption="ベクター化（二値化）結果")
