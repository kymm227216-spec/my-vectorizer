import streamlit as st
import cv2
import vtracer
import os
import numpy as np
import base64
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ページ設定
st.set_page_config(page_title="AI Vectorizer Ultra", layout="wide")
st.title("🚀 AI Image to Vector Ultra")

# --- モデルのロード（キャッシュ機能で2回目以降を高速化） ---
@st.cache_resource
def load_upsampler():
    # 軽量なRRDBNetモデルを使用
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        half=False  # クラウド環境のCPUで動かすためFalse
    )

def render_svg(svg_path):
    """SVGをブラウザにプレビュー表示するための関数"""
    with open(svg_path, "r") as f:
        svg = f.read()
        b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%; max-width:500px;"/>'
        st.write(html, unsafe_allow_html=True)

# 準備
upsampler = load_upsampler()

# サイドバー設定
st.sidebar.header("⚙️ 変換設定")
mode = st.sidebar.selectbox("トレースモード", ["spline (滑らか)", "polygon (直線)"], index=0)
color_limit = st.sidebar.slider("色の細かさ", 2, 32, 16)

# メイン画面
uploaded_file = st.file_uploader("画像をアップロード（PNG/JPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="元の画像")

    if st.button("ベクター変換を開始"):
        with st.spinner("AIがエッジを再構築しています...（1分ほどかかる場合があります）"):
            # 1. AI超解像（解像度を上げてガタつきを抑える）
            # メモリ節約のため2倍に設定
            enhanced_img, _ = upsampler.enhance(img, outscale=2)
            temp_path = "temp_enhanced.png"
            svg_path = "result.svg"
            cv2.imwrite(temp_path, enhanced_img)

            # 2. ベクター変換実行
            vtracer.convert_image_to_svg(
                temp_path, 
                svg_path,
                mode='spline' if "spline" in mode else 'polygon',
                clustering='color',
                iteration_count=color_limit,
                filter_speckle=4,
                corner_threshold=60
            )

            with col2:
                st.markdown("### 変換結果のプレビュー")
                render_svg(svg_path)
                with open(svg_path, "rb") as f:
                    st.download_button("📥 SVGファイルをダウンロード", f, file_name="vector_result.svg")
            
            # 後片付け
            if os.path.exists(temp_path):
                os.remove(temp_path)
