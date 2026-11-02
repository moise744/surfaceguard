"""
SurfaceGuard — Industrial Defect Segmentation Dashboard.

Streamlit interactive app providing:
- Pixel-level defect mask segmentation
- Interactive probability threshold tuning
- Defect blob region table (area, centroid, bbox)
- Multi-panel diagnostics comparison

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from surfaceguard.inference.segmentor import SurfaceSegmentor
from surfaceguard.models.deeplabv3 import DeepLabV3Plus
from surfaceguard.models.unet import UNet
from surfaceguard.visualization.overlay import MaskVisualizer

# Page configuration
st.set_page_config(
    page_title="SurfaceGuard — Defect Segmentation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ff5722;
        margin-bottom: 0.1rem;
    }
    .brand-sub {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .status-badge-defect {
        background: #d32f2f;
        color: white;
        padding: 6px 16px;
        border-radius: 16px;
        font-weight: bold;
    }
    .status-badge-clean {
        background: #388e3c;
        color: white;
        padding: 6px 16px;
        border-radius: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Model & Calibration")
    st.markdown("---")

    model_arch = st.selectbox(
        "Segmentation Architecture",
        ["U-Net (ResNet Encoder)", "DeepLabV3+ (ASPP)"],
        index=0,
    )
    threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay Transparency (Alpha)", 0.1, 0.9, 0.45, 0.05)
    color_choice = st.color_picker("Defect Highlight Color", "#FF0000")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **SurfaceGuard** isolates exact defective pixel boundaries on industrial surfaces.
    
    - **U-Net**: Skip connections preserve fine scratch/crack edges
    - **DeepLabV3+**: Multi-scale Atrous Pyramid captures irregular texture flaws
    """)

# Header
st.markdown('<div class="brand-title">🛡️ SurfaceGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">Pixel-Level Industrial Defect Segmentation & Spatial Failure Diagnostics</div>',
    unsafe_allow_html=True,
)

# Initialize Segmentor
@st.cache_resource
def get_segmentor(arch_name: str, thresh: float):
    if "DeepLab" in arch_name:
        model = DeepLabV3Plus(in_channels=3, num_classes=1)
    else:
        model = UNet(in_channels=3, num_classes=1)
    return SurfaceSegmentor(model=model, threshold=thresh)

segmentor = get_segmentor(model_arch, threshold)
segmentor.threshold = threshold

# File Upload
uploaded_file = st.file_uploader(
    "Upload an industrial surface image (bottle, metal, tile, wood, etc.)",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    np_img = np.array(raw_img)

    with st.spinner("Executing pixel segmentation..."):
        res = segmentor.predict(np_img)

    # Convert hex color to BGR for OpenCV
    h_col = color_choice.lstrip("#")
    r_c, g_c, b_c = tuple(int(h_col[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (b_c, g_c, r_c)

    # Render overlay
    annotated = MaskVisualizer.overlay_mask(
        np_img, res.defect_probability_map, color=bgr_color, alpha=overlay_alpha, threshold=threshold
    )

    # Top KPI summary
    c1, c2, c3, c4 = st.columns(4)
    if res.has_defect:
        c1.markdown(
            '<div style="text-align:center; padding:10px;"><span class="status-badge-defect">DEFECT DETECTED</span></div>',
            unsafe_allow_html=True,
        )
    else:
        c1.markdown(
            '<div style="text-align:center; padding:10px;"><span class="status-badge-clean">SURFACE NORMAL</span></div>',
            unsafe_allow_html=True,
        )

    c2.metric("Defect Blobs", len(res.defect_regions))
    c3.metric("Defect Pixels", f"{res.total_defect_area_pixels:,} px")
    c4.metric("Inference Latency", f"{res.latency_ms:.1f} ms")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Overlay View", "Probability Heatmap", "Blob Diagnostics"])

    with tab1:
        v1, v2 = st.columns(2)
        with v1:
            st.image(raw_img, caption="Original Inspection Image", use_container_width=True)
        with v2:
            st.image(annotated, caption="Predicted Pixel Segmentation Overlay", use_container_width=True)

    with tab2:
        h1, h2 = st.columns(2)
        with h1:
            st.image(res.binary_mask, caption="Binary Mask (Cutoff Applied)", use_container_width=True)
        with h2:
            # Color map probability
            prob_colored = cv2.applyColorMap((res.defect_probability_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB), caption="Continuous Defect Probability Map", use_container_width=True)

    with tab3:
        if res.defect_regions:
            st.markdown("#### Detected Defect Regions Table")
            data = [
                {
                    "Region ID": r.region_id,
                    "Area (Pixels)": r.area_pixels,
                    "Centroid (X, Y)": f"({r.centroid[0]:.1f}, {r.centroid[1]:.1f})",
                    "Bounding Box": f"x={r.bbox[0]}, y={r.bbox[1]}, w={r.bbox[2]}, h={r.bbox[3]}",
                }
                for r in res.defect_regions
            ]
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No defective regions identified above the probability threshold.")

else:
    st.info("Upload an image above or test with demo data.")
    st.markdown("""
    #### Sample Industrial Segmentation Capabilities
    - **Scratches**: Fine linear pixel discontinuities
    - **Cracks**: Irregular high-frequency pixel splits
    - **Surface Dents**: Smooth localized intensity gradients
    """)
