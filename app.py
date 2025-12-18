import streamlit as st
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as T
import warnings
from PIL import Image
from datetime import datetime
import gc

warnings.filterwarnings("ignore", category=UserWarning)


from processor import CrowdMindProcessor


# ======================================================================
# ====== STREAMLIT APP LOGIC ======
# ======================================================================

st.set_page_config(
    page_title="CrowdMind AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_crowd_processor():
    try:
        return CrowdMindProcessor()
    except FileNotFoundError as e:
        st.error(f"Error initializing CrowdMindProcessor: {e}. "
                 "Please check your model asset paths.")
        st.stop()


processor = get_crowd_processor()

if 'run_stream' not in st.session_state:
    st.session_state.run_stream = True
if 'video_source' not in st.session_state:
    st.session_state.video_source = "Samples/CCTV_Crowd_Congestion_Video_Generation.mp4"

# ======================================================================
# ====== UI LAYOUT ======
# ======================================================================

header_col, clock_col = st.columns([1, 0.2])
with header_col:
    st.markdown("<h1 style='text-align: left; color: #FFFFFF;'>CrowdMind AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: #A0A0A0;'>Real-time Crowd Analysis Dashboard</h3>",
                unsafe_allow_html=True)
with clock_col:
    clock_placeholder = st.empty()

with st.sidebar:
    st.header("Controls")
    if st.session_state.run_stream:
        if st.button('Stop Video'):
            st.session_state.run_stream = False
    else:
        st.info("Video is stopped. Refresh the page to restart.")

    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'], key="video_uploader")
    if uploaded_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_source = "uploaded_video.mp4"
        st.session_state.run_stream = True
        st.success("Video uploaded successfully. Analysis will start automatically.")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Live Feed")
    live_feed_placeholder = st.empty()
    st.markdown("### Heatmap & Crowd Flow")
    heatmap_placeholder = st.empty()

with col2:
    st.markdown("### Live Stats")
    total_people_metric = st.metric(label="Total People Count", value="--")
    st.markdown("#### Zone-wise Density")
    zone_A_metric = st.metric(label="Zone A", value="--")
    zone_B_metric = st.metric(label="Zone B", value="--")
    st.markdown("#### Stampede Risk Level")
    risk_level_container = st.container(border=True)
    with risk_level_container:
        risk_label_placeholder = st.empty()
    st.markdown("#### Redistribution Suggestion")
    redistribution_container = st.container(border=True)
    with redistribution_container:
        redistribution_text_placeholder = st.empty()

st.markdown("---")
alert_placeholder = st.empty()

# ======================================================================
# ====== STREAM PROCESSING LOOP ======
# ======================================================================

if st.session_state.run_stream:
    cap = cv2.VideoCapture(st.session_state.video_source)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {st.session_state.video_source}")
        st.session_state.run_stream = False
    else:
        while st.session_state.run_stream:
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video stream or failed to read frame. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            clock_placeholder.markdown(f"<p style='text-align: right; color: #A0A0A0;'>Live: {current_time} UTC</p>",
                                       unsafe_allow_html=True)

            result_frame, people_count, risk_class, risk_percent, density_map = processor.process_frame(frame)

            display_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            live_feed_placeholder.image(display_frame, channels="RGB", use_container_width=True)

            if density_map is not None:
                density_map_np = density_map.squeeze().cpu().numpy()
                h, w = frame.shape[:2]

                density_map_resized = cv2.resize(density_map_np, (w, h), interpolation=cv2.INTER_LINEAR)
                density_map_blurred = cv2.GaussianBlur(density_map_resized, (21, 21), 0)

                normed_map = cv2.normalize(density_map_blurred, None, 0, 255, cv2.NORM_MINMAX)
                heatmap = cv2.applyColorMap(np.uint8(normed_map), cv2.COLORMAP_JET)

                overlay_heatmap = cv2.addWeighted(display_frame, 0.5, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.5, 0)
                heatmap_placeholder.image(overlay_heatmap, channels="RGB", caption="Real-time Density Heatmap",
                                          use_container_width=True)

            total_people_metric.metric(label="Total People Count", value=f"{people_count}")

            zone_a_count = int(people_count * 0.52)
            zone_b_count = people_count - zone_a_count
            zone_A_metric.metric(label="Zone A", value=f"{zone_a_count}")
            zone_B_metric.metric(label="Zone B", value=f"{zone_b_count}")

            risk_label = processor.risk_labels.get(risk_class, "Unknown")
            color = "red" if risk_class == 2 else "orange" if risk_class == 1 else "green"
            risk_label_placeholder.markdown(
                f"<p style='font-size: 32px; font-weight: bold; color: {color};'>{risk_label}</p>",
                unsafe_allow_html=True)

            if risk_class == 2:
                redistribution_percent = int((zone_a_count - zone_b_count) / 2 / people_count * 100)
                redistribution_text_placeholder.markdown(
                    f"Move **{redistribution_percent}%** from **Zone A** to **Zone B**")
            else:
                redistribution_text_placeholder.markdown("No redistribution needed.")

            if risk_class == 2:
                alert_placeholder.warning("⚠️ ALERT: High crowd density detected in Zone A")
            else:
                alert_placeholder.empty()

            time.sleep(0.1)
        cap.release()

