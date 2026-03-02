import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from PIL import Image
import time
import os
import numpy as np
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="Vehicle Accident Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2d7cd6;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f538d 0%, #2d7cd6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .detection-box {
        border: 2px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class AccidentDetectionSystem:
    def __init__(self):
        self.model = None
        self.class_list = []
        self.load_model()
        self.load_classes()

        # Create evidence directory
        if not os.path.exists("accident_evidence"):
            os.makedirs("accident_evidence")

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO('best.pt')
        except Exception as e:
            st.error(f"Error loading model: {e}")

    def load_classes(self):
        """Load class list"""
        try:
            with open("coco1.txt", "r") as f:
                self.class_list = f.read().split("\n")
        except FileNotFoundError:
            st.warning("Class list file 'coco1.txt' not found!")
            self.class_list = []

    def detect_accidents(self, frame, confidence_threshold):
        """Detect accidents in frame"""
        results = self.model.predict(
            frame,
            conf=confidence_threshold,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
        )
        detections = []
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls < len(self.class_list):
                    class_name = self.class_list[cls]
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })

                    # Draw bounding box
                    color = (0, 0, 255) if 'accident' in class_name.lower() else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cvzone.putTextRect(
                        annotated_frame,
                        f"{class_name} {conf:.2f}",
                        (x1, y1 - 10),
                        scale=1,
                        thickness=2,
                        colorR=(0, 255, 0)
                    )

        return annotated_frame, detections

    def save_evidence(self, frame, detection_info):
        """Save accident evidence"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accident_evidence/accident_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = AccidentDetectionSystem()
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []

# Main UI
st.markdown('<p class="main-header">🚗 Real-Time Vehicle Accident Detection System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Detection settings
    st.subheader("Detection Configuration")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Minimum confidence score for detection (higher = fewer false positives)"
    )

    save_evidence = st.checkbox(
        "Save Evidence",
        value=True,
        help="Save images when accidents are detected"
    )

    st.divider()

    # Camera settings
    st.subheader("📹 Camera Settings")
    camera_location = st.text_input(
        "Camera Location",
        value="Main Entrance",
        help="Identify the camera location"
    )

    st.divider()

    # Notification settings
    st.subheader("🔔 Notifications")
    enable_notifications = st.checkbox("Enable Notifications", value=False)

    if enable_notifications:
        st.info("💡 Configure Twilio/WhatsApp settings for notifications")
        twilio_sid = st.text_input(
            "Twilio Account SID",
            type="password",
            help="Your Twilio Account SID"
        )
        twilio_token = st.text_input(
            "Twilio Auth Token",
            type="password",
            help="Your Twilio Auth Token"
        )
        twilio_phone = st.text_input(
            "Twilio Phone Number",
            placeholder="+1XXXXXXXXXX",
            help="Your Twilio phone number"
        )
        recipient_phone = st.text_input(
            "Recipient Phone Number",
            placeholder="+1XXXXXXXXXX",
            help="Phone number to receive notifications"
        )

    st.divider()

    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Detections", st.session_state.detection_count)

    if st.button("🗑️ Clear Statistics"):
        st.session_state.detection_count = 0
        st.session_state.detection_log = []
        st.success("Statistics cleared!")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📷 Live Detection", "📁 Upload Image/Video", "📋 Detection Log"])

with tab1:
    st.subheader("Live Camera Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        run_camera = st.checkbox("Start Camera Detection", value=False)

        if run_camera:
            st.info("📹 Camera is active. Press 'q' in the camera window to stop.")

            # Camera capture
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_btn = st.button("Stop Detection")

            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break

                # Detect accidents
                annotated_frame, detections = st.session_state.detector.detect_accidents(
                    frame, confidence_threshold
                )

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Log detections
                if detections:
                    st.session_state.detection_count += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    for det in detections:
                        st.session_state.detection_log.append({
                            'timestamp': timestamp,
                            'location': camera_location,
                            'class': det['class'],
                            'confidence': det['confidence']
                        })

                    if save_evidence:
                        filename = st.session_state.detector.save_evidence(frame, detections)
                        st.success(f"🚨 Accident detected! Evidence saved: {filename}")

                time.sleep(0.1)

            cap.release()

    with col2:
        st.info("💡 **Instructions**\n\n1. Adjust confidence threshold\n2. Enable camera detection\n3. System will alert on accidents\n4. Evidence is automatically saved")

with tab2:
    st.subheader("Upload Image or Video for Detection")

    upload_type = st.radio("Select upload type:", ["Image", "Video"])

    if upload_type == "Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect accidents"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Detection Results")

                # Convert PIL to OpenCV format
                img_array = np.array(image)
                frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Detect
                annotated_frame, detections = st.session_state.detector.detect_accidents(
                    frame, confidence_threshold
                )

                # Display results
                rgb_result = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_result, use_container_width=True)

                if detections:
                    st.success(f"✅ Detected {len(detections)} object(s)")

                    for i, det in enumerate(detections, 1):
                        st.write(f"**Detection {i}:** {det['class']} ({det['confidence']:.2%})")

                    if save_evidence:
                        filename = st.session_state.detector.save_evidence(frame, detections)
                        st.info(f"Evidence saved: {filename}")
                else:
                    st.warning("No accidents detected")

    else:  # Video
        uploaded_video = st.file_uploader(
            "Choose a video...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to detect accidents"
        )

        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())

            st.video(tfile.name)

            if st.button("🔍 Analyze Video"):
                st.info("Processing video... This may take a while.")

                cap = cv2.VideoCapture(tfile.name)
                frame_count = 0
                detections_in_video = []

                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Process every 10th frame to speed up
                    if frame_count % 10 == 0:
                        annotated_frame, detections = st.session_state.detector.detect_accidents(
                            frame, confidence_threshold
                        )

                        if detections:
                            detections_in_video.append({
                                'frame': frame_count,
                                'detections': detections
                            })

                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()

                st.success(f"✅ Video analysis complete!")
                st.write(f"Total frames analyzed: {frame_count}")
                st.write(f"Accidents detected in: {len(detections_in_video)} frames")

                if detections_in_video:
                    st.warning("🚨 Accidents detected in the video!")
                    for det in detections_in_video[:5]:  # Show first 5
                        st.write(f"Frame {det['frame']}: {len(det['detections'])} object(s)")

with tab3:
    st.subheader("Detection History")

    if st.session_state.detection_log:
        df = pd.DataFrame(st.session_state.detection_log)

        # Display as dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Log as CSV",
            data=csv,
            file_name=f"accident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Statistics
        st.subheader("📊 Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Detections", len(df))
        with col2:
            if 'class' in df.columns:
                most_common = df['class'].mode()[0] if not df['class'].mode().empty else "N/A"
                st.metric("Most Common Class", most_common)
        with col3:
            if 'confidence' in df.columns:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
    else:
        st.info("No detections recorded yet. Start detecting accidents to see logs here.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚗 Vehicle Accident Detection System | Powered by YOLOv8</p>
    <p style='font-size: 0.8rem;'>⚠️ For demonstration purposes only. Always verify with proper authorities.</p>
</div>
""", unsafe_allow_html=True)
