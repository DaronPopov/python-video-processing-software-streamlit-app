import os
import streamlit as st
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from concurrent.futures import ThreadPoolExecutor


# --- Video Processing Functions ---
# --- Video Processing Functions ---
def apply_processing(frame, processing_options, prev_frame=None):
    if processing_options['motion_detection']:
        frame = detect_motion_single_frame(frame)

    if processing_options['face_detection']:
        frame = detect_faces_single_frame(frame)

    if processing_options['optical_flow'] and prev_frame is not None:
        frame = compute_optical_flow_single_frame(prev_frame, frame)

    if processing_options['edge_detection']:
        frame = detect_edges_single_frame(frame)

    if processing_options['ocr']:
        frame = detect_text_single_frame(frame)

    return frame


def detect_motion_single_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    if not hasattr(detect_motion_single_frame, "prev_frame"):
        detect_motion_single_frame.prev_frame = gray_frame
    frame_delta = cv2.absdiff(detect_motion_single_frame.prev_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    detect_motion_single_frame.prev_frame = gray_frame
    return frame


def detect_faces_single_frame(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


def compute_optical_flow_single_frame(prev_frame, frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_frame)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


def detect_edges_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return frame


def detect_text_single_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Preprocess for better OCR accuracy
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray)
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


# Streamlit UI setup
st.set_page_config(layout="wide", page_title="Real-Time Video Processing", page_icon="🎥")

st.markdown("""
   <style>
       .main {
           background-color: #0e1117;
           color: #e1e1e1;
       }
       .sidebar .sidebar-content {
           background-color: #1e2126;
       }
       .stButton>button {
           background-color: #3b4048;
           color: #e1e1e1;
       }
       .stFileUploader, .stProgress {
           color: #e1e1e1;
       }
       .stSidebar .css-1y2b0x9 {
           background-color: #1e2126;
       }
       .css-1n76uvr, .css-1y2b0x9 {
           border-radius: 10px;
       }
       .stFileUploader {
           width: 400px;
           margin: 0 auto;
           transition: transform 0.3s ease-in-out;
       }
       .stFileUploader:hover {
           transform: scale(1.05);
       }
   </style>
""", unsafe_allow_html=True)

st.title("Real-Time Video Processing")

# Processing Options and File Upload
st.sidebar.header("Processing Options")

with st.sidebar.expander("Choose Processing to be Performed", expanded=True):
    motion_checkbox = st.checkbox("Motion Detection")
    face_detection_checkbox = st.checkbox("Face Detection")
    optical_flow_checkbox = st.checkbox("Optical Flow Analysis")
    edge_detection_checkbox = st.checkbox("Edge Detection")
    ocr_checkbox = st.checkbox("OCR (Text Detection)")

# Single area for video display
video_display = st.empty()

# Function to process and display uploaded video
def process_and_display_video(video_path, progress_output, processing_options):
    progress_output.text("Starting video processing...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        progress_output.text("Error: Unable to open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stframe = st.empty()  # Placeholder for video frames

    processed_frames = []

    def process_single_frame(i, frame, prev_frame):
        return apply_processing(frame, processing_options, prev_frame)

    with ThreadPoolExecutor() as executor:
        prev_frame = None
        future_to_frame = {executor.submit(process_single_frame, i, frame, prev_frame): (i, frame) for i, frame in enumerate(read_frames(cap))}

        for future in future_to_frame:
            i, frame = future_to_frame[future]
            processed_frame = future.result()
            processed_frames.append(processed_frame)
            prev_frame = processed_frame

            # Display the frame in Streamlit
            if i % 5 == 0:  # Update display less frequently
                _, buffer = cv2.imencode('.jpg', processed_frame)
                stframe.image(buffer.tobytes(), channels="BGR")

            if i % 20 == 0:
                progress_output.text(f"Processed {i + 1}/{frame_count} frames...")

    cap.release()
    progress_output.text("Video processing completed.")

    # Save processed video
    output_video_path = os.path.join('/mnt/data', 'processed_video.mp4')
    save_processed_video(output_video_path, processed_frames, fps, width, height)

    return output_video_path


def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def save_processed_video(output_path, frames, fps, width, height):
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


# Real-Time Video Stream Processing
def process_real_time_video(processing_options):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return

    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_processing(frame, processing_options, prev_frame)
        prev_frame = frame

        # Display the frame in Streamlit
        _, buffer = cv2.imencode('.jpg', frame)
        video_display.image(buffer.tobytes(), channels="BGR")

    cap.release()


# Process Video Button for uploaded video
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'], key="upload")

if uploaded_file is not None:
    video_path = os.path.join('/mnt/data', uploaded_file.name)
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    if st.button("Process Uploaded Video"):
        with st.spinner("Processing..."):
            progress_output = st.empty()
            processing_options = {
                'motion_detection': motion_checkbox,
                'face_detection': face_detection_checkbox,
                'optical_flow': optical_flow_checkbox,
                'edge_detection': edge_detection_checkbox,
                'ocr': ocr_checkbox
            }
            output_video_path = process_and_display_video(video_path, progress_output, processing_options)
            st.video(output_video_path)
            st.download_button("Download Processed Video", open(output_video_path, 'rb').read(), file_name="processed_video.mp4")


# Real-Time Video Processing Button
if st.button("Start Real-Time Video Processing"):
    with st.spinner("Processing..."):
        processing_options = {
            'motion_detection': motion_checkbox,
            'face_detection': face_detection_checkbox,
            'optical_flow': optical_flow_checkbox,
            'edge_detection': edge_detection_checkbox,
            'ocr': ocr_checkbox
        }
        process_real_time_video(processing_options)
