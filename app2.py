import cv2
import streamlit as st
import tempfile
import numpy as np

st.set_page_config(page_title="Lane Detection App", layout="wide")
st.title("🚗 Real-Time Lane Detection")

option = st.sidebar.radio("Choose input source:", ("Upload Video", "Camera"))

@st.cache_resource
class LaneDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height = frame.shape[0]
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (frame.shape[1], height),
            (frame.shape[1], int(height*0.6)),
            (0, int(height*0.6))
        ]])
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=150)
        line_img = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
        return combo

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = detector.detect(frame)
        stframe.image(output, channels=\"BGR\", use_container_width=True)
        time.sleep(0.03)  # ~30 FPS
    cap.release()


if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a road video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        detector = LaneDetector()
        process_video(tfile.name, detector)

elif option == "Camera":
    st.warning("Camera access only works when run locally via Streamlit or Render Live App.")
    run = st.checkbox("Start Camera")
    if run:
        cap = cv2.VideoCapture(0)
        detector = LaneDetector()
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output = detector.detect(frame)
            stframe.image(output, channels="BGR", use_column_width=True)
        cap.release()
