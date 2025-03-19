import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
import keras

# Load the saved DeepFake Detection model
model_deepfake = load_model('inceptionNet_model.h5')

# Define constants for DeepFake Detection
img_size = 224
max_seq_length = 20
num_features = 2048

# Load the InceptionV3 model with pre-trained weights on ImageNet
feature_extractor = keras.applications.InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(img_size, img_size, 3)
)

# Function to crop center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Function to load a video
def load_video(path, max_frames=0, resize=(img_size, img_size)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Function to prepare a single video for prediction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_seq_length,), dtype="bool")
    frame_features = np.zeros(shape=(1, max_seq_length, num_features), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(max_seq_length, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

# Function to predict whether the video is fake or real
def sequence_prediction(path):
    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames)
    prediction = model_deepfake.predict([frame_features, frame_mask])
    return np.squeeze(prediction)  # Ensure prediction is a scalar

# Streamlit application
def main():
    st.title('DeepFake Detection')
    st.image("image.png")

    # DeepFake Detection
    st.title('DeepFake Detection')

    # File uploader for DeepFake Detection
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_video is not None:
        # Check if the uploaded file is an MP4 video
        if uploaded_video.type == "video/mp4":
            # Save the uploaded video to a temporary file
            video_path = "temp.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

            # Display the uploaded video
            st.video(video_path)

            # Predict button
            if st.button('Predict'):
                # Predict whether the video is fake or real
                prediction = sequence_prediction(video_path)

                # Display the prediction result
                if prediction >= 0.5:
                    st.write('<span style="color:red">The predicted class of the video is FAKE</span>', unsafe_allow_html=True)
                else:
                    st.write('<span style="color:blue">The predicted class of the video is REAL</span>', unsafe_allow_html=True)
        else:
            st.error("Please upload a valid MP4 video file.")

if __name__ == '__main__':
    main()
