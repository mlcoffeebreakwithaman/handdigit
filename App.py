import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import time
import serial

# ------------------------------ #
# 🚀 Predict Digit Function
# ------------------------------ #
def predict_digit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]))
    return predicted_class, confidence

# ------------------------------ #
# ✉️ Send to Arduino (Optional)
# ------------------------------ #
def send_to_arduino_serial(digit, port='COM4', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(2)
        payload = str(digit).strip().encode()
        ser.write(payload)
        ser.close()
    except Exception as e:
        print(f"[Error] {e}")

# ------------------------------ #
# ✂️ Crop Utility for Webcam Image
# ------------------------------ #
def crop_image_interactive(image):
    st.image(image, caption="🖼 Original Webcam Image", use_container_width=True)
    st.markdown("🔲 Select a region to crop using the sliders:")

    h, w = image.shape[:2]
    x1 = st.slider("Left (X1)", 0, w - 1, 50)
    x2 = st.slider("Right (X2)", x1 + 1, w, w - 50)
    y1 = st.slider("Top (Y1)", 0, h - 1, 50)
    y2 = st.slider("Bottom (Y2)", y1 + 1, h, h - 50)

    if st.button("✂️ Crop Now"):
        cropped_img = image[y1:y2, x1:x2]
        st.image(cropped_img, caption="🖼 Cropped Image", use_container_width=True)
        return cropped_img
    return None

# ------------------------------ #
# 🎨 UI Layout
# ------------------------------ #
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("🔢 Handwritten Digit Recognizer")

mode = st.radio("Choose Input Mode:", ["🖌️ Draw on Canvas", "📷 Use Webcam"])

# ------------------------------ #
# 🎨 Mode 1: Draw on Canvas
# ------------------------------ #
if mode == "🖌️ Draw on Canvas":
    st.markdown("Draw a digit and click **Predict Now**")
    stroke_width = st.slider("✏️ Stroke Width", 1, 30, 15)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("✅ Predict Now (Canvas)"):
        if canvas_result.image_data is not None:
            img_array = np.array(canvas_result.image_data)
            image = Image.fromarray(img_array.astype("uint8"), "RGBA").convert("RGB")
            result, confidence = predict_digit(image)
            st.success(f"🎯 Predicted Digit: **{result}** (Confidence: {confidence:.2f})")
            send_to_arduino_serial(result)
        else:
            st.warning("⚠️ Please draw a digit before predicting.")

# ------------------------------ #
# 📷 Mode 2: Use Webcam
# ------------------------------ #
else:
    st.markdown("Make sure your webcam is connected. Position a digit (0–9) in view and press capture.")
    capture = st.button("📸 Capture from Webcam")

    if capture:
        cap = cv2.VideoCapture(0)
        st.info("📷 Capturing... Please hold your digit still.")
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cropped = crop_image_interactive(frame)
            if cropped is None:
                cropped = frame

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            roi = cv2.resize(thresh, (28, 28))
            roi_pil = Image.fromarray(roi)

            result, confidence = predict_digit(roi_pil)
            st.image(cropped, caption="🖼 Final Input for Prediction", use_container_width=True)
            st.success(f"🎯 Predicted Digit: **{result}** (Confidence: {confidence:.2f})")
            send_to_arduino_serial(result)
        else:
            st.error("❌ Failed to capture image from webcam.")

# ------------------------------ #
# ℹ️ Sidebar Info
# ------------------------------ #
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app supports:
- ✍️ Canvas drawing (mouse/touch)
- 📷 Webcam digit recognition
- ✂️ Interactive cropping
- 🎛 Adaptive thresholding
- 📈 Confidence score display

Built with Streamlit + TensorFlow + OpenCV
""")
