# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

This project is a custom implementation of a Handwritten Digit Recognition system using Convolutional Neural Networks (CNNs). It allows users to draw digits on a virtual canvas and get real-time predictions.

**Author:** Your Name  
**Contact:** your.email@example.com

---

## 🧠 Dataset

The project uses the MNIST dataset, which contains:

- **60,000** labeled training images
- **10,000** labeled test images

Each image is:
- Grayscale
- 28x28 pixels
- Pixel values normalized between 0 and 1

---

## 🏗️ Model Architecture

The CNN model is built using Keras/TensorFlow and consists of:

- **Conv2D Layer 1**: 32 filters, (3x3) kernel, ReLU activation  
- **MaxPooling2D Layer 1**: (2x2) pool size  
- **Conv2D Layer 2**: 64 filters, (3x3) kernel, ReLU activation  
- **MaxPooling2D Layer 2**: (2x2) pool size  
- **Flatten Layer**: Converts 2D data to 1D  
- **Dense Layer 1**: 128 units, ReLU activation  
- **Dense Layer 2 (Output)**: 10 units, softmax activation (for digits 0–9)

---

## 📊 Model Evaluation

The model is trained and tested using MNIST data, and shows strong accuracy on unseen digit samples.

![Evaluation Graph](https://user-images.githubusercontent.com/97530517/232014919-390ab15f-67e6-4a63-bef3-9005d795135f.PNG)

---

## 🌐 Model Deployment

### Streamlit Front-End

A lightweight web UI using [Streamlit](https://streamlit.io/) for digit prediction:

- 🖌️ Draw any digit (0–9) using your mouse/touchpad
- 📤 The digit is processed and sent to the model
- ✅ The predicted digit is displayed in real time

📎 Sample App Screenshot:
![UI Example](https://user-images.githubusercontent.com/97530517/232018256-94749378-9d7b-4b33-a0a9-376bd2862392.PNG)

![Canvas](https://user-images.githubusercontent.com/97530517/232014753-7cd8a16c-1b42-4a5c-b67b-27998331ef8e.png)

---

## 🚀 How to Run

1. Clone the repo:
 
   git clone https://github.com/mlcoffeebreakwithaman/handdigit.git
   cd handwritten-digit-recognition

2. Set up a virtual environment and install dependencies:


python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Run the Streamlit app:

Run the Streamlit app:
streamlit run app.py