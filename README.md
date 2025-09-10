# Breast_Cancer_Detection
🎯 Automated Breast Cancer Cell Detection using Deep Learning This capstone project leverages deep learning models to accurately classify breast cancer histopathological images. Trained on a publicly available dataset and deployed using Streamlit, the application offers a clean, user-friendly interface for real-time cancer prediction.

# 🧠 Breast Cancer Cell Classification using Deep Learning

A deep learning-based web application that detects breast cancer from histopathological images using a custom CNN and ResNet50 model. This project is deployed via Streamlit.

## 🚀 Live Demo
👉 [Click here to try the deployed app](https://mini-project-miniproject-g3gazp4ukngvvyiuhxl88l.streamlit.app/)

---

## 📌 Features

- 🔬 Classifies histopathological images as **Benign** or **Malignant**
- 🧠 Trained using both a **custom CNN** and **ResNet50 (transfer learning)**
- 🖼️ Accepts image uploads and displays predictions instantly
- 🌐 Hosted using **Streamlit** — serverless and fast
- 📊 Built with simplicity and accessibility in mind

---

## 📁 Project Structure

📁 Detect_BreastCancer.ipynb # Data loading, preprocessing, evaluation
📁 train_ResNet50_32_20k.ipynb # ResNet50 training notebook
📁 app.py # Streamlit app code (UI + model inference)
📁 requirements.txt # Dependencies for running the app


---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV & NumPy**
- **Matplotlib, Seaborn (visualization)**
- **Streamlit** (for web UI and deployment)

---

## 🧪 Dataset

- **BreaKHis**: Breast Cancer Histopathological Image Classification Dataset
- Includes 7,909 images across 8 tumor subclasses under 4 magnification levels (40X, 100X, 200X, 400X)
- Classes: **Benign** and **Malignant**
- Source: [Kaggle - BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

---

## 📈 Model Performance

| Model         | Accuracy | Notes                          |
|---------------|----------|---------------------------------|
| ResNet50      | ~95%     | Transfer learning model with fine-tuning |

---

## 👨‍🎓 Academic Context

- 🎓 **Mini Project**  
- 🏫 **Alvas Institute of Engineering and Technology**   
- 📆 Third Year B.E (Computer Science and Design) — Class of 2026  
- 👨‍💻 Developed by **Likith G** and **Team**

---

## 🧑‍💻 How to Run Locally

  1. **Clone the repository**
     ```bash
     git clone https://github.com/your-username/mini-project.git
     cd mini-project
     
  2. **Create a virtual environment (optional but recommended)**
    
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
  
  3. **Install dependencies**
    
    pip install -r requirements.txt
  
  4. **Run the Streamlit app**
   
    streamlit run app.py

📦 requirements.txt

streamlit  
tensorflow  
opencv-python  
numpy  
pillow  
matplotlib  
