🌿 Plant Disease Detection

An AI-powered web application that detects plant diseases from leaf images using Deep Learning and provides disease information and crop-care recommendations.

🌱 Supported Crops

- Tomato – 10 classes
- Mango – 8 classes
- Rice – 10 classes

Total: 28 classes

🚀 Features

- Plant disease detection using a trained CNN model
- Image-based disease prediction
- Top 3 prediction results
- Disease information and severity estimation
- Grad-CAM visualization
- Weather information and forecast
- Crop-care recommendations
- PDF report generation
- Plant care chatbot

🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pillow
- ReportLab

📂 Project Structure

Plant-Disease-Detection/
│
├── main.py
├── model_training.py
├── requirements.txt
│
└── models/
    ├── plant_disease_model.keras
    └── class_names.json

⚙️ Installation

1. Clone the repository:

git clone https://github.com/gayathripandiyan29/Plant-Disease-Detection.git

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

streamlit run main.py

🎯 Objective

To help users identify plant diseases quickly through leaf image analysis and provide useful information to support better plant care.

👩‍💻 Developed By

Gayathri Pandiyan
