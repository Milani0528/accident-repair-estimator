# Accident Repair Estimator 🚗🔧

A **Flask backend + HTML/CSS frontend** demo application for estimating accident repair costs using a deep learning model.  
This project was developed as part of a **Final Year Project (FYP)** in the vehicle insurance domain.

---

## 📌 Features
- **Image Upload** → Upload a vehicle accident image (front, rear, or side).  
- **Deep Learning Prediction** → Classifies the uploaded image into one of 6 categories:  
  - `Front_View`  
  - `Non_Front_View`  
  - `Non_Rear_Bumper`  
  - `Non_Sedan_Side_View`  
  - `Rear_Bumper`  
  - `Sedan_Side_View`  
- **Repair Cost Estimate** → Provides a **rough repair quotation** in LKR.  
- **Web Interface** → Clean and responsive HTML/CSS frontend.  
- **Flask Backend** → Connects the deep learning `.h5` model with the frontend.

---


---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/accident-repair-estimator.git
cd accident-repair-estimator


cd backend
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)

pip install -r requirements.txt
python app.py


curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@/path/to/test.jpg"


{
  "label": "Front_View",
  "confidence": 0.873,
  "estimate_lkr": 128000,
  "currency": "LKR",
  "message": "Rough repair estimate"
}


📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

👨‍💻 Author

Milani Vichara
Final Year Project – Accident Vehicle Image Prediction & Repair Estimation
