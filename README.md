
# CodeX Beverage Price Range Prediction

An end-to-end machine learning project that predicts beverage price ranges based on customer demographics, preferences, and behavioral patterns. The system includes data preprocessing, feature engineering, model training, hyperparameter optimization, and a complete deployment-ready inference pipeline with a Streamlit web application.

---

## 🚀 Project Highlights
- Developed a robust ML model (XGBoost) for beverage price range prediction.
- Engineered advanced features such as **CF-AB Score**, **ZAS Score**, and **BSI Index** to improve predictive accuracy.
- Implemented automated hyperparameter tuning using **Optuna**.
- Built a clean and modular **inference pipeline** for production readiness.
- Designed an interactive **Streamlit web app** for real-time predictions.
- Integrated encoders, scalers, and saved ML artifacts to ensure reproducibility.

---

## 🧠 Tech Stack
- **Python**
- **Machine Learning**: XGBoost, RandomForest, Logistic Regression, SVM  
- **Hyperparameter Tuning**: Optuna  
- **Model Tracking**: MLflow (optional)  
- **Deployment**: Streamlit  
- **Preprocessing**: LabelEncoder, OneHotEncoder, StandardScaler  

---

## 📁 Project Structure
```
CodeX_Beverage/
│
├── App/
│   └── Main.py                 # Streamlit UI
│
├── src/
│   └── inference_pipeline.py   # Core inference pipeline
│
├── Artifacts/
│   ├── encoders/
│   │   ├── train_label_encoders.pkl
│   │   ├── onehot_encoder.pkl
│   │   └── scaler.pkl
│   └── models/
│       └── xgboost.pkl
│
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the App

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Streamlit App**
```bash
streamlit run App/Main.py
```

---

## 🧪 Prediction Workflow
1. User enters demographic + consumption behavior inputs.
2. Backend converts:
   - Age → Age Group  
   - Frequency + Awareness → CF-AB Score  
   - Zone + Income → ZAS Score  
   - Brand behavior → BSI  
3. Encoders & scaler transform the data.
4. XGBoost model predicts the **Beverage Price Range**.

---

## 📌 Key Features
- Modular production-ready ML pipeline  
- Real-time scoring web app  
- Fully reproducible predictions  
- Clean separation of UI, logic, and ML artifacts  

---

## 📄 License
This project is open for personal and educational use.

---

## ⭐ Contribution
Feel free to fork, improve, and customize the project.

