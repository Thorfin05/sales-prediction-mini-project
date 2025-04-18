# 🛍️ Sales Prediction Web Application

This is a machine learning-based **Sales Prediction Web App** that forecasts item sales in a retail store using input features like item type, MRP, outlet size, and more. It uses a trained regression model and is deployed with a **Flask web framework**.

---

## 📌 Features

- Predicts item outlet sales using 8 input features.
- Interactive and clean web interface built using HTML, CSS, and Bootstrap.
- Dropdown menus for categorical inputs.
- Bar chart comparison of predicted vs average sales using Chart.js.
- Easy to use for non-technical users.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: pandas, scikit-learn, numpy, pickle, Flask
- **Frontend**: HTML, Bootstrap, Chart.js
- **Model**: Linear Regression

---

## 📁 Project Structure
sales-prediction-app/ 
- ├── data/ Train.csv # Dataset for training 
- ├── model/ trained_model.pkl # Trained ML model 
- ├── templates/ index.html # Frontend HTML page 
- ├── app.py # Flask backend 
- ── model_training.py # ML model training script 
- ── requirements.txt # Python dependencies 
- ──README.md # Project documentation


---

## ⚙️ Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Flask
- HTML, CSS, Bootstrap
- Chart.js

---

## 🧠 ML Features Used

The model predicts based on the following 8 features:

- `Item_Weight`
- `Item_Fat_Content`
- `Item_Visibility`
- `Item_Type`
- `Item_MRP`
- `Outlet_Size`
- `Outlet_Location_Type`
- `Outlet_Type`

---

## ✅ Setup Instructions
Create Virtual Environment
### Windows
python -m venv venv
venv\Scripts\activate

### macOS/Linux
python3 -m venv venv
source venv/bin/activate


### Install Dependencies
pip install -r requirements.txt

### Train the Model
python model_training.py

### Run the Web Application
python app.py

### Go to your browser and open:
http://127.0.0.1:5000/
