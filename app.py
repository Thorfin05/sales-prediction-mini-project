# app.py

from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

# Load dataset to get average
data = pd.read_csv("data/Train.csv")
average_sales = round(data['Item_Outlet_Sales'].mean(), 2)

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model/sales_model.pkl", "rb"))
le_fat = pickle.load(open("model/le_fat.pkl", "rb"))
le_type = pickle.load(open("model/le_type.pkl", "rb"))
le_outlet_size = pickle.load(open("model/le_outlet_size.pkl", "rb"))
le_outlet_location = pickle.load(open("model/le_outlet_location.pkl", "rb"))
le_outlet_type = pickle.load(open("model/le_outlet_type.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = None

    if request.method == "POST":
        try:
            # Capture form input
            Item_Weight = float(request.form['Item_Weight'])
            Item_Visibility = float(request.form['Item_Visibility'])
            Item_MRP = float(request.form['Item_MRP'])

            Item_Fat_Content = le_fat.transform([request.form['Item_Fat_Content']])[0]
            Item_Type = le_type.transform([request.form['Item_Type']])[0]
            Outlet_Size = le_outlet_size.transform([request.form['Outlet_Size']])[0]
            Outlet_Location_Type = le_outlet_location.transform([request.form['Outlet_Location_Type']])[0]
            Outlet_Type = le_outlet_type.transform([request.form['Outlet_Type']])[0]

            # Make prediction
            features = np.array([[Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type,
                                  Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type]])
            prediction = round(model.predict(features)[0], 2)

            # Preserve form data
            form_data = request.form

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
    "index.html",
    prediction=prediction,
    average_sales=average_sales,
    form_data=form_data
    )

if __name__ == "__main__":
    app.run(debug=True)
