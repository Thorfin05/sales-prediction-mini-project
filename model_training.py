# model_training.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("data/Train.csv")

# Fill missing values
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Selected features
features = [
    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
    'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]
target = 'Item_Outlet_Sales'

X = data[features]
y = data[target]

# Label Encoding for categoricals
le_fat = LabelEncoder()
le_type = LabelEncoder()
le_outlet_size = LabelEncoder()
le_outlet_location = LabelEncoder()
le_outlet_type = LabelEncoder()

X['Item_Fat_Content'] = le_fat.fit_transform(X['Item_Fat_Content'])
X['Item_Type'] = le_type.fit_transform(X['Item_Type'])
X['Outlet_Size'] = le_outlet_size.fit_transform(X['Outlet_Size'])
X['Outlet_Location_Type'] = le_outlet_location.fit_transform(X['Outlet_Location_Type'])
X['Outlet_Type'] = le_outlet_type.fit_transform(X['Outlet_Type'])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model & encoders
pickle.dump(model, open("model/sales_model.pkl", "wb"))
pickle.dump(le_fat, open("model/le_fat.pkl", "wb"))
pickle.dump(le_type, open("model/le_type.pkl", "wb"))
pickle.dump(le_outlet_size, open("model/le_outlet_size.pkl", "wb"))
pickle.dump(le_outlet_location, open("model/le_outlet_location.pkl", "wb"))
pickle.dump(le_outlet_type, open("model/le_outlet_type.pkl", "wb"))

print("✅ Model trained and saved.")
