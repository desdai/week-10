import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import sklearn

print("Using scikit-learn version:", sklearn.__version__)

# Remove old model files if they exist
for fname in ["model_1.pickle", "model_2.pickle"]:
    if os.path.exists(fname):
        os.remove(fname)
        print("Removed old", fname)

# Helper function to convert roast to numeric category
def roast_category(roast):
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast, np.nan)

# Load the dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Exercise 1 — Linear Regression
X1 = df[["100g_USD"]]
y1 = df["rating"]

lr = LinearRegression()
lr.fit(X1, y1)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)
print("model_1.pickle saved successfully.")

# Exercise 2 — Decision Tree Regressor
df["roast_cat"] = df["roast"].apply(roast_category)

X2 = df[["100g_USD", "roast_cat"]]
y2 = df["rating"]

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X2, y2)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)
print("model_2.pickle saved successfully.")

# Test the model immediately
print("Testing freshly trained model_2...")

df_test = pd.DataFrame([
    [10.00, 1],
    [15.00, 3],
    [8.50, np.nan],
    [12.00, -99],
    [20.00, 2938.24]
], columns=["100g_USD", "roast_cat"])

preds = dtr.predict(df_test)
print("Predictions:", preds)

print("All models trained and tested successfully under sklearn", sklearn.__version__)
