# train.py
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import sklearn

print("Using scikit-learn version:", sklearn.__version__)

# Load data
URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(URL)

# ----------------------------
# Exercise 1: Linear Regression
# ----------------------------
X1 = df[["100g_USD"]]
y1 = df["rating"]

lr = LinearRegression()
lr.fit(X1, y1)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)
print("model_1.pickle saved")

# --------------------------------------------
# Exercise 2: DecisionTree on 100g_USD and roast
# --------------------------------------------
def roast_category(roast):
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4,
    }
    # Unknown/missing -> np.nan; we'll fill later
    return mapping.get(roast, np.nan)

# Map roast IN PLACE to numeric and fill missing with -1
df["roast"] = df["roast"].apply(roast_category).astype(float)
df["roast"] = df["roast"].fillna(-1.0)

# Build features with exact column names the grader expects
X2 = df[["100g_USD", "roast"]]
y2 = df["rating"]

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X2, y2)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)
print("model_2.pickle saved")

# Optional quick check that feature names match
try:
    print("feature_names_in_:", dtr.feature_names_in_)
except Exception:
    pass
