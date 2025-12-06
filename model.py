import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load CSV
df = pd.read_csv("retail_store_inventory.csv")

# Drop unused columns
df.drop(columns=['Date', 'Store ID', 'Product ID'], inplace=True)

# Rename columns
df = df.rename(columns={
    'Inventory Level': 'Inventory',
    'Units Sold': 'Sales',
    'Units Ordered': 'Orders',
    'Demand Forecast': 'Demand',
    'Weather Condition': 'Weather',
    'Holiday/Promotion': 'Promotion',
    'Competitor Pricing': 'Competitor Price'
})

# Encode categorical columns
label_encoders = {}
cat_cols = ['Category', 'Region', 'Weather', 'Seasonality', 'Promotion']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# FINAL TRAINING FEATURE ORDER (IMPORTANT)
final_features = [
    'Category',
    'Region',
    'Inventory',
    'Orders',
    'Discount',
    'Price',
    'Weather',
    'Promotion',
    'Competitor Price',
    'Seasonality'
]

X = df[final_features]
y = df['Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model and encoders saved successfully!")
