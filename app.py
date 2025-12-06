from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)   # ✅ fix: use __name__ not _name_

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Same order used in training
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read inputs
    category = request.form['category']
    region = request.form['region']
    weather = request.form['weather']
    seasonality = request.form['seasonality']
    promotion = request.form['promotion']

    competitor_price = float(request.form['competitor_price'])
    discount = float(request.form['discount'])
    inventory = float(request.form['inventory'])
    orders = float(request.form['orders'])
    price = float(request.form['price'])

    # Encode categorical
    input_data = {
        'Category': encoders['Category'].transform([category])[0],
        'Region': encoders['Region'].transform([region])[0],
        'Weather': encoders['Weather'].transform([weather])[0],
        'Seasonality': encoders['Seasonality'].transform([seasonality])[0],
        'Promotion': encoders['Promotion'].transform([promotion])[0],
        'Inventory': inventory,
        'Orders': orders,
        'Discount': discount,
        'Price': price,
        'Competitor Price': competitor_price
    }

    # Convert to dataframe
    df = pd.DataFrame([input_data])

    # Apply correct column order
    df = df[final_features]

    prediction = model.predict(df)[0]

    # ✅ render result.html instead of index.html
    return render_template(
        "result.html",
        prediction_text=f"Predicted Demand: {prediction:.2f}"
    )

if __name__ == "__main__":   # ✅ fix: use __name__ not _name_
    app.run(debug=True)