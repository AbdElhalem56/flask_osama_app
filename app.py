from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("theModel.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("🟡 Received from client:", data)

    J = data.get("day_of_year")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    tmax = data.get("tmax")

    if None in [latitude, longitude, J, tmax]:
        return jsonify({"error": "Missing one or more required fields"}), 400

    try:
        # ترتيب الأعمدة يطابق ترتيب التدريب
        input_data = np.array([[tmax, latitude, longitude, J]])
        prediction = model.predict(input_data)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
