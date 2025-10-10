import joblib
import pandas as pd
from ai_wrapper import generate_explanation
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)

CORS(app)

scaler = joblib.load("./pkl/Scaler.pkl")

encoders = {}
for col in [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal",
]:
    encoders[col] = joblib.load(f"./pkl/Le{col}.pkl")

le_target = joblib.load("./pkl/Letarget.pkl")

models = {
    "rf": joblib.load("./best_model/best_model_RandomForest.joblib"),
    "knn": joblib.load("./best_model/best_model_KNN.joblib"),
    "logreg": joblib.load("./best_model/best_model_LogisticRegression.joblib"),
    "xgb": joblib.load("./best_model/best_model_XGB.joblib"),
}


numerical_col = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

categorical_col = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

model_feature_order = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        cat_data = {}
        for col in categorical_col:
            if col not in data:
                return jsonify({"error": f"Missing categorical field '{col}'"}), 400
            value = data[col]
            try:
                cat_data[col] = encoders[col].transform([value])[0]
            except Exception:
                return jsonify(
                    {"error": f"Invalid value '{value}' for column '{col}'"}
                ), 400

        try:
            num_data = {col: float(data[col]) for col in numerical_col}
        except KeyError as e:
            return jsonify({"error": f"Missing numerical field {str(e)}"}), 400
        except ValueError as e:
            return jsonify({"error": f"Invalid numerical value: {str(e)}"}), 400

        df = pd.DataFrame([{**num_data, **cat_data}])

        df[numerical_col] = scaler.transform(df[numerical_col])

        model_name = data.get("selected_model")
        if not model_name or model_name not in models:
            return jsonify({"error": "Invalid or missing 'selected_model'"}), 400
        model = models[model_name]

        df = df[model_feature_order]

        y_pred_encoded = model.predict(df)[0]

        y_pred_decoded = le_target.inverse_transform([y_pred_encoded])[0]

        prediction = "No_deasease" if y_pred_decoded == "0" else "Deasease_detected"

        filtered_data = {k: v for k, v in data.items() if k != "selected_model"}
        ai_response = generate_explanation(prediction, filtered_data)

        return jsonify(
            {
                "prediction_encoded": str(y_pred_encoded),
                "prediction_decoded": str(y_pred_decoded),
                "model_used": model_name,
                "meaning": prediction,
                "Airesponse": ai_response,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
