from flask import Flask, render_template, request

from predict import load_category_options, predict_fertilizer

app = Flask(__name__)
app.config["SECRET_KEY"] = "fertilizer-prediction-system"

NUMERIC_FIELDS = ["Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
CATEGORICAL_FIELDS = ["Soil Type", "Crop Type"]


@app.route("/", methods=["GET", "POST"])
def home():
    categories = load_category_options()
    prediction = None
    confidence = None
    error_message = None
    form_data = {field: "" for field in NUMERIC_FIELDS + CATEGORICAL_FIELDS}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            for field in NUMERIC_FIELDS:
                value = form_data.get(field, "").strip()
                if not value:
                    raise ValueError(f"{field} is required.")
                numeric_value = float(value)
                if numeric_value < 0:
                    raise ValueError(f"{field} must be non-negative.")
                form_data[field] = numeric_value

            for field in CATEGORICAL_FIELDS:
                value = form_data.get(field, "").strip()
                if not value:
                    raise ValueError(f"{field} is required.")
                if value not in categories.get(field, []):
                    raise ValueError(f"{field} must be one of the provided options.")

            prediction_result = predict_fertilizer(form_data)
            prediction = prediction_result.get("prediction")
            confidence = prediction_result.get("confidence")
        except ValueError as exc:
            error_message = str(exc)
        except Exception as exc:
            error_message = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error_message=error_message,
        form_data=form_data,
        categories=categories,
    )


if __name__ == "__main__":
    app.run(debug=True)
