import joblib
import pandas as pd
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fertilizer_model.pkl"
ENCODERS_PATH = BASE_DIR / "models" / "label_encoders.pkl"
TARGET_ENCODER_PATH = BASE_DIR / "models" / "target_encoder.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "Fertilizer Prediction.csv"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the trained model, feature encoders, scaler, and target encoder."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        feature_names = bundle.get("feature_names", [])
        target_encoder = bundle.get("target_encoder")
    else:
        model = bundle
        scaler = None
        feature_names = []
        target_encoder = None

    encoders = joblib.load(ENCODERS_PATH) if ENCODERS_PATH.exists() else {}
    if target_encoder is None and TARGET_ENCODER_PATH.exists():
        target_encoder = joblib.load(TARGET_ENCODER_PATH)

    return model, scaler, feature_names, encoders, target_encoder


def load_category_options():
    """Read the available soil and crop categories from the dataset."""
    if not DATASET_PATH.exists():
        return {"Soil Type": [], "Crop Type": []}

    df = pd.read_csv(DATASET_PATH)
    df.columns = [col.strip() for col in df.columns]
    if "Temparature" in df.columns:
        df.rename(columns={"Temparature": "Temperature"}, inplace=True)
    if "Humidity " in df.columns:
        df.rename(columns={"Humidity ": "Humidity"}, inplace=True)

    return {
        "Soil Type": sorted(df["Soil Type"].astype(str).dropna().unique().tolist()),
        "Crop Type": sorted(df["Crop Type"].astype(str).dropna().unique().tolist()),
    }


def prepare_features(input_data, feature_names, encoders, scaler):
    """Convert user input into the numeric feature matrix expected by the model."""
    feature_row = {}
    for feature_name in feature_names:
        value = input_data.get(feature_name)
        if feature_name in encoders:
            feature_row[feature_name] = encoders[feature_name].transform([str(value).strip()])[0]
        else:
            feature_row[feature_name] = float(value)

    features_df = pd.DataFrame([feature_row], columns=feature_names)
    if scaler is None:
        return features_df.to_numpy(dtype=float)
    return scaler.transform(features_df)


def predict_fertilizer(input_data):
    """Predict the fertilizer label for a single user input sample and return confidence."""
    model, scaler, feature_names, encoders, target_encoder = load_artifacts()
    features = prepare_features(input_data, feature_names, encoders, scaler)
    prediction_index = model.predict(features)[0]

    prediction_label = None
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        max_index = int(probabilities.argmax())
        confidence = float(probabilities[max_index]) * 100
        if target_encoder is not None:
            prediction_label = target_encoder.inverse_transform([prediction_index])[0]
        else:
            prediction_label = str(prediction_index)
    else:
        if target_encoder is not None:
            prediction_label = target_encoder.inverse_transform([prediction_index])[0]
        else:
            prediction_label = str(prediction_index)
        confidence = None

    return {
        "prediction": prediction_label,
        "confidence": confidence,
    }
