import joblib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "dataset" / "Fertilizer Prediction.csv",
    BASE_DIR / "dataset" / "fertilizer.csv",
]
MODEL_PATH = BASE_DIR / "models" / "fertilizer_model.pkl"
ENCODERS_PATH = BASE_DIR / "models" / "label_encoders.pkl"
TARGET_ENCODER_PATH = BASE_DIR / "models" / "target_encoder.pkl"


def load_dataset():
    """Load the fertilizer dataset from the available local CSV files."""
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("No fertilizer dataset found in the dataset folder")

    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]

    if "Temparature" in df.columns:
        df.rename(columns={"Temparature": "Temperature"}, inplace=True)
    if "Humidity " in df.columns:
        df.rename(columns={"Humidity ": "Humidity"}, inplace=True)

    target_column = "Label" if "Label" in df.columns else "Fertilizer Name"
    if target_column not in df.columns:
        raise ValueError(f"Expected target column not found. Available columns: {list(df.columns)}")

    print(f"Dataset loaded from {data_path.name} with {len(df)} rows")
    return df, target_column


def preprocess_data(df, target_column):
    """Encode categorical features and target labels for model training."""
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column]

    feature_encoders = {}
    for column in X.columns:
        if X[column].dtype == "object" or pd.api.types.is_string_dtype(X[column]):
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            feature_encoders[column] = encoder

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, feature_encoders, label_encoder


def evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple classifiers and return their evaluation metrics."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        results.append(metrics)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return results_df, trained_models, scaler


def plot_feature_importance(model, feature_names):
    """Plot and save a feature importance chart for the Random Forest model."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance Values:")
    print(feature_importance_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = BASE_DIR / "static" / "feature_importance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nFeature importance chart saved to {output_path}")


def train_model():
    """Train and compare multiple classifiers, then save the best-performing model."""
    df, target_column = load_dataset()
    X, y, feature_encoders, target_encoder = preprocess_data(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results_df, trained_models, scaler = evaluate_models(X_train, X_test, y_train, y_test)

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    if best_model_name != "Random Forest":
        print("\nRandom Forest was not selected as the best model; feature importance chart will be generated using the trained Random Forest model.")

    random_forest_model = trained_models["Random Forest"]
    X_train_scaled = scaler.fit_transform(X_train)
    random_forest_model.fit(X_train_scaled, y_train)
    plot_feature_importance(random_forest_model, X.columns.tolist())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "target_encoder": target_encoder,
    }
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(feature_encoders, ENCODERS_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    print(f"\nBest model: {best_model_name}")
    print(f"Model bundle saved to {MODEL_PATH}")
    print(f"Feature encoders saved to {ENCODERS_PATH}")


if __name__ == "__main__":
    train_model()
