import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model():

    np.random.seed(42)

    # Create student dataset
    data = pd.DataFrame({
        "marks": np.random.randint(20, 101, 200),
        "attendance": np.random.randint(40, 101, 200)
    })

    # Generate result labels
    data["result"] = (
        0.7 * data["marks"] +
        0.3 * data["attendance"] >= 50
    ).astype(int)

    X = data[["marks", "attendance"]]
    y = data["result"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    # Train ML model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict results
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Save trained model
    joblib.dump(model, "student_result_model.pkl")

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

    # Save dataset
    data.to_csv("student_results.csv", index=False)

    print("Model Training Completed")
    print("Model Accuracy:", accuracy)


if __name__ == "__main__":
    train_model()
