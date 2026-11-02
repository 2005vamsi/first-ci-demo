import unittest
import os
import json
import joblib


class TestMLPipeline(unittest.TestCase):

    def test_model_exists(self):
        self.assertTrue(
            os.path.exists("student_result_model.pkl")
        )

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        prediction = model.predict([[80, 90]])

        self.assertIn(int(prediction[0]), [0, 1])

    def test_accuracy_exists(self):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
